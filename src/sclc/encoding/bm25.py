from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import read_retrieval_units
from sclc.data.schema import RetrievalUnitRecord
from sclc.paths import encoding_dir, retrieval_unit_dir


def lexical_tokens(text: str, *, pattern: str, lowercase: bool) -> list[str]:
    """Tokenise text for the controlled BM25 baseline.

    No stemming, stop-word removal, query expansion, or generated context is
    applied. The exact same function will later be used for BM25 queries.
    """
    source = text.casefold() if lowercase else text
    return re.findall(pattern, source)


def _safe_document_name(document_id: str) -> str:
    digest = hashlib.sha1(document_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"document_{digest}.json.gz"


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _units_fingerprint(units: Sequence[RetrievalUnitRecord]) -> str:
    records = [
        {
            "retrieval_unit_id": unit.retrieval_unit_id,
            "document_id": unit.document_id,
            "analysis_set": unit.analysis_set,
            "unit_index": unit.unit_index,
            "span": [unit.span.start, unit.span.end],
            "text": unit.text,
        }
        for unit in sorted(
            units, key=lambda item: (item.document_id, item.unit_index, item.retrieval_unit_id)
        )
    ]
    return _fingerprint({"units": records})


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return cast(dict[str, Any], json.load(handle))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cached BM25 file {path} is invalid. Re-run with --overwrite."
        ) from exc


def _validate_cached_document(
    path: Path, *, expected_document_id: str, expected_unit_ids: Sequence[str]
) -> None:
    payload = _read_gzip_json(path)
    if payload.get("document_id") != expected_document_id:
        raise RuntimeError(
            f"Cached BM25 file {path} belongs to another document. Re-run with --overwrite."
        )
    unit_ids = [str(unit["retrieval_unit_id"]) for unit in payload.get("units", [])]
    if unit_ids != list(expected_unit_ids):
        raise RuntimeError(
            f"Cached BM25 file {path} contains different retrieval units. "
            "Re-run with --overwrite."
        )


def _write_gzip_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def build_bm25_encoding(
    config: AppConfig,
    *,
    chunk_size: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a resumable, fingerprinted lexical representation for BM25."""
    units_path = retrieval_unit_dir(config, chunk_size) / "continuous_units.jsonl"
    if not units_path.exists():
        raise FileNotFoundError(f"{units_path} does not exist. Run `sclc build-units` first.")

    units = list(read_retrieval_units(units_path))
    units_by_document: dict[str, list[RetrievalUnitRecord]] = defaultdict(list)
    for unit in units:
        units_by_document[unit.document_id].append(unit)

    configuration = {
        "condition": "bm25",
        "chunk_size_tokens": chunk_size,
        "segmentation_plan": "continuous",
        "lowercase": config.bm25.lowercase,
        "token_pattern": config.bm25.token_pattern,
        "k1": config.bm25.k1,
        "b": config.bm25.b,
        "input_fingerprint": _units_fingerprint(units),
    }
    fingerprint = _fingerprint(configuration)
    output_dir = encoding_dir(config, chunk_size) / "bm25"
    document_dir = output_dir / "documents"
    manifest_path = output_dir / "manifest.json"
    existing = _read_json(manifest_path)
    has_cached_files = document_dir.exists() and any(document_dir.glob("*.json.gz"))
    if has_cached_files and existing is None and not overwrite:
        raise RuntimeError(
            f"Cached BM25 files exist at {document_dir}, but the manifest is missing. "
            "Re-run with --overwrite."
        )
    if (
        existing is not None
        and existing.get("configuration_fingerprint") != fingerprint
        and not overwrite
    ):
        raise RuntimeError(
            "Cached BM25 encoding does not match the current configuration or retrieval "
            "units. Re-run with --overwrite."
        )
    if overwrite and document_dir.exists():
        for path in document_dir.glob("*.json.gz"):
            path.unlink()

    document_dir.mkdir(parents=True, exist_ok=True)
    manifest_documents: list[dict[str, Any]] = []
    total_units = 0
    skipped_documents = 0

    for document_id in sorted(units_by_document):
        document_units = sorted(
            units_by_document[document_id], key=lambda item: item.unit_index
        )
        filename = _safe_document_name(document_id)
        output_path = document_dir / filename

        if output_path.exists() and not overwrite:
            _validate_cached_document(
                output_path,
                expected_document_id=document_id,
                expected_unit_ids=[unit.retrieval_unit_id for unit in document_units],
            )
            skipped_documents += 1
            manifest_documents.append(
                {
                    "document_id": document_id,
                    "analysis_set": document_units[0].analysis_set,
                    "file": str(output_path.relative_to(output_dir)),
                    "unit_count": len(document_units),
                    "status": "cached",
                }
            )
            total_units += len(document_units)
            continue

        encoded_units: list[dict[str, Any]] = []
        document_frequency: Counter[str] = Counter()
        total_document_length = 0
        for unit in document_units:
            tokens = lexical_tokens(
                unit.text,
                pattern=config.bm25.token_pattern,
                lowercase=config.bm25.lowercase,
            )
            term_frequency = Counter(tokens)
            document_frequency.update(term_frequency.keys())
            total_document_length += len(tokens)
            encoded_units.append(
                {
                    "retrieval_unit_id": unit.retrieval_unit_id,
                    "unit_index": unit.unit_index,
                    "length": len(tokens),
                    "term_frequency": dict(sorted(term_frequency.items())),
                }
            )

        average_length = total_document_length / len(encoded_units) if encoded_units else 0.0
        payload = {
            "schema_version": 2,
            "condition": "bm25",
            "chunk_size_tokens": chunk_size,
            "document_id": document_id,
            "segmentation_plan": "continuous",
            "lowercase": config.bm25.lowercase,
            "token_pattern": config.bm25.token_pattern,
            "k1": config.bm25.k1,
            "b": config.bm25.b,
            "unit_count": len(encoded_units),
            "average_document_length": average_length,
            "document_frequency": dict(sorted(document_frequency.items())),
            "units": encoded_units,
        }
        _write_gzip_json(payload, output_path)
        manifest_documents.append(
            {
                "document_id": document_id,
                "analysis_set": document_units[0].analysis_set,
                "file": str(output_path.relative_to(output_dir)),
                "unit_count": len(document_units),
                "status": "written",
            }
        )
        total_units += len(document_units)

    manifest = {
        "schema_version": 2,
        "condition": "bm25",
        "chunk_size_tokens": chunk_size,
        "segmentation_plan": "continuous",
        "document_count": len(units_by_document),
        "unit_count": total_units,
        "cached_document_count": skipped_documents,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "documents": manifest_documents,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest

