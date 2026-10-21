#!/usr/bin/env python3
"""Instrument the real LMLC context windows without running embedding models.

Run from the repository root:
  python scripts/audit_lmlc_window_details.py \
    --config configs/qasper_full.yaml \
    --config configs/doc2dial_full.yaml \
    --config configs/multidoc2dial_full.yaml \
    --config configs/qasa_full.yaml

The script reuses the project's own LMLC window builder and tokenizer settings.
It loads tokenizers only (no model weights / forward passes), does not modify
experiment outputs, and writes per-target detail plus grouped summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_NAMES = ("multidoc2dial", "doc2dial", "qasper", "qasa")


def dataset_label(config_path: Path, config: Any) -> str:
    text = " ".join(
        [
            str(config_path),
            str(getattr(getattr(config, "dataset", None), "repo_id", "")),
            str(getattr(getattr(config, "paths", None), "processed_dir", "")),
        ]
    ).lower()
    for name in DATASET_NAMES:
        if name in text:
            return name
    return config_path.stem


def _flat_list(value: Any) -> list[Any]:
    """Flatten a tokenizer's optional one-item batch dimension for 1-D values."""
    value = value.tolist() if hasattr(value, "tolist") else value
    if value and isinstance(value[0], list) and (
        not value[0] or not isinstance(value[0][0], (list, tuple))
    ):
        return list(value[0])
    return list(value)


def _offset_pairs(value: Any) -> list[tuple[int, int]]:
    """Read sequence-level offsets without mistaking [start, end] for a batch."""
    value = value.tolist() if hasattr(value, "tolist") else value
    if value and isinstance(value[0], (list, tuple)) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return [(int(pair[0]), int(pair[1])) for pair in value]


def encode_with_offsets(tokenizer: Any, text: str) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = _flat_list(encoded["input_ids"])
    offsets = _offset_pairs(encoded["offset_mapping"])
    if "special_tokens_mask" in encoded:
        special = [int(x) for x in _flat_list(encoded["special_tokens_mask"])]
    else:
        special = [0] * len(offsets)
    if len(ids) != len(offsets) or len(special) != len(offsets):
        raise RuntimeError("Tokenizer returned inconsistent input-id / offset lengths")
    content_count = sum(1 for (s, e), sp in zip(offsets, special) if not sp and e > s)
    return {
        "input_tokens_including_special": len(ids),
        "content_tokens": content_count,
        "offsets": offsets,
        "special_mask": special,
    }


def overlap_token_count(offsets: list[tuple[int, int]], special: list[int], start: int, end: int) -> int:
    return sum(
        1
        for (s, e), sp in zip(offsets, special)
        if not sp and e > s and s < end and e > start
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model"]), int(row["chunk_size_tokens"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, model, chunk), group in sorted(groups.items()):
        n = len(group)
        deltas = [float(r["model_input_token_delta_lmlc_minus_sclc"]) for r in group]
        abs_deltas = [abs(x) for x in deltas]
        target_deltas = [int(r["target_model_token_delta_lmlc_minus_sclc"]) for r in group]
        lmlc_lengths = [int(r["lmlc_model_content_tokens"]) for r in group]
        sclc_lengths = [int(r["sclc_model_content_tokens"]) for r in group]
        boundaries = [bool(r["overlaps_nonparent_section"]) for r in group]
        outside = [bool(r["window_extends_outside_parent_span"]) for r in group]
        edge_shifted = [not bool(r["relative_position_preserved"]) for r in group]
        exact_length = [d == 0 for d in deltas]
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "chunk_size_tokens": chunk,
                "target_unit_count": n,
                "fraction_windows_overlapping_other_top_sections": round(sum(boundaries) / n, 6),
                "fraction_windows_extending_outside_parent_span": round(sum(outside) / n, 6),
                "fraction_contexts_equal_in_model_input_token_count": round(sum(exact_length) / n, 6),
                "median_lmlc_minus_sclc_model_input_tokens": statistics.median(deltas),
                "mean_lmlc_minus_sclc_model_input_tokens": round(statistics.mean(deltas), 4),
                "median_absolute_model_input_token_difference": statistics.median(abs_deltas),
                "p95_absolute_model_input_token_difference": round(percentile(abs_deltas, 0.95) or 0.0, 4),
                "fraction_lmlc_longer_in_model_tokens": round(sum(x > 0 for x in deltas) / n, 6),
                "fraction_lmlc_shorter_in_model_tokens": round(sum(x < 0 for x in deltas) / n, 6),
                "median_lmlc_minus_sclc_pooled_target_tokens": statistics.median(target_deltas),
                "fraction_windows_shifted_by_document_edge": round(sum(edge_shifted) / n, 6),
                "median_sclc_model_content_tokens": statistics.median(sclc_lengths),
                "median_lmlc_model_content_tokens": statistics.median(lmlc_lengths),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, action="append", required=True,
                        help="Experiment config path; repeat once per dataset.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--models", nargs="+", choices=["granite", "jina"], default=["granite", "jina"])
    parser.add_argument("--outdir", type=Path, default=Path("outputs/audit/lmlc_window_instrumentation"))
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    try:
        from transformers import AutoTokenizer
        from sclc.config import load_config
        from sclc.data.context_windows import build_length_matched_windows, load_canonical_tokenizer
        from sclc.data.io import read_documents_jsonl
        from sclc.data.retrieval_unit_io import read_retrieval_units, read_top_level_sections
        from sclc.encoding.dense import _eligible_analysis_set, _model_config
        from sclc.options import EmbeddingModel
        from sclc.paths import retrieval_unit_dir
    except Exception as exc:
        raise SystemExit(
            "Could not import the project's SCLC modules. Run this from the repository "
            f"root in the project environment. Import error: {exc}"
        ) from exc

    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    run_info: list[dict[str, Any]] = []
    warnings: list[str] = []

    for config_input in args.config:
        config_path = config_input if config_input.is_absolute() else root / config_input
        config_path = config_path.resolve()
        if not config_path.exists():
            warnings.append(f"Config not found: {config_path}")
            continue
        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.append(f"Could not load config {config_path}: {exc}")
            continue

        dataset = dataset_label(config_path, config)
        docs_path = Path(config.paths.processed_dir) / "documents.jsonl"
        if not docs_path.is_absolute():
            docs_path = root / docs_path
        if not docs_path.exists():
            warnings.append(f"{dataset}: missing prepared documents file {docs_path}")
            continue
        documents = {d.document_id: d for d in read_documents_jsonl(docs_path)}
        canonical_tokenizer = load_canonical_tokenizer(config)

        model_tokenizers: dict[str, Any] = {}
        model_max_tokens: dict[str, int | None] = {}
        for model_name in args.models:
            model_enum = EmbeddingModel.GRANITE if model_name == "granite" else EmbeddingModel.JINA
            profile = _model_config(config, model_enum)
            kwargs: dict[str, Any] = {
                "trust_remote_code": profile.model_trust_remote_code,
                "cache_dir": str(config.paths.hf_cache_dir),
            }
            if getattr(profile, "revision", None) is not None:
                kwargs["revision"] = profile.revision
            tok = AutoTokenizer.from_pretrained(profile.model_id, **kwargs)
            if not getattr(tok, "is_fast", False):
                raise RuntimeError(f"{model_name} tokenizer is not fast; target-offset audit requires a fast tokenizer")
            model_tokenizers[model_name] = tok
            model_max_tokens[model_name] = getattr(profile, "max_document_tokens", None)
            run_info.append(
                {
                    "dataset": dataset,
                    "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
                    "model": model_name,
                    "model_id": profile.model_id,
                    "model_revision": getattr(profile, "revision", None),
                    "canonical_tokenizer": config.chunking.canonical_tokenizer,
                    "canonical_tokenizer_revision": config.chunking.canonical_tokenizer_revision,
                    "model_max_document_tokens": model_max_tokens[model_name],
                }
            )

        for size in args.sizes:
            unit_path = retrieval_unit_dir(config, size) / "section_bounded_units.jsonl"
            section_path = retrieval_unit_dir(config, size) / "top_level_sections.jsonl"
            if not unit_path.exists() or not section_path.exists():
                warnings.append(
                    f"{dataset} size={size}: missing units or section file "
                    f"({unit_path.exists()=}, {section_path.exists()=}); skipped"
                )
                continue
            units_all = list(read_retrieval_units(unit_path))
            sections_all = list(read_top_level_sections(section_path))
            section_by_id = {s.parent_section_id: s for s in sections_all}
            sections_by_doc: dict[str, list[Any]] = defaultdict(list)
            for section in sections_all:
                sections_by_doc[section.document_id].append(section)

            for model_name in args.models:
                model_enum = EmbeddingModel.GRANITE if model_name == "granite" else EmbeddingModel.JINA
                units = [u for u in units_all if _eligible_analysis_set(model_enum, u.analysis_set)]
                units_by_doc: dict[str, list[Any]] = defaultdict(list)
                for unit in units:
                    units_by_doc[unit.document_id].append(unit)

                tokenizer = model_tokenizers[model_name]
                for doc_id, doc_units in units_by_doc.items():
                    document = documents.get(doc_id)
                    if document is None:
                        warnings.append(f"{dataset} {model_name} size={size}: missing document {doc_id!r}")
                        continue
                    windows = build_length_matched_windows(
                        document_text=document.text,
                        units=list(doc_units),
                        parent_sections=section_by_id,
                        tokenizer=canonical_tokenizer,
                    )
                    # The exact implementation measures section length using canonical
                    # token offsets over the complete prepared document.
                    canonical_doc = canonical_tokenizer(
                        document.text,
                        add_special_tokens=False,
                        truncation=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        verbose=False,
                    )
                    canonical_offsets = _offset_pairs(canonical_doc["offset_mapping"])
                    canonical_parent_count_cache: dict[str, int] = {}
                    parent_encoding_cache: dict[str, dict[str, Any]] = {}
                    by_window: dict[tuple[int, int], list[Any]] = defaultdict(list)
                    for unit in doc_units:
                        w = windows[unit.retrieval_unit_id]
                        by_window[(w.span.start, w.span.end)].append(unit)

                    doc_sections = sections_by_doc.get(doc_id, [])
                    for (win_start, win_end), group in by_window.items():
                        first_window = windows[group[0].retrieval_unit_id]
                        # Tokenize this unique context once, then score all target spans in it.
                        window_enc = encode_with_offsets(tokenizer, first_window.text)
                        for unit in group:
                            window = windows[unit.retrieval_unit_id]
                            parent = section_by_id[unit.parent_section_id]
                            parent_enc = parent_encoding_cache.get(parent.parent_section_id)
                            if parent_enc is None:
                                parent_enc = encode_with_offsets(tokenizer, parent.text)
                                parent_encoding_cache[parent.parent_section_id] = parent_enc
                            local_target = window.target
                            lmlc_target_tokens = overlap_token_count(
                                window_enc["offsets"], window_enc["special_mask"],
                                local_target.start, local_target.end,
                            )
                            section_target_start = unit.span.start - parent.span.start
                            section_target_end = unit.span.end - parent.span.start
                            sclc_target_tokens = overlap_token_count(
                                parent_enc["offsets"], parent_enc["special_mask"],
                                section_target_start, section_target_end,
                            )
                            other_sections = [
                                s for s in doc_sections
                                if s.parent_section_id != unit.parent_section_id
                                and window.span.start < s.span.end
                                and window.span.end > s.span.start
                            ]
                            canonical_parent_tokens = canonical_parent_count_cache.get(parent.parent_section_id)
                            if canonical_parent_tokens is None:
                                overlapping_indices = [
                                    i for i, (s0, e0) in enumerate(canonical_offsets)
                                    if s0 < parent.span.end and e0 > parent.span.start
                                ]
                                if not overlapping_indices:
                                    raise RuntimeError(
                                        f"No canonical tokens overlap section {parent.parent_section_id}"
                                    )
                                canonical_parent_tokens = max(overlapping_indices) + 1 - min(overlapping_indices)
                                canonical_parent_count_cache[parent.parent_section_id] = canonical_parent_tokens
                            # Verify that the target remains the exact original unit text.
                            target_text_ok = (
                                window.text[local_target.start:local_target.end] == unit.text
                            )
                            model_cap = model_max_tokens[model_name]
                            all_rows.append(
                                {
                                    "dataset": dataset,
                                    "config": str(config_path),
                                    "model": model_name,
                                    "chunk_size_tokens": size,
                                    "document_id": doc_id,
                                    "retrieval_unit_id": unit.retrieval_unit_id,
                                    "analysis_set": unit.analysis_set,
                                    "parent_section_id": unit.parent_section_id,
                                    "target_char_start": unit.span.start,
                                    "target_char_end": unit.span.end,
                                    "parent_char_start": parent.span.start,
                                    "parent_char_end": parent.span.end,
                                    "lmlc_window_char_start": window.span.start,
                                    "lmlc_window_char_end": window.span.end,
                                    "lmlc_canonical_token_count": window.canonical_token_count,
                                    "parent_canonical_token_count": canonical_parent_tokens,
                                    "canonical_length_match_ok": window.canonical_token_count == canonical_parent_tokens,
                                    "window_crosses_parent_start": window.span.start < parent.span.start,
                                    "window_crosses_parent_end": window.span.end > parent.span.end,
                                    "window_extends_outside_parent_span": (
                                        window.span.start < parent.span.start or window.span.end > parent.span.end
                                    ),
                                    "overlaps_nonparent_section": bool(other_sections),
                                    "nonparent_sections_overlapped": len(other_sections),
                                    "window_equals_parent_span": (
                                        window.span.start == parent.span.start and window.span.end == parent.span.end
                                    ),
                                    "relative_position_preserved": window.relative_position_preserved,
                                    "target_text_preserved_exactly": target_text_ok,
                                    "sclc_model_input_tokens_including_special": parent_enc["input_tokens_including_special"],
                                    "lmlc_model_input_tokens_including_special": window_enc["input_tokens_including_special"],
                                    "sclc_model_content_tokens": parent_enc["content_tokens"],
                                    "lmlc_model_content_tokens": window_enc["content_tokens"],
                                    "model_input_token_delta_lmlc_minus_sclc": (
                                        window_enc["input_tokens_including_special"] -
                                        parent_enc["input_tokens_including_special"]
                                    ),
                                    "sclc_target_model_tokens_pooled": sclc_target_tokens,
                                    "lmlc_target_model_tokens_pooled": lmlc_target_tokens,
                                    "target_model_token_delta_lmlc_minus_sclc": lmlc_target_tokens - sclc_target_tokens,
                                    "lmlc_exceeds_configured_model_token_limit": (
                                        None if model_cap is None else
                                        window_enc["input_tokens_including_special"] > int(model_cap)
                                    ),
                                }
                            )

    summary_rows = summarize(all_rows)
    details_path = outdir / "lmlc_window_details.csv"
    summary_path = outdir / "lmlc_window_summary.csv"
    info_path = outdir / "lmlc_window_run_info.json"
    report_path = outdir / "lmlc_window_instrumentation.txt"
    write_csv(details_path, all_rows)
    write_csv(summary_path, summary_rows)
    info_path.write_text(
        json.dumps({"configs_and_models": run_info, "warnings": warnings,
                    "detail_rows": len(all_rows), "summary_rows": len(summary_rows)},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "LMLC WINDOW INSTRUMENTATION",
        "=" * 72,
        f"Detail rows: {len(all_rows)}",
        f"Summary groups: {len(summary_rows)}",
        f"Details CSV: {details_path}",
        f"Summary CSV: {summary_path}",
        f"Run info: {info_path}",
        "",
        "GROUPED SUMMARY",
        "-" * 72,
    ]
    if summary_rows:
        cols = list(summary_rows[0].keys())
        lines.append("\t".join(cols))
        for row in summary_rows:
            lines.append("\t".join(str(row.get(k, "")) for k in cols))
    if warnings:
        lines += ["", "WARNINGS", "-" * 72, *warnings]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("LMLC window instrumentation complete.")
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")
    print(f"Run report: {report_path}")
    print(f"Rows recorded: {len(all_rows)}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see {info_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
