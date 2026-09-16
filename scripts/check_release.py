#!/usr/bin/env python3
"""Run lightweight consistency checks for the dissertation software release."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = REPO_ROOT / "reproducibility/frozen_inputs.json"
FROZEN_FILES = {
    "sample_manifest": REPO_ROOT / "data/subsets/selected_documents.csv",
    "query_types": REPO_ROOT / "data/retrieval_units/query_types.csv",
    "query_type_coding_record": REPO_ROOT
    / "data/retrieval_units/query_type_coding_record.csv",
    "challenge_review": REPO_ROOT
    / "data/subsets_cross_section_challenge/review_decisions.csv",
    "challenge_documents": REPO_ROOT
    / "data/subsets_cross_section_challenge/selected_documents.csv",
}
REMOVED_PATHS = (
    REPO_ROOT / "data/retrieval_units/query_type_coding_completed.csv",
    REPO_ROOT / "data/retrieval_units/query_type_coding_summary.txt",
    REPO_ROOT / "data/retrieval_units_cross_section_challenge/query_type_coding.csv",
    REPO_ROOT / "examples/query_types.csv",
    REPO_ROOT / "tests/test_test_qasper_parquet_loader.py",
)
LEGACY_PATTERNS = (
    re.compile(r"v0\.\d+", re.IGNORECASE),
    re.compile(r"chunk[- ]size pilot", re.IGNORECASE),
    re.compile(r"chunk[- ]size sensitivity", re.IGNORECASE),
    re.compile(r"select-chunk-size", re.IGNORECASE),
    re.compile(r"\bsensitivity analysis\b", re.IGNORECASE),
)
SEARCH_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".cff", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_frozen_inputs() -> dict[str, str]:
    for label, path in FROZEN_FILES.items():
        if not path.is_file():
            raise SystemExit(
                f"Missing frozen input ({label}): {path.relative_to(REPO_ROOT)}"
            )

    sample = csv_rows(FROZEN_FILES["sample_manifest"])
    if len(sample) != 200:
        raise SystemExit(f"Expected 200 frozen papers, found {len(sample)}")
    analysis_counts: dict[str, int] = {}
    for row in sample:
        analysis_counts[row["analysis_set"]] = analysis_counts.get(row["analysis_set"], 0) + 1
    if analysis_counts != {"cross_model_core": 150, "granite_extended": 50}:
        raise SystemExit(f"Unexpected frozen analysis-set counts: {analysis_counts}")

    query_types = csv_rows(FROZEN_FILES["query_types"])
    if len(query_types) != 554:
        raise SystemExit(f"Expected 554 coded questions, found {len(query_types)}")
    coding_record = csv_rows(FROZEN_FILES["query_type_coding_record"])
    if len(coding_record) != 554:
        raise SystemExit(
            f"Expected 554 rows in the detailed coding record, found {len(coding_record)}"
        )
    compact_labels = {row["query_id"]: row["query_type"] for row in query_types}
    detailed_labels = {row["query_id"]: row["query_type"] for row in coding_record}
    if compact_labels != detailed_labels:
        raise SystemExit("query_types.csv does not match query_type_coding_record.csv")

    type_counts: dict[str, int] = {}
    for row in query_types:
        query_type = row["query_type"]
        type_counts[query_type] = type_counts.get(query_type, 0) + 1
    expected_types = {
        "factual": 432,
        "section_specific": 66,
        "multi_hop": 21,
        "synthesis": 35,
    }
    if type_counts != expected_types:
        raise SystemExit(f"Unexpected query-type counts: {type_counts}")

    review = csv_rows(FROZEN_FILES["challenge_review"])
    if len(review) != 53:
        raise SystemExit(f"Expected 53 reviewed challenge candidates, found {len(review)}")
    accepted = [row for row in review if row.get("include", "").strip().lower() == "yes"]
    if len(accepted) != 23:
        raise SystemExit(f"Expected 23 accepted challenge questions, found {len(accepted)}")
    accepted_groups: dict[str, int] = {}
    for row in accepted:
        group = row["eligibility_group"]
        accepted_groups[group] = accepted_groups.get(group, 0) + 1
    if accepted_groups != {"cross_model_core": 18, "granite_extended": 5}:
        raise SystemExit(f"Unexpected accepted challenge breakdown: {accepted_groups}")

    challenge_documents = csv_rows(FROZEN_FILES["challenge_documents"])
    if len(challenge_documents) != 20:
        raise SystemExit(
            f"Expected 20 accepted challenge papers, found {len(challenge_documents)}"
        )

    observed_hashes = {label: sha256(path) for label, path in FROZEN_FILES.items()}
    if not FROZEN_MANIFEST.is_file():
        raise SystemExit("Missing reproducibility/frozen_inputs.json")
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    expected_hashes = {
        label: entry["sha256"] for label, entry in manifest.get("files", {}).items()
    }
    if observed_hashes != expected_hashes:
        raise SystemExit(
            "Frozen input hashes differ from reproducibility/frozen_inputs.json"
        )
    return observed_hashes


def check_removed_files() -> None:
    remaining = [path.relative_to(REPO_ROOT) for path in REMOVED_PATHS if path.exists()]
    if remaining:
        raise SystemExit(f"Redundant release artefacts still exist: {remaining}")


def check_config() -> None:
    config_path = REPO_ROOT / "configs/base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_bootstrap_metrics = [
        "ndcg_at_5",
        "recall_at_5",
        "evidence_paragraph_recall_at_5",
        "complete_evidence_at_5",
        "evidence_paragraph_recall_at_token_budget_1024",
        "complete_evidence_at_token_budget_2048",
    ]
    expected = {
        "project.seed": 42,
        "dataset.repo_id": "allenai/qasper",
        "chunking.canonical_tokenizer": (
            "ibm-granite/granite-embedding-311m-multilingual-r2"
        ),
        "chunking.supported_chunk_sizes": [128, 256, 512],
        "chunking.overlap_tokens": 0,
        "models.granite.max_document_tokens": 32768,
        "models.jina.max_document_tokens": 8192,
        "sampling.core_documents": 150,
        "sampling.granite_extended_documents": 50,
        "dense.pooling": "mean",
        "dense.normalize": True,
        "bm25.k1": 1.5,
        "bm25.b": 0.75,
        "ranking.store_complete_ranking": True,
        "evaluation.bootstrap_iterations": 10000,
        "evaluation.confirmatory_split": "test",
        "evaluation.primary_metric": "ndcg_at_5",
        "evaluation.bootstrap_metrics": expected_bootstrap_metrics,
    }
    observed = {
        "project.seed": config["project"]["seed"],
        "dataset.repo_id": config["dataset"]["repo_id"],
        "chunking.canonical_tokenizer": config["chunking"]["canonical_tokenizer"],
        "chunking.supported_chunk_sizes": config["chunking"]["supported_chunk_sizes"],
        "chunking.overlap_tokens": config["chunking"]["overlap_tokens"],
        "models.granite.max_document_tokens": config["models"]["granite"][
            "max_document_tokens"
        ],
        "models.jina.max_document_tokens": config["models"]["jina"][
            "max_document_tokens"
        ],
        "sampling.core_documents": config["sampling"]["core_documents"],
        "sampling.granite_extended_documents": config["sampling"][
            "granite_extended_documents"
        ],
        "dense.pooling": config["dense"]["pooling"],
        "dense.normalize": config["dense"]["normalize"],
        "bm25.k1": config["bm25"]["k1"],
        "bm25.b": config["bm25"]["b"],
        "ranking.store_complete_ranking": config["ranking"]["store_complete_ranking"],
        "evaluation.bootstrap_iterations": config["evaluation"]["bootstrap_iterations"],
        "evaluation.confirmatory_split": config["evaluation"]["confirmatory_split"],
        "evaluation.primary_metric": config["evaluation"]["primary_metric"],
        "evaluation.bootstrap_metrics": config["evaluation"]["bootstrap_metrics"],
    }
    if observed != expected:
        raise SystemExit(
            f"Primary configuration drift detected:\n{json.dumps(observed, indent=2)}"
        )


def check_versions() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (REPO_ROOT / "src/sclc/__init__.py").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    project_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    init_match = re.search(r'__version__\s*=\s*"([^"]+)"', package_init)
    citation_match = re.search(r'^version:\s*([^\s]+)', citation, re.MULTILINE)
    versions = {
        project_match.group(1) if project_match else None,
        init_match.group(1) if init_match else None,
        citation_match.group(1) if citation_match else None,
    }
    if len(versions) != 1 or None in versions:
        raise SystemExit(f"Release version mismatch: {versions}")
    release_version = next(iter(versions))
    frozen_manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if frozen_manifest.get("release_version") != release_version:
        raise SystemExit(
            "reproducibility/frozen_inputs.json has a different release version"
        )


def check_legacy_text() -> None:
    hits: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SEARCH_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LEGACY_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
    if hits:
        raise SystemExit("Legacy public-facing text remains:\n" + "\n".join(hits))


def main() -> None:
    hashes = check_frozen_inputs()
    check_removed_files()
    check_config()
    check_versions()
    check_legacy_text()
    print("Release consistency checks passed.")
    for label, digest in hashes.items():
        print(f"  {label}: {digest}")


if __name__ == "__main__":
    main()
