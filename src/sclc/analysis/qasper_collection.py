from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from sclc.analysis.evidence_structure import (
    EVIDENCE_STRUCTURES,
    classify_evidence_structure,
)
from sclc.config import AppConfig
from sclc.data.chunking import build_top_level_sections
from sclc.data.io import read_documents_jsonl
from sclc.data.relevance import prepare_queries
from sclc.data.schema import DocumentRecord, PreparedQueryRecord

AUDIT_DIRNAME = "qasper_collection_audit"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"document_id": "string"})


def _paragraph_parent_metadata(
    document: DocumentRecord,
) -> tuple[dict[str, str], dict[str, str]]:
    parents = build_top_level_sections(document, analysis_set="qasper_collection")
    section_to_parent: dict[str, str] = {}
    parent_headings: dict[str, str] = {}
    for parent in parents:
        parent_headings[parent.parent_section_id] = parent.heading
        for source_section_id in parent.source_section_ids:
            if source_section_id in section_to_parent:
                raise RuntimeError(
                    "A source section maps to multiple top-level sections: "
                    f"{document.document_id}/{source_section_id}"
                )
            section_to_parent[source_section_id] = parent.parent_section_id

    paragraph_to_parent: dict[str, str] = {}
    for paragraph in document.paragraphs:
        parent_id = section_to_parent.get(paragraph.section_id)
        if parent_id is None:
            raise RuntimeError(
                "Paragraph could not be mapped to a top-level section: "
                f"{document.document_id}/{paragraph.paragraph_id}"
            )
        paragraph_to_parent[paragraph.paragraph_id] = parent_id
    return paragraph_to_parent, parent_headings


def _evidence_details(
    query: PreparedQueryRecord,
    document: DocumentRecord,
    paragraph_to_parent: Mapping[str, str],
    parent_headings: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paragraph_lookup = {
        paragraph.paragraph_id: paragraph for paragraph in document.paragraphs
    }
    rows: list[dict[str, Any]] = []
    for evidence_set in query.evidence_sets:
        paragraph_ids = sorted(set(evidence_set.paragraph_ids))
        parent_ids = sorted({paragraph_to_parent[item] for item in paragraph_ids})
        rows.append(
            {
                "evidence_set_id": evidence_set.evidence_set_id,
                "paragraph_ids": paragraph_ids,
                "top_level_section_ids": parent_ids,
                "top_level_section_headings": [
                    parent_headings[parent_id] for parent_id in parent_ids
                ],
                "paragraph_texts": [
                    paragraph_lookup[paragraph_id].text
                    for paragraph_id in paragraph_ids
                ],
            }
        )

    classification = classify_evidence_structure(query, paragraph_to_parent)
    primary_id = classification["primary_evidence_set_id"]
    primary = next(
        row for row in rows if row["evidence_set_id"] == primary_id
    )
    return rows, primary


def _candidate_status(
    *,
    split: str,
    selected_document: bool,
    eligibility_group: str | None,
) -> str:
    if selected_document:
        return "already_in_current_experiment"
    if split == "validation":
        return "validation_excluded_from_new_challenge"
    if eligibility_group is None:
        return "profile_required_before_selection"
    if eligibility_group == "excluded_too_long":
        return "excluded_too_long_for_strict_global"
    if split == "test":
        return "new_test_candidate"
    if split == "train":
        return "new_train_candidate"
    return "new_candidate_other_split"


def build_qasper_collection_audit_frame(
    config: AppConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    documents_path = config.paths.processed_dir / "documents.jsonl"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"{documents_path} does not exist. Run `sclc prepare` first."
        )

    profile_path = config.paths.profile_dir / "document_lengths.csv"
    selected_path = config.paths.subset_dir / "selected_documents.csv"
    profile = _optional_frame(profile_path)
    selected = _optional_frame(selected_path)

    profile_lookup: dict[str, dict[str, Any]] = {}
    if not profile.empty:
        profile_lookup = {
            str(row["document_id"]): row.to_dict()
            for _, row in profile.iterrows()
        }

    selected_lookup: dict[str, dict[str, Any]] = {}
    if not selected.empty:
        selected_lookup = {
            str(row["document_id"]): row.to_dict()
            for _, row in selected.iterrows()
        }

    rows: list[dict[str, Any]] = []
    total_documents = 0
    total_questions = 0
    total_usable_questions = 0

    for document in read_documents_jsonl(documents_path):
        total_documents += 1
        total_questions += len(document.queries)
        paragraph_to_parent, parent_headings = _paragraph_parent_metadata(document)
        prepared_queries = prepare_queries(document, analysis_set="qasper_collection")
        total_usable_questions += len(prepared_queries)

        profile_row = profile_lookup.get(document.document_id, {})
        selected_row = selected_lookup.get(document.document_id, {})
        selected_document = bool(selected_row)
        eligibility_value = profile_row.get("eligibility_group")
        eligibility_group = (
            None if pd.isna(eligibility_value) else str(eligibility_value)
        )

        top_level_section_count = len(set(paragraph_to_parent.values()))
        for query in prepared_queries:
            details, primary = _evidence_details(
                query,
                document,
                paragraph_to_parent,
                parent_headings,
            )
            classification = classify_evidence_structure(
                query, paragraph_to_parent
            )
            candidate_status = _candidate_status(
                split=document.split,
                selected_document=selected_document,
                eligibility_group=eligibility_group,
            )
            strict_cross_section = bool(classification["cross_section_required"])
            minimal_cross_section = bool(
                classification["cross_section_required_among_minimal_sets"]
            )

            rows.append(
                {
                    "query_id": query.query_id,
                    "document_id": document.document_id,
                    "split": document.split,
                    "title": document.title,
                    "question": query.question,
                    "evidence_structure": classification["evidence_structure"],
                    "strict_cross_section_required": strict_cross_section,
                    "minimal_cross_section_required": minimal_cross_section,
                    "cross_section_possible": bool(
                        classification["cross_section_possible"]
                    ),
                    "primary_evidence_set_id": classification[
                        "primary_evidence_set_id"
                    ],
                    "minimum_evidence_paragraph_count": classification[
                        "minimum_evidence_paragraph_count"
                    ],
                    "minimal_evidence_set_count": classification[
                        "minimal_evidence_set_count"
                    ],
                    "minimum_top_level_section_count_among_minimal_sets": (
                        classification[
                            "minimum_top_level_section_count_among_minimal_sets"
                        ]
                    ),
                    "maximum_top_level_section_count_among_minimal_sets": (
                        classification[
                            "maximum_top_level_section_count_among_minimal_sets"
                        ]
                    ),
                    "minimal_structure_variants": classification[
                        "minimal_structure_variants"
                    ],
                    "minimal_structure_ambiguous": classification[
                        "minimal_structure_ambiguous"
                    ],
                    "acceptable_set_structures": classification[
                        "acceptable_set_structures"
                    ],
                    "evidence_set_count": classification["evidence_set_count"],
                    "evidence_union_structure": classification[
                        "evidence_union_structure"
                    ],
                    "evidence_union_paragraph_count": classification[
                        "evidence_union_paragraph_count"
                    ],
                    "evidence_union_top_level_section_count": classification[
                        "evidence_union_top_level_section_count"
                    ],
                    "primary_evidence_paragraph_ids": "|".join(
                        primary["paragraph_ids"]
                    ),
                    "primary_evidence_section_ids": "|".join(
                        primary["top_level_section_ids"]
                    ),
                    "primary_evidence_section_headings": "|".join(
                        primary["top_level_section_headings"]
                    ),
                    "primary_evidence_texts_json": json.dumps(
                        primary["paragraph_texts"], ensure_ascii=False
                    ),
                    "evidence_sets_detailed_json": json.dumps(
                        details, ensure_ascii=False, sort_keys=True
                    ),
                    "document_character_count": len(document.text),
                    "document_source_section_count": len(document.sections),
                    "document_top_level_section_count": top_level_section_count,
                    "document_paragraph_count": len(document.paragraphs),
                    "document_usable_question_count": len(prepared_queries),
                    "granite_tokens": profile_row.get("granite_tokens"),
                    "jina_tokens": profile_row.get("jina_tokens"),
                    "eligibility_group": eligibility_group,
                    "selected_document": selected_document,
                    "selected_analysis_set": selected_row.get("analysis_set"),
                    "selected_length_stratum": selected_row.get("length_stratum"),
                    "candidate_status": candidate_status,
                    "eligible_new_cross_model_strict_candidate": bool(
                        strict_cross_section
                        and candidate_status
                        in {"new_test_candidate", "new_train_candidate"}
                        and eligibility_group == "cross_model_core"
                    ),
                    "eligible_new_granite_strict_candidate": bool(
                        strict_cross_section
                        and candidate_status
                        in {"new_test_candidate", "new_train_candidate"}
                        and eligibility_group
                        in {"cross_model_core", "granite_extended"}
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No usable textual-evidence questions were found")
    frame = frame.sort_values(
        [
            "split",
            "strict_cross_section_required",
            "minimal_cross_section_required",
            "document_id",
            "query_id",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    metadata = {
        "total_documents": total_documents,
        "total_questions": total_questions,
        "usable_textual_evidence_questions": total_usable_questions,
        "profile_available": profile_path.exists(),
        "selected_manifest_available": selected_path.exists(),
    }
    return frame, metadata


def audit_qasper_collection(
    config: AppConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    documents_path = config.paths.processed_dir / "documents.jsonl"
    profile_path = config.paths.profile_dir / "document_lengths.csv"
    selected_path = config.paths.subset_dir / "selected_documents.csv"
    output_dir = config.paths.analysis_dir / AUDIT_DIRNAME
    manifest_path = output_dir / "manifest.json"

    source_hashes = {"documents": _file_sha256(documents_path)}
    if profile_path.exists():
        source_hashes["profile"] = _file_sha256(profile_path)
    if selected_path.exists():
        source_hashes["selected_documents"] = _file_sha256(selected_path)

    configuration = {
        "source_hashes": source_hashes,
        "hierarchy_separator": config.document.hierarchy_separator,
        "reference_sections_removed_during_prepare": (
            config.document.remove_reference_sections
        ),
        "structures": list(EVIDENCE_STRUCTURES),
        "classification_policy": {
            "primary_set": "minimum_paragraph_count_complete_acceptable_set",
            "tie_break": "least_distributed_structure",
            "strict_cross_section": (
                "every acceptable evidence set spans multiple top-level sections"
            ),
            "minimal_cross_section": (
                "every minimum-paragraph acceptable set spans multiple "
                "top-level sections"
            ),
        },
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached QASPER audit at {output_dir} does not match current "
                "inputs. Re-run with --overwrite."
            )
        return existing

    frame, metadata = build_qasper_collection_audit_frame(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = output_dir / "qasper_evidence_structure_queries.csv"
    counts_path = output_dir / "qasper_evidence_structure_counts.csv"
    candidates_path = output_dir / "qasper_cross_section_candidates.csv"
    documents_output_path = output_dir / "qasper_cross_section_documents.csv"
    frame.to_csv(query_path, index=False)

    counts = (
        frame.groupby(
            [
                "split",
                "evidence_structure",
                "strict_cross_section_required",
                "minimal_cross_section_required",
                "candidate_status",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            query_count=("query_id", "count"),
            document_count=("document_id", "nunique"),
            ambiguous_minimal_structure_count=(
                "minimal_structure_ambiguous",
                "sum",
            ),
        )
        .sort_values(
            [
                "split",
                "evidence_structure",
                "strict_cross_section_required",
                "candidate_status",
            ]
        )
    )
    counts.to_csv(counts_path, index=False)

    candidates = frame[
        frame["minimal_cross_section_required"]
        | frame["strict_cross_section_required"]
    ].copy()
    candidates = candidates.sort_values(
        [
            "strict_cross_section_required",
            "candidate_status",
            "split",
            "document_id",
            "query_id",
        ],
        ascending=[False, True, True, True, True],
    )
    candidates.to_csv(candidates_path, index=False)

    document_candidates = (
        candidates.groupby(
            [
                "document_id",
                "split",
                "title",
                "eligibility_group",
                "selected_document",
                "selected_analysis_set",
                "candidate_status",
                "granite_tokens",
                "jina_tokens",
                "document_character_count",
                "document_top_level_section_count",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            strict_cross_section_query_count=(
                "strict_cross_section_required",
                "sum",
            ),
            minimal_cross_section_query_count=(
                "minimal_cross_section_required",
                "sum",
            ),
            candidate_query_count=("query_id", "count"),
        )
        .sort_values(
            [
                "strict_cross_section_query_count",
                "minimal_cross_section_query_count",
                "document_id",
            ],
            ascending=[False, False, True],
        )
    )
    document_candidates.to_csv(documents_output_path, index=False)

    strict = frame[frame["strict_cross_section_required"]]
    minimal = frame[frame["minimal_cross_section_required"]]
    new_cross_model_strict = frame[
        frame["eligible_new_cross_model_strict_candidate"]
    ]
    new_granite_strict = frame[
        frame["eligible_new_granite_strict_candidate"]
    ]
    manifest = {
        "schema_version": 1,
        "kind": "qasper_collection_evidence_audit",
        **metadata,
        "classified_query_count": len(frame),
        "strict_cross_section_query_count": len(strict),
        "strict_cross_section_document_count": int(strict["document_id"].nunique()),
        "minimal_cross_section_query_count": len(minimal),
        "minimal_cross_section_document_count": int(minimal["document_id"].nunique()),
        "eligible_new_cross_model_strict_candidate_count": len(
            new_cross_model_strict
        ),
        "eligible_new_cross_model_strict_candidate_document_count": int(
            new_cross_model_strict["document_id"].nunique()
        ),
        "eligible_new_granite_strict_candidate_count": len(
            new_granite_strict
        ),
        "eligible_new_granite_strict_candidate_document_count": int(
            new_granite_strict["document_id"].nunique()
        ),
        "counts_by_split": {
            split: {
                "usable_queries": int(len(group)),
                "strict_cross_section_queries": int(
                    group["strict_cross_section_required"].sum()
                ),
                "minimal_cross_section_queries": int(
                    group["minimal_cross_section_required"].sum()
                ),
                "new_cross_model_strict_candidates": int(
                    group["eligible_new_cross_model_strict_candidate"].sum()
                ),
                "new_granite_strict_candidates": int(
                    group["eligible_new_granite_strict_candidate"].sum()
                ),
            }
            for split, group in frame.groupby("split")
        },
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "all_queries": query_path.name,
            "counts": counts_path.name,
            "cross_section_candidates": candidates_path.name,
            "cross_section_documents": documents_output_path.name,
        },
        "interpretation_notes": [
            "This command audits the complete prepared QASPER collection and "
            "does not change the frozen primary sample.",
            "Strict cross-section candidates require every acceptable evidence "
            "route to span multiple top-level sections.",
            "The dissertation challenge uses only the original QASPER test split; "
            "train and validation candidates remain audit-only records.",
            "Train and test candidates are reported separately; no challenge set "
            "is sampled automatically.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "AUDIT_DIRNAME",
    "audit_qasper_collection",
    "build_qasper_collection_audit_frame",
]
