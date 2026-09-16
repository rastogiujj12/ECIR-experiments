from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from sclc.analysis.qasper_collection import AUDIT_DIRNAME
from sclc.config import AppConfig

CHALLENGE_DIRNAME = "qasper_cross_section_challenge"
CHALLENGE_VERSION = "1.0"
REVIEW_ORDER_SEED = "sclc-qasper-strict-cross-section-review-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def _review_order(query_id: str) -> str:
    return hashlib.sha256(
        f"{REVIEW_ORDER_SEED}:{query_id}".encode()
    ).hexdigest()


def _write_review_instructions(path: Path) -> None:
    text = """# Blinded review instructions

Review all rows without opening `review_key.csv`, retrieval rankings, or
condition-level results. Decisions must be based only on the question and
annotated evidence.

## Required fields

- `include`: `yes` or `no`.
- `joint_evidence_required`: `yes`, `no`, or `unclear`. Select `yes` only
  when evidence from multiple top-level sections is genuinely needed for a
  complete answer.
- `duplicate_support`: `yes`, `no`, or `unclear`. Select `yes` when the
  sections merely repeat equivalent support.
- `section_mapping_valid`: `yes`, `no`, or `unclear`.
- `answerable_from_evidence`: `yes`, `no`, or `unclear`.
- `rejection_reason`: leave blank when included. Otherwise use one of:
  `duplicate_evidence`, `one_section_is_actually_sufficient`,
  `incorrect_section_mapping`, `evidence_not_jointly_required`,
  `ambiguous_or_unsupported_question`, or `other`.
- `reviewer_notes`: a short reason supporting the decision.

## Inclusion rule

Include a question only when the section mapping is valid, the answer is
supported, and evidence from at least two top-level sections is genuinely
required rather than duplicated or optional. Use `unclear` rather than
guessing.

The review sheet is deliberately separated from `review_key.csv`, which
contains query identifiers and challenge-set membership.
"""
    path.write_text(text, encoding="utf-8")


def freeze_cross_section_challenge(
    config: AppConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    audit_dir = config.paths.analysis_dir / AUDIT_DIRNAME
    candidates_path = audit_dir / "qasper_cross_section_candidates.csv"
    audit_manifest_path = audit_dir / "manifest.json"
    if not candidates_path.exists() or not audit_manifest_path.exists():
        raise FileNotFoundError(
            "Complete QASPER audit outputs are missing. Run `sclc qasper-audit` "
            "first."
        )

    output_dir = config.paths.analysis_dir / CHALLENGE_DIRNAME
    manifest_path = output_dir / "manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))

    selection_policy = {
        "challenge_version": CHALLENGE_VERSION,
        "source_candidate_sha256": _file_sha256(candidates_path),
        "source_audit_fingerprint": audit_manifest.get(
            "configuration_fingerprint"
        ),
        "split": "test",
        "strict_cross_section_required": True,
        "allowed_eligibility_groups": [
            "cross_model_core",
            "granite_extended",
        ],
        "cross_model_complete": (
            "all strict test questions in cross_model_core"
        ),
        "cross_model_unseen": (
            "cross_model_complete questions with candidate_status=new_test_candidate"
        ),
        "granite_extension": (
            "all strict test questions in granite_extended"
        ),
        "review_order_seed": REVIEW_ORDER_SEED,
        "review_blinding": (
            "review sheet excludes query/document IDs, current-sample status, "
            "eligibility group, and challenge membership"
        ),
    }
    configuration_fingerprint = _fingerprint(selection_policy)

    if manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != configuration_fingerprint:
            raise RuntimeError(
                f"Cached challenge at {output_dir} does not match the current "
                "audit. Re-run with --overwrite."
            )
        return existing

    frame = pd.read_csv(
        candidates_path,
        dtype={"document_id": "string", "query_id": "string"},
    )
    strict = _bool_series(frame, "strict_cross_section_required")
    challenge = frame[
        frame["split"].eq("test")
        & strict
        & frame["eligibility_group"].isin(
            ["cross_model_core", "granite_extended"]
        )
    ].copy()
    if challenge.empty:
        raise RuntimeError("No eligible strict test questions were found")
    if challenge["query_id"].duplicated().any():
        duplicates = challenge.loc[
            challenge["query_id"].duplicated(), "query_id"
        ].tolist()
        raise RuntimeError(f"Duplicate challenge query IDs: {duplicates[:5]}")

    challenge["challenge_version"] = CHALLENGE_VERSION
    challenge["review_sort_key"] = challenge["query_id"].map(_review_order)
    challenge = challenge.sort_values("review_sort_key").reset_index(drop=True)
    challenge["review_id"] = [
        f"CSQ-{index:03d}" for index in range(1, len(challenge) + 1)
    ]
    challenge["in_cross_model_complete"] = challenge[
        "eligibility_group"
    ].eq("cross_model_core")
    challenge["in_cross_model_unseen"] = (
        challenge["in_cross_model_complete"]
        & challenge["candidate_status"].eq("new_test_candidate")
    )
    challenge["in_granite_extension"] = challenge[
        "eligibility_group"
    ].eq("granite_extended")

    expected_counts = {
        "all_strict_test": 53,
        "cross_model_complete": 42,
        "cross_model_unseen": 37,
        "granite_extension": 11,
    }
    observed_counts = {
        "all_strict_test": int(len(challenge)),
        "cross_model_complete": int(challenge["in_cross_model_complete"].sum()),
        "cross_model_unseen": int(challenge["in_cross_model_unseen"].sum()),
        "granite_extension": int(challenge["in_granite_extension"].sum()),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Challenge counts differ from the frozen v1 design: "
            f"expected {expected_counts}, observed {observed_counts}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_challenge_manifest.csv"
    core_path = output_dir / "cross_model_complete.csv"
    unseen_path = output_dir / "cross_model_unseen.csv"
    extension_path = output_dir / "granite_extension.csv"
    review_path = output_dir / "blinded_review_sheet.csv"
    key_path = output_dir / "review_key.csv"
    schema_path = output_dir / "review_schema.json"
    instructions_path = output_dir / "REVIEW_INSTRUCTIONS.md"

    frozen_columns = [
        "challenge_version",
        "review_id",
        "query_id",
        "document_id",
        "split",
        "eligibility_group",
        "candidate_status",
        "selected_document",
        "selected_analysis_set",
        "in_cross_model_complete",
        "in_cross_model_unseen",
        "in_granite_extension",
        "title",
        "question",
        "minimum_evidence_paragraph_count",
        "minimum_top_level_section_count_among_minimal_sets",
        "evidence_set_count",
        "primary_evidence_set_id",
        "primary_evidence_paragraph_ids",
        "primary_evidence_section_ids",
        "primary_evidence_section_headings",
        "primary_evidence_texts_json",
        "evidence_sets_detailed_json",
        "granite_tokens",
        "jina_tokens",
    ]
    frozen = challenge[frozen_columns].copy()
    frozen.to_csv(frozen_path, index=False)
    frozen[challenge["in_cross_model_complete"].to_numpy()].to_csv(
        core_path, index=False
    )
    frozen[challenge["in_cross_model_unseen"].to_numpy()].to_csv(
        unseen_path, index=False
    )
    frozen[challenge["in_granite_extension"].to_numpy()].to_csv(
        extension_path, index=False
    )

    review = pd.DataFrame(
        {
            "review_id": challenge["review_id"],
            "include": "",
            "joint_evidence_required": "",
            "duplicate_support": "",
            "section_mapping_valid": "",
            "answerable_from_evidence": "",
            "rejection_reason": "",
            "reviewer_notes": "",
            "title": challenge["title"],
            "question": challenge["question"],
            "minimum_evidence_paragraph_count": challenge[
                "minimum_evidence_paragraph_count"
            ],
            "minimum_top_level_section_count": challenge[
                "minimum_top_level_section_count_among_minimal_sets"
            ],
            "evidence_set_count": challenge["evidence_set_count"],
            "primary_evidence_section_headings": challenge[
                "primary_evidence_section_headings"
            ],
            "primary_evidence_texts_json": challenge[
                "primary_evidence_texts_json"
            ],
            "all_acceptable_evidence_sets_json": challenge[
                "evidence_sets_detailed_json"
            ],
        }
    )
    review.to_csv(review_path, index=False)

    key = challenge[
        [
            "review_id",
            "query_id",
            "document_id",
            "eligibility_group",
            "candidate_status",
            "in_cross_model_complete",
            "in_cross_model_unseen",
            "in_granite_extension",
        ]
    ].copy()
    key.to_csv(key_path, index=False)

    review_schema = {
        "include": ["yes", "no"],
        "ternary_fields": ["yes", "no", "unclear"],
        "rejection_reasons": [
            "",
            "duplicate_evidence",
            "one_section_is_actually_sufficient",
            "incorrect_section_mapping",
            "evidence_not_jointly_required",
            "ambiguous_or_unsupported_question",
            "other",
        ],
        "decision_rule": (
            "Include only when multiple top-level sections are genuinely required, "
            "the section mapping is valid, and the evidence supports the question."
        ),
    }
    schema_path.write_text(
        json.dumps(review_schema, indent=2), encoding="utf-8"
    )
    _write_review_instructions(instructions_path)

    output_hashes = {
        path.name: _file_sha256(path)
        for path in [
            frozen_path,
            core_path,
            unseen_path,
            extension_path,
            review_path,
            key_path,
            schema_path,
            instructions_path,
        ]
    }
    manifest = {
        "schema_version": 1,
        "kind": "qasper_strict_cross_section_challenge_freeze",
        "challenge_version": CHALLENGE_VERSION,
        "configuration": selection_policy,
        "configuration_fingerprint": configuration_fingerprint,
        "counts": observed_counts,
        "document_counts": {
            "all_strict_test": int(challenge["document_id"].nunique()),
            "cross_model_complete": int(
                challenge.loc[
                    challenge["in_cross_model_complete"], "document_id"
                ].nunique()
            ),
            "cross_model_unseen": int(
                challenge.loc[
                    challenge["in_cross_model_unseen"], "document_id"
                ].nunique()
            ),
            "granite_extension": int(
                challenge.loc[
                    challenge["in_granite_extension"], "document_id"
                ].nunique()
            ),
        },
        "files": {
            "frozen_challenge_manifest": frozen_path.name,
            "cross_model_complete": core_path.name,
            "cross_model_unseen": unseen_path.name,
            "granite_extension": extension_path.name,
            "blinded_review_sheet": review_path.name,
            "review_key": key_path.name,
            "review_schema": schema_path.name,
            "review_instructions": instructions_path.name,
        },
        "output_sha256": output_hashes,
        "interpretation_notes": [
            "The primary 180-question experiment remains unchanged.",
            "No retrieval results were used to select or order challenge questions.",
            "The blinded review sheet omits identifiers and prior sample membership.",
            "Review decisions must be completed before challenge retrieval is run.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _normalise_review_value(value: object) -> str:
    return str(value).strip().lower()


def finalize_cross_section_challenge(
    config: AppConfig,
    *,
    decisions_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate the completed blinded review and freeze the accepted set.

    The decision file may contain identifier columns, but identifiers and
    challenge membership are taken from the frozen review key so spreadsheet
    formatting cannot silently alter document IDs.
    """
    output_dir = config.paths.analysis_dir / CHALLENGE_DIRNAME
    freeze_manifest_path = output_dir / "manifest.json"
    key_path = output_dir / "review_key.csv"
    frozen_path = output_dir / "frozen_challenge_manifest.csv"
    if not freeze_manifest_path.exists() or not key_path.exists() or not frozen_path.exists():
        raise FileNotFoundError(
            "Frozen challenge outputs are missing. Run "
            "`sclc qasper-challenge --overwrite` first."
        )
    if not decisions_path.exists():
        raise FileNotFoundError(decisions_path)

    final_manifest_path = output_dir / "finalized_manifest.json"
    decisions_hash = _file_sha256(decisions_path)
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    configuration = {
        "challenge_version": CHALLENGE_VERSION,
        "freeze_fingerprint": freeze_manifest.get("configuration_fingerprint"),
        "decisions_sha256": decisions_hash,
        "decision_rule": (
            "include=yes only when joint_evidence_required=yes, "
            "duplicate_support=no, section_mapping_valid=yes, and "
            "answerable_from_evidence=yes"
        ),
    }
    configuration_fingerprint = _fingerprint(configuration)
    if final_manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(final_manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != configuration_fingerprint:
            raise RuntimeError(
                f"Cached finalized challenge at {output_dir} does not match the "
                "current review. Re-run with overwrite=True."
            )
        return existing

    key = pd.read_csv(
        key_path,
        dtype={"review_id": "string", "query_id": "string", "document_id": "string"},
    )
    frozen = pd.read_csv(
        frozen_path,
        dtype={"review_id": "string", "query_id": "string", "document_id": "string"},
    )
    decisions = pd.read_csv(
        decisions_path,
        dtype={"review_id": "string", "query_id": "string", "document_id": "string"},
        keep_default_na=False,
    )

    required = {
        "review_id",
        "include",
        "joint_evidence_required",
        "duplicate_support",
        "section_mapping_valid",
        "answerable_from_evidence",
        "rejection_reason",
        "reviewer_notes",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"Completed review is missing columns: {sorted(missing)}"
        )
    if decisions["review_id"].duplicated().any():
        duplicates = decisions.loc[
            decisions["review_id"].duplicated(), "review_id"
        ].tolist()
        raise ValueError(f"Duplicate review IDs: {duplicates[:5]}")

    expected_ids = set(key["review_id"].astype(str))
    observed_ids = set(decisions["review_id"].astype(str))
    if observed_ids != expected_ids:
        missing_ids = sorted(expected_ids.difference(observed_ids))
        extra_ids = sorted(observed_ids.difference(expected_ids))
        raise ValueError(
            "Completed review IDs do not match the frozen review: "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )

    decision_fields = [
        "review_id",
        "include",
        "joint_evidence_required",
        "duplicate_support",
        "section_mapping_valid",
        "answerable_from_evidence",
        "rejection_reason",
        "reviewer_notes",
    ]
    review = decisions[decision_fields].copy()
    for column in [
        "include",
        "joint_evidence_required",
        "duplicate_support",
        "section_mapping_valid",
        "answerable_from_evidence",
        "rejection_reason",
    ]:
        review[column] = review[column].map(_normalise_review_value)

    if not review["include"].isin({"yes", "no"}).all():
        raise ValueError("Every include decision must be yes or no")
    ternary_columns = [
        "joint_evidence_required",
        "duplicate_support",
        "section_mapping_valid",
        "answerable_from_evidence",
    ]
    for column in ternary_columns:
        if not review[column].isin({"yes", "no"}).all():
            raise ValueError(
                f"Every finalized {column} decision must be yes or no; "
                "unclear or blank decisions must be resolved first"
            )

    included = review["include"].eq("yes")
    valid_inclusion = (
        review["joint_evidence_required"].eq("yes")
        & review["duplicate_support"].eq("no")
        & review["section_mapping_valid"].eq("yes")
        & review["answerable_from_evidence"].eq("yes")
    )
    if not (included == valid_inclusion).all():
        invalid = review.loc[included != valid_inclusion, "review_id"].tolist()
        raise ValueError(
            "Include decisions do not follow the frozen decision rule; first: "
            f"{invalid[:5]}"
        )
    rejected = ~included
    if review.loc[rejected, "rejection_reason"].eq("").any():
        invalid = review.loc[
            rejected & review["rejection_reason"].eq(""), "review_id"
        ].tolist()
        raise ValueError(
            f"Rejected questions require a rejection reason; first: {invalid[:5]}"
        )

    merged = key.merge(review, on="review_id", how="inner", validate="one_to_one")
    merged = merged.merge(
        frozen.drop(
            columns=[
                column
                for column in [
                    "query_id",
                    "document_id",
                    "eligibility_group",
                    "candidate_status",
                    "in_cross_model_complete",
                    "in_cross_model_unseen",
                    "in_granite_extension",
                ]
                if column in frozen.columns
            ]
        ),
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    accepted = merged[merged["include"].eq("yes")].copy()
    rejected_frame = merged[merged["include"].eq("no")].copy()
    accepted_core = accepted[accepted["in_cross_model_complete"].astype(bool)].copy()
    accepted_extension = accepted[accepted["in_granite_extension"].astype(bool)].copy()

    profile_path = config.paths.profile_dir / "document_lengths.csv"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    profile = pd.read_csv(profile_path, dtype={"document_id": "string"})
    accepted_documents = accepted[
        ["document_id", "eligibility_group"]
    ].drop_duplicates()
    if accepted_documents["document_id"].duplicated().any():
        raise RuntimeError("An accepted document belongs to multiple eligibility groups")
    accepted_documents = profile.merge(
        accepted_documents,
        on="document_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_review"),
    )
    expected_document_ids = set(accepted["document_id"].astype(str))
    found_document_ids = set(accepted_documents["document_id"].astype(str))
    if found_document_ids != expected_document_ids:
        missing_documents = sorted(expected_document_ids.difference(found_document_ids))
        raise RuntimeError(
            f"Profile is missing accepted documents: {missing_documents[:5]}"
        )
    accepted_documents["eligibility_group"] = accepted_documents[
        "eligibility_group_review"
    ]
    accepted_documents["analysis_set"] = accepted_documents["eligibility_group"]
    accepted_documents = accepted_documents.drop(
        columns=["eligibility_group_review"], errors="ignore"
    )
    if "length_stratum" not in accepted_documents.columns:
        accepted_documents["length_stratum"] = "challenge"

    final_review_path = output_dir / "review_decisions_final.csv"
    accepted_path = output_dir / "accepted_queries.csv"
    accepted_core_path = output_dir / "accepted_cross_model_queries.csv"
    accepted_extension_path = output_dir / "accepted_granite_extension_queries.csv"
    rejected_path = output_dir / "rejected_queries.csv"
    accepted_documents_path = output_dir / "accepted_documents.csv"

    merged.sort_values("review_id").to_csv(final_review_path, index=False)
    accepted.sort_values("review_id").to_csv(accepted_path, index=False)
    accepted_core.sort_values("review_id").to_csv(accepted_core_path, index=False)
    accepted_extension.sort_values("review_id").to_csv(
        accepted_extension_path, index=False
    )
    rejected_frame.sort_values("review_id").to_csv(rejected_path, index=False)
    accepted_documents.sort_values(
        ["analysis_set", "document_id"]
    ).to_csv(accepted_documents_path, index=False)

    counts = {
        "reviewed": int(len(merged)),
        "accepted": int(len(accepted)),
        "rejected": int(len(rejected_frame)),
        "accepted_cross_model": int(len(accepted_core)),
        "accepted_cross_model_unseen": int(
            accepted["in_cross_model_unseen"].astype(bool).sum()
        ),
        "accepted_granite_extension": int(len(accepted_extension)),
        "accepted_documents": int(accepted["document_id"].nunique()),
    }
    manifest = {
        "schema_version": 1,
        "kind": "qasper_strict_cross_section_challenge_finalized",
        "challenge_version": CHALLENGE_VERSION,
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "counts": counts,
        "files": {
            "review_decisions_final": final_review_path.name,
            "accepted_queries": accepted_path.name,
            "accepted_cross_model_queries": accepted_core_path.name,
            "accepted_granite_extension_queries": accepted_extension_path.name,
            "rejected_queries": rejected_path.name,
            "accepted_documents": accepted_documents_path.name,
        },
        "output_sha256": {
            path.name: _file_sha256(path)
            for path in [
                final_review_path,
                accepted_path,
                accepted_core_path,
                accepted_extension_path,
                rejected_path,
                accepted_documents_path,
            ]
        },
        "interpretation_notes": [
            "The adjudication was completed before challenge retrieval analysis.",
            "The accepted set is secondary and does not alter the primary sample.",
            "Accepted documents may contain non-challenge questions; downstream "
            "challenge analysis filters evaluation outputs to accepted query IDs.",
        ],
    }
    final_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "CHALLENGE_DIRNAME",
    "CHALLENGE_VERSION",
    "freeze_cross_section_challenge",
    "finalize_cross_section_challenge",
]
