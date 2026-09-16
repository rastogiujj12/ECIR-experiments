import json
from pathlib import Path

import pandas as pd

from sclc.analysis.cross_section_challenge import freeze_cross_section_challenge
from sclc.config import AppConfig


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 42},
            "paths": {
                "processed_dir": tmp_path / "processed",
                "profile_dir": tmp_path / "profiles",
                "subset_dir": tmp_path / "subsets",
                "retrieval_unit_dir": tmp_path / "retrieval_units",
                "encoding_dir": tmp_path / "encodings",
                "ranking_dir": tmp_path / "rankings",
                "evaluation_dir": tmp_path / "evaluation",
                "analysis_dir": tmp_path / "analysis",
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 128,
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
        }
    )


def candidate_row(index: int, group: str, status: str) -> dict[str, object]:
    query_id = f"query-{index:03d}"
    document_id = f"doc-{index // 2:03d}"
    return {
        "query_id": query_id,
        "document_id": document_id,
        "split": "test",
        "title": f"Paper {index}",
        "question": f"Question {index}?",
        "strict_cross_section_required": True,
        "eligibility_group": group,
        "candidate_status": status,
        "selected_document": status == "already_in_current_experiment",
        "selected_analysis_set": group if status == "already_in_current_experiment" else "",
        "minimum_evidence_paragraph_count": 2,
        "minimum_top_level_section_count_among_minimal_sets": 2,
        "evidence_set_count": 1,
        "primary_evidence_set_id": f"{query_id}:set:0",
        "primary_evidence_paragraph_ids": f"{document_id}:p1|{document_id}:p2",
        "primary_evidence_section_ids": f"{document_id}:s1|{document_id}:s2",
        "primary_evidence_section_headings": "Methods|Results",
        "primary_evidence_texts_json": json.dumps(["method evidence", "result evidence"]),
        "evidence_sets_detailed_json": json.dumps(
            [
                {
                    "evidence_set_id": f"{query_id}:set:0",
                    "paragraph_ids": [f"{document_id}:p1", f"{document_id}:p2"],
                    "top_level_section_ids": [f"{document_id}:s1", f"{document_id}:s2"],
                    "top_level_section_headings": ["Methods", "Results"],
                    "paragraph_texts": ["method evidence", "result evidence"],
                }
            ]
        ),
        "granite_tokens": 1000 + index,
        "jina_tokens": 1000 + index if group == "cross_model_core" else 9000 + index,
    }


def test_challenge_freeze_creates_blinded_review_and_fixed_subsets(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    audit_dir = config.paths.analysis_dir / "qasper_collection_audit"
    audit_dir.mkdir(parents=True)

    rows = []
    for index in range(42):
        status = "already_in_current_experiment" if index < 5 else "new_test_candidate"
        rows.append(candidate_row(index, "cross_model_core", status))
    for index in range(42, 53):
        status = "already_in_current_experiment" if index == 42 else "new_test_candidate"
        rows.append(candidate_row(index, "granite_extended", status))

    pd.DataFrame(rows).to_csv(
        audit_dir / "qasper_cross_section_candidates.csv", index=False
    )
    (audit_dir / "manifest.json").write_text(
        json.dumps({"configuration_fingerprint": "audit-fingerprint"}),
        encoding="utf-8",
    )

    manifest = freeze_cross_section_challenge(config)
    output_dir = config.paths.analysis_dir / "qasper_cross_section_challenge"

    assert manifest["counts"] == {
        "all_strict_test": 53,
        "cross_model_complete": 42,
        "cross_model_unseen": 37,
        "granite_extension": 11,
    }
    assert manifest["document_counts"]["all_strict_test"] == 27

    review = pd.read_csv(output_dir / "blinded_review_sheet.csv")
    assert len(review) == 53
    assert "query_id" not in review.columns
    assert "document_id" not in review.columns
    assert "candidate_status" not in review.columns
    assert "eligibility_group" not in review.columns
    assert review["review_id"].is_unique

    key = pd.read_csv(output_dir / "review_key.csv")
    assert len(key) == 53
    assert set(review["review_id"]) == set(key["review_id"])

    complete = pd.read_csv(output_dir / "cross_model_complete.csv")
    unseen = pd.read_csv(output_dir / "cross_model_unseen.csv")
    extension = pd.read_csv(output_dir / "granite_extension.csv")
    assert len(complete) == 42
    assert len(unseen) == 37
    assert len(extension) == 11
    assert unseen["candidate_status"].eq("new_test_candidate").all()
    assert extension["eligibility_group"].eq("granite_extended").all()

    cached = freeze_cross_section_challenge(config)
    assert cached["configuration_fingerprint"] == manifest["configuration_fingerprint"]


def test_finalize_cross_section_challenge_validates_review_and_writes_sample(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    audit_dir = config.paths.analysis_dir / "qasper_collection_audit"
    audit_dir.mkdir(parents=True)

    rows = []
    for index in range(42):
        status = "already_in_current_experiment" if index < 5 else "new_test_candidate"
        rows.append(candidate_row(index, "cross_model_core", status))
    for index in range(42, 53):
        status = "already_in_current_experiment" if index == 42 else "new_test_candidate"
        rows.append(candidate_row(index, "granite_extended", status))
    pd.DataFrame(rows).to_csv(
        audit_dir / "qasper_cross_section_candidates.csv", index=False
    )
    (audit_dir / "manifest.json").write_text(
        json.dumps({"configuration_fingerprint": "audit-fingerprint"}),
        encoding="utf-8",
    )
    freeze_cross_section_challenge(config)

    profile = pd.DataFrame(
        {
            "document_id": [row["document_id"] for row in rows],
            "split": "test",
            "title": [row["title"] for row in rows],
            "character_count": 1000,
            "section_count": 2,
            "paragraph_count": 3,
            "question_count": 1,
            "usable_question_count": 1,
            "granite_tokens": [row["granite_tokens"] for row in rows],
            "jina_tokens": [row["jina_tokens"] for row in rows],
            "eligibility_group": [row["eligibility_group"] for row in rows],
        }
    ).drop_duplicates("document_id")
    config.paths.profile_dir.mkdir(parents=True)
    profile.to_csv(config.paths.profile_dir / "document_lengths.csv", index=False)

    review_path = tmp_path / "review_decisions.csv"
    review = pd.read_csv(
        config.paths.analysis_dir
        / "qasper_cross_section_challenge"
        / "blinded_review_sheet.csv"
    )
    review["include"] = "no"
    review["joint_evidence_required"] = "no"
    review["duplicate_support"] = "yes"
    review["section_mapping_valid"] = "yes"
    review["answerable_from_evidence"] = "yes"
    review["rejection_reason"] = "one_section_is_actually_sufficient"
    review["reviewer_notes"] = "synthetic review"
    accepted_ids = set(review.loc[:2, "review_id"])
    accepted_mask = review["review_id"].isin(accepted_ids)
    review.loc[accepted_mask, "include"] = "yes"
    review.loc[accepted_mask, "joint_evidence_required"] = "yes"
    review.loc[accepted_mask, "duplicate_support"] = "no"
    review.loc[accepted_mask, "rejection_reason"] = ""
    review.to_csv(review_path, index=False)

    from sclc.analysis.cross_section_challenge import (
        finalize_cross_section_challenge,
    )

    manifest = finalize_cross_section_challenge(
        config,
        decisions_path=review_path,
    )
    assert manifest["counts"]["reviewed"] == 53
    assert manifest["counts"]["accepted"] == 3
    assert manifest["counts"]["rejected"] == 50

    output_dir = config.paths.analysis_dir / "qasper_cross_section_challenge"
    accepted = pd.read_csv(output_dir / "accepted_queries.csv")
    accepted_documents = pd.read_csv(
        output_dir / "accepted_documents.csv",
        dtype={"document_id": "string"},
    )
    assert len(accepted) == 3
    assert set(accepted_documents["analysis_set"]).issubset(
        {"cross_model_core", "granite_extended"}
    )
