import json
from pathlib import Path

import pandas as pd

from sclc.analysis.qasper_collection import audit_qasper_collection
from sclc.config import AppConfig
from sclc.data.io import write_documents_jsonl
from sclc.data.schema import (
    AnswerAnnotation,
    CharacterSpan,
    DocumentRecord,
    EvidenceItem,
    ParagraphRecord,
    QueryRecord,
    SectionRecord,
)


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


def evidence(paragraph_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        text=text,
        is_float=False,
        matched_paragraph_ids=[paragraph_id],
        alignment_status="matched_unique",
    )


def answer(*items: EvidenceItem) -> AnswerAnnotation:
    return AnswerAnnotation(evidence=list(items))


def make_document(
    document_id: str,
    *,
    split: str,
) -> DocumentRecord:
    text = "A" * 120
    section_1 = f"{document_id}:section:1"
    section_2 = f"{document_id}:section:2"
    p1 = f"{document_id}:p1"
    p2 = f"{document_id}:p2"
    p3 = f"{document_id}:p3"
    paragraphs = [
        ParagraphRecord(
            paragraph_id=p1,
            section_id=section_1,
            source_section_index=0,
            source_paragraph_index=0,
            text="method one",
            span=CharacterSpan(start=0, end=20),
        ),
        ParagraphRecord(
            paragraph_id=p2,
            section_id=section_1,
            source_section_index=0,
            source_paragraph_index=1,
            text="method two",
            span=CharacterSpan(start=25, end=45),
        ),
        ParagraphRecord(
            paragraph_id=p3,
            section_id=section_2,
            source_section_index=1,
            source_paragraph_index=0,
            text="result one",
            span=CharacterSpan(start=65, end=85),
        ),
    ]
    sections = [
        SectionRecord(
            section_id=section_1,
            source_section_index=0,
            heading="Methods",
            top_level_heading="Methods",
            hierarchy=["Methods"],
            span=CharacterSpan(start=0, end=55),
            paragraph_ids=[p1, p2],
        ),
        SectionRecord(
            section_id=section_2,
            source_section_index=1,
            heading="Results",
            top_level_heading="Results",
            hierarchy=["Results"],
            span=CharacterSpan(start=60, end=100),
            paragraph_ids=[p3],
        ),
    ]
    queries = [
        QueryRecord(
            query_id=f"{document_id}:single",
            question="Single?",
            answers=[answer(evidence(p1, "method one"))],
        ),
        QueryRecord(
            query_id=f"{document_id}:same",
            question="Same section?",
            answers=[
                answer(
                    evidence(p1, "method one"),
                    evidence(p2, "method two"),
                )
            ],
        ),
        QueryRecord(
            query_id=f"{document_id}:cross",
            question="Cross section?",
            answers=[
                answer(
                    evidence(p1, "method one"),
                    evidence(p3, "result one"),
                )
            ],
        ),
        QueryRecord(
            query_id=f"{document_id}:alternative",
            question="Alternative?",
            answers=[
                answer(evidence(p1, "method one")),
                answer(
                    evidence(p2, "method two"),
                    evidence(p3, "result one"),
                ),
            ],
        ),
    ]
    return DocumentRecord(
        document_id=document_id,
        split=split,
        title=f"Paper {document_id}",
        abstract="",
        text=text,
        sections=sections,
        paragraphs=paragraphs,
        queries=queries,
    )


def test_full_collection_audit_separates_strict_candidates_and_sample_overlap(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config.paths.processed_dir.mkdir(parents=True)
    write_documents_jsonl(
        [
            make_document("selected", split="test"),
            make_document("newtest", split="test"),
            make_document("validation", split="validation"),
        ],
        config.paths.processed_dir / "documents.jsonl",
    )

    config.paths.profile_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "document_id": "selected",
                "granite_tokens": 1000,
                "jina_tokens": 1000,
                "eligibility_group": "cross_model_core",
            },
            {
                "document_id": "newtest",
                "granite_tokens": 1200,
                "jina_tokens": 1200,
                "eligibility_group": "cross_model_core",
            },
            {
                "document_id": "validation",
                "granite_tokens": 1300,
                "jina_tokens": 1300,
                "eligibility_group": "cross_model_core",
            },
        ]
    ).to_csv(config.paths.profile_dir / "document_lengths.csv", index=False)

    config.paths.subset_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "document_id": "selected",
                "analysis_set": "cross_model_core",
                "length_stratum": "length_1",
            }
        ]
    ).to_csv(config.paths.subset_dir / "selected_documents.csv", index=False)

    manifest = audit_qasper_collection(config)
    output_dir = config.paths.analysis_dir / "qasper_collection_audit"
    all_queries = pd.read_csv(output_dir / manifest["files"]["all_queries"])

    assert len(all_queries) == 12
    strict = all_queries[all_queries["strict_cross_section_required"]]
    assert len(strict) == 3
    assert set(strict["candidate_status"]) == {
        "already_in_current_experiment",
        "new_test_candidate",
        "validation_excluded_from_new_challenge",
    }

    new_candidate = strict[strict["document_id"] == "newtest"].iloc[0]
    assert bool(new_candidate["eligible_new_cross_model_strict_candidate"])
    assert "Methods" in new_candidate["primary_evidence_section_headings"]
    assert "Results" in new_candidate["primary_evidence_section_headings"]

    alternatives = all_queries[
        all_queries["query_id"] == "newtest:alternative"
    ].iloc[0]
    assert alternatives["evidence_structure"] == "single_paragraph"
    assert bool(alternatives["cross_section_possible"])
    assert not bool(alternatives["strict_cross_section_required"])

    assert manifest["strict_cross_section_query_count"] == 3
    assert manifest["eligible_new_cross_model_strict_candidate_count"] == 1
    loaded_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert loaded_manifest["configuration_fingerprint"]
