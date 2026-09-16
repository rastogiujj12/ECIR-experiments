from sclc.data.retrieval_unit_io import query_type_coding_frame
from sclc.data.schema import (
    CharacterSpan,
    DocumentRecord,
    EvidenceSetRecord,
    ParagraphRecord,
    PreparedQueryRecord,
    SectionRecord,
)


def test_query_type_coding_sheet_includes_evidence_location() -> None:
    document = DocumentRecord(
        document_id="d1",
        split="test",
        title="Paper",
        abstract="",
        text="Methods text",
        sections=[
            SectionRecord(
                section_id="s1",
                source_section_index=0,
                heading="Data",
                top_level_heading="Methods",
                hierarchy=["Methods", "Data"],
                span=CharacterSpan(start=0, end=12),
                paragraph_ids=["p1"],
            )
        ],
        paragraphs=[
            ParagraphRecord(
                paragraph_id="p1",
                section_id="s1",
                source_section_index=0,
                source_paragraph_index=0,
                text="Methods text",
                span=CharacterSpan(start=0, end=12),
            )
        ],
        queries=[],
    )
    query = PreparedQueryRecord(
        query_id="q1",
        document_id="d1",
        split="test",
        analysis_set="cross_model_core",
        question="What data was used?",
        evidence_union_paragraph_ids=["p1"],
        evidence_sets=[EvidenceSetRecord(evidence_set_id="e1", paragraph_ids=["p1"])],
    )
    frame = query_type_coding_frame([document], [query])
    assert frame.loc[0, "evidence_section_headings"] == "Methods"
    assert frame.loc[0, "evidence_section_count"] == 1
    assert frame.loc[0, "query_type"] == ""


def test_uncertain_is_a_supported_query_type(tmp_path) -> None:
    import pandas as pd

    from sclc.config import AppConfig
    from sclc.data.query_types import load_query_types

    config = AppConfig.model_validate(
        {
            "project": {"seed": 42},
            "paths": {
                "processed_dir": tmp_path / "processed",
                "profile_dir": tmp_path / "profiles",
                "subset_dir": tmp_path / "subsets",
                "retrieval_unit_dir": tmp_path / "retrieval_units",
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {"canonical_tokenizer": "granite"},
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
        }
    )
    config.paths.retrieval_unit_dir.mkdir(parents=True)
    pd.DataFrame([{"query_id": "q1", "query_type": "uncertain"}]).to_csv(
        config.paths.retrieval_unit_dir / "query_types.csv", index=False
    )
    assert load_query_types(config, {"q1"}) == {"q1": "uncertain"}
