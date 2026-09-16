from sclc.analysis.evidence_structure import (
    MULTI_PARAGRAPH_CROSS_SECTION,
    MULTI_PARAGRAPH_SAME_SECTION,
    SINGLE_PARAGRAPH,
    classify_evidence_structure,
)
from sclc.data.schema import EvidenceSetRecord, PreparedQueryRecord


def make_query(
    query_id: str,
    evidence_sets: list[list[str]],
    *,
    document_id: str = "d1",
    analysis_set: str = "cross_model_core",
) -> PreparedQueryRecord:
    return PreparedQueryRecord(
        query_id=query_id,
        document_id=document_id,
        split="test",
        analysis_set=analysis_set,
        question=f"Question {query_id}",
        evidence_union_paragraph_ids=sorted(
            {paragraph for evidence_set in evidence_sets for paragraph in evidence_set}
        ),
        evidence_sets=[
            EvidenceSetRecord(
                evidence_set_id=f"{query_id}:set:{index}",
                paragraph_ids=evidence_set,
            )
            for index, evidence_set in enumerate(evidence_sets)
        ],
    )


def test_classification_uses_minimal_complete_acceptable_set() -> None:
    paragraph_to_parent = {"p1": "s1", "p2": "s1", "p3": "s2"}

    single = classify_evidence_structure(
        make_query("q1", [["p1"], ["p2", "p3"]]),
        paragraph_to_parent,
    )
    assert single["evidence_structure"] == SINGLE_PARAGRAPH
    assert single["cross_section_possible"] is True
    assert single["cross_section_required"] is False

    same = classify_evidence_structure(
        make_query("q2", [["p1", "p2"]]),
        paragraph_to_parent,
    )
    assert same["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION

    cross = classify_evidence_structure(
        make_query("q3", [["p1", "p3"]]),
        paragraph_to_parent,
    )
    assert cross["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
    assert cross["cross_section_required"] is True
    assert cross["cross_section_required_among_minimal_sets"] is True

    ambiguous = classify_evidence_structure(
        make_query("q4", [["p1", "p2"], ["p1", "p3"]]),
        paragraph_to_parent,
    )
    assert ambiguous["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION
    assert ambiguous["minimal_structure_ambiguous"] is True
    assert (
        ambiguous["minimal_structure_variants"]
        == "multi_paragraph_same_section|multi_paragraph_cross_section"
    )

