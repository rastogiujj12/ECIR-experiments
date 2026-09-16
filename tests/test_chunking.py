import re

from sclc.config import DocumentConfig
from sclc.data.chunking import (
    build_continuous_units,
    build_section_bounded_units,
    build_top_level_sections,
)
from sclc.data.reconstruct import reconstruct_document
from sclc.data.relevance import build_relevance_judgements, prepare_queries


class WhitespaceTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        matches = list(re.finditer(r"\S+", text))
        output: dict[str, object] = {
            "input_ids": list(range(len(matches))),
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = [match.span() for match in matches]
        return output


class WhitespaceAwareTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        matches = list(re.finditer(r"\S+|\s+", text))
        output: dict[str, object] = {
            "input_ids": list(range(len(matches))),
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = [match.span() for match in matches]
        return output


def record() -> dict:
    return {
        "id": "paper-1",
        "title": "Chunking Test",
        "abstract": "Abstract evidence is here.",
        "full_text": {
            "section_name": [
                "Methods ::: Data",
                "Methods ::: Training",
                "Results",
                "References",
            ],
            "paragraphs": [
                ["alpha beta gamma delta epsilon zeta eta theta"],
                ["iota kappa lambda mu nu xi omicron pi"],
                ["result evidence appears in this paragraph"],
                ["reference content"],
            ],
        },
        "qas": {
            "question": ["Where does the result evidence appear?"],
            "question_id": ["q1"],
            "answers": [
                {
                    "annotation_id": ["a1"],
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["result evidence"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["result evidence appears in this paragraph"],
                            "highlighted_evidence": [
                                "result evidence appears in this paragraph"
                            ],
                        }
                    ],
                }
            ],
        },
    }


def test_section_bounded_units_do_not_cross_top_level_boundaries() -> None:
    document = reconstruct_document(record(), "train", DocumentConfig())
    tokenizer = WhitespaceTokenizer()
    parents = build_top_level_sections(document, "cross_model_core")

    # Abstract, Methods (two subsection records merged), and Results.
    assert [parent.heading for parent in parents] == ["Abstract", "Methods", "Results"]

    continuous, continuous_stats = build_continuous_units(
        document,
        "cross_model_core",
        tokenizer,
        chunk_size_tokens=12,
        overlap_tokens=0,
    )
    section_bounded, section_stats = build_section_bounded_units(
        document,
        "cross_model_core",
        tokenizer,
        chunk_size_tokens=12,
        overlap_tokens=0,
        parent_sections=parents,
    )

    assert continuous_stats.expected_tokens == continuous_stats.emitted_tokens
    assert section_stats.expected_tokens == section_stats.emitted_tokens
    assert any(len(unit.overlapping_parent_section_ids) > 1 for unit in continuous)
    assert all(
        unit.overlapping_parent_section_ids == [unit.parent_section_id]
        for unit in section_bounded
    )
    assert all(unit.token_count <= 12 for unit in continuous + section_bounded)


def test_relevance_is_mapped_for_both_segmentation_plans() -> None:
    document = reconstruct_document(record(), "train", DocumentConfig())
    tokenizer = WhitespaceTokenizer()
    parents = build_top_level_sections(document, "cross_model_core")
    continuous, _ = build_continuous_units(
        document,
        "cross_model_core",
        tokenizer,
        chunk_size_tokens=8,
        overlap_tokens=0,
    )
    section_bounded, _ = build_section_bounded_units(
        document,
        "cross_model_core",
        tokenizer,
        chunk_size_tokens=8,
        overlap_tokens=0,
        parent_sections=parents,
    )
    queries = prepare_queries(document, "cross_model_core")

    continuous_qrels = build_relevance_judgements(queries, continuous)
    section_qrels = build_relevance_judgements(queries, section_bounded)

    assert len(queries) == 1
    assert continuous_qrels
    assert section_qrels
    assert {qrel.segmentation_plan for qrel in continuous_qrels} == {"continuous"}
    assert {qrel.segmentation_plan for qrel in section_qrels} == {"section_bounded"}


def test_paragraph_overlap_requires_non_whitespace_content() -> None:
    from sclc.data.chunking import _overlapping_paragraph_ids
    from sclc.data.schema import CharacterSpan, DocumentRecord, ParagraphRecord

    document = DocumentRecord(
        document_id="d",
        split="test",
        title="",
        abstract="",
        text="alpha   beta",
        sections=[],
        paragraphs=[
            ParagraphRecord(
                paragraph_id="p1",
                section_id="s1",
                source_section_index=0,
                source_paragraph_index=0,
                text="alpha",
                span=CharacterSpan(start=0, end=5),
            ),
            ParagraphRecord(
                paragraph_id="p2",
                section_id="s2",
                source_section_index=1,
                source_paragraph_index=0,
                text="beta",
                span=CharacterSpan(start=8, end=12),
            ),
        ],
        queries=[],
    )
    assert _overlapping_paragraph_ids(CharacterSpan(start=5, end=8), document) == []
    assert _overlapping_paragraph_ids(CharacterSpan(start=5, end=9), document) == ["p2"]


def test_trailing_whitespace_token_does_not_form_retrieval_unit() -> None:
    document = reconstruct_document(
        {
            "id": "paper-whitespace",
            "title": "Trailing whitespace",
            "abstract": "",
            "full_text": {
                "section_name": ["Results"],
                "paragraphs": [["alpha beta "]],
            },
            "qas": {
                "question": [],
                "question_id": [],
                "answers": [],
            },
        },
        "train",
        DocumentConfig(),
    )
    tokenizer = WhitespaceAwareTokenizer()
    parents = build_top_level_sections(document, "cross_model_core")

    units, stats = build_section_bounded_units(
        document,
        "cross_model_core",
        tokenizer,
        chunk_size_tokens=2,
        overlap_tokens=0,
        parent_sections=parents,
    )

    assert units
    assert all(unit.text.strip() for unit in units)
    assert all(unit.token_count <= 2 for unit in units)
    assert stats.expected_tokens == stats.emitted_tokens
