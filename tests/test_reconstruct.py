from sclc.config import DocumentConfig
from sclc.data.reconstruct import reconstruct_document


def synthetic_record() -> dict:
    return {
        "id": "paper-1",
        "title": "A Test Paper",
        "abstract": "This paper studies retrieval.",
        "full_text": {
            "section_name": [
                "Introduction",
                "Method ::: Data",
                "References",
                "Appendix",
            ],
            "paragraphs": [
                ["First paragraph.", "Second paragraph."],
                ["The evidence paragraph."],
                ["Reference one.", "Reference two."],
                ["This should not be retained when stop_at_references is enabled."],
            ],
        },
        "qas": {
            "question": ["What is the evidence?"],
            "question_id": ["q1"],
            "answers": [
                {
                    "annotation_id": ["a1"],
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["evidence"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["The evidence paragraph."],
                            "highlighted_evidence": ["The evidence paragraph."],
                        }
                    ],
                }
            ],
        },
    }


def test_reconstruction_removes_references_and_aligns_evidence() -> None:
    document = reconstruct_document(
        synthetic_record(),
        split="train",
        config=DocumentConfig(),
    )

    assert "Reference one." not in document.text
    assert "Appendix" in document.text
    assert len(document.sections) == 4

    query = document.queries[0]
    evidence = query.answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"
    assert len(evidence.matched_paragraph_ids) == 1
    assert query.has_usable_textual_evidence


def test_offsets_recover_exact_paragraph_text() -> None:
    document = reconstruct_document(
        synthetic_record(),
        split="train",
        config=DocumentConfig(),
    )

    for paragraph in document.paragraphs:
        recovered = document.text[paragraph.span.start : paragraph.span.end]
        assert recovered == paragraph.text

    for section in document.sections:
        recovered = document.text[section.span.start : section.span.end]
        assert recovered.strip()
