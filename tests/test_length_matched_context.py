from __future__ import annotations

from sclc.data.context_windows import build_length_matched_windows
from sclc.data.schema import CharacterSpan, RetrievalUnitRecord, TopLevelSectionRecord


class FakeTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs):
        offsets = []
        ids = []
        cursor = 0
        for token in text.split(" "):
            start = text.find(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            ids.append(len(ids))
            cursor = end + 1
        return {"input_ids": ids, "offset_mapping": offsets}


def unit(document_id: str, start: int, end: int, parent: str) -> RetrievalUnitRecord:
    return RetrievalUnitRecord(
        retrieval_unit_id=f"{document_id}:u",
        document_id=document_id,
        analysis_set="cross_model_core",
        segmentation_plan="section_bounded",
        unit_index=0,
        span=CharacterSpan(start=start, end=end),
        text="one two",
        token_count=2,
        scope_token_start=0,
        scope_token_end=2,
        parent_section_id=parent,
    )


def test_length_matched_window_preserves_section_length_and_can_cross_boundary() -> None:
    text = "a a a a one two b b b b b b"
    parent_end = text.index("b")
    parent = TopLevelSectionRecord(
        parent_section_id="doc:top_section:0000",
        document_id="doc",
        analysis_set="cross_model_core",
        heading="A",
        span=CharacterSpan(start=0, end=parent_end),
        source_section_ids=[],
        text=text[:parent_end],
    )
    # Parent contains 6 tokens. A centred length-matched window is also 6
    # tokens here, but its target is not tied to the section boundary.
    target_start = text.index("one")
    target_end = text.index("two") + len("two")
    target = unit("doc", target_start, target_end, parent.parent_section_id)

    result = build_length_matched_windows(
        document_text=text,
        units=[target],
        parent_sections={parent.parent_section_id: parent},
        tokenizer=FakeTokenizer(),
    )
    window = result[target.retrieval_unit_id]

    assert window.canonical_token_count == 6
    assert window.text == "a a one two b b"
    assert window.target.start == window.text.index("one")
    assert window.target.end == window.text.index("two") + len("two")
    assert window.relative_position_preserved is True


def test_boundary_shift_keeps_target_inside_window() -> None:
    text = "one two a a a a b b b b"
    parent = TopLevelSectionRecord(
        parent_section_id="doc:top_section:0000",
        document_id="doc",
        analysis_set="cross_model_core",
        heading="A",
        span=CharacterSpan(start=0, end=7),
        source_section_ids=[],
        text=text[:7],
    )
    target_end = text.index("two") + len("two")
    target = unit("doc", 0, target_end, parent.parent_section_id)

    result = build_length_matched_windows(
        document_text=text,
        units=[target],
        parent_sections={parent.parent_section_id: parent},
        tokenizer=FakeTokenizer(),
    )
    window = result[target.retrieval_unit_id]

    assert window.canonical_token_count == 2
    assert window.target.start == 0
    assert window.target.end == target_end
