from sclc.encoding.dense import condition_spec
from sclc.options import RetrievalCondition


def test_length_matched_condition_uses_section_units_and_linear_context() -> None:
    spec = condition_spec(RetrievalCondition.LENGTH_MATCHED)

    assert spec.unit_filename == "section_bounded_units.jsonl"
    assert spec.segmentation_plan == "section_bounded"
    assert spec.context_scope == "length_matched_linear_window"
    assert spec.independent_units is False
