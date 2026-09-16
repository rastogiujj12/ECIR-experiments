import pytest

from sclc.options import EmbeddingModel, RetrievalCondition


def test_conditions_are_strict_and_complete() -> None:
    assert [condition.value for condition in RetrievalCondition] == [
        "bm25",
        "fixed_dense",
        "section_isolated",
        "section_constrained",
        "global",
        "length_matched",
    ]
    with pytest.raises(ValueError):
        RetrievalCondition("section_constraned")


def test_models_are_strict() -> None:
    assert [model.value for model in EmbeddingModel] == ["granite", "jina"]
    with pytest.raises(ValueError):
        EmbeddingModel("granitte")
