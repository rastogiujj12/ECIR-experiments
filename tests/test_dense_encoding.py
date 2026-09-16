from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sclc.config import AppConfig
from sclc.data.io import write_documents_jsonl
from sclc.data.context_windows import build_length_matched_windows
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import (
    CharacterSpan,
    DocumentRecord,
    PreparedQueryRecord,
    RetrievalUnitRecord,
    TopLevelSectionRecord,
)
from sclc.encoding.dense import TargetSpan, _pool_target_spans, build_dense_encoding
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import encoding_dir, retrieval_unit_dir


class FakeRuntime:
    dimension = 3

    def __init__(self) -> None:
        self.task = "passage"
        self.independent_calls: list[tuple[str, list[str]]] = []
        self.contextual_calls: list[tuple[str, list[TargetSpan]]] = []

    def set_task(self, task: str) -> None:
        self.task = task

    def encode_independent(self, texts: list[str]) -> np.ndarray:
        self.independent_calls.append((self.task, list(texts)))
        task_value = 1.0 if self.task == "passage" else 2.0
        return np.asarray(
            [[float(len(text)), task_value, float(index)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )

    def encode_contextual(self, text: str, targets: list[TargetSpan]) -> np.ndarray:
        self.contextual_calls.append((text, list(targets)))
        return np.asarray(
            [[float(target.start), float(target.end), float(len(text))] for target in targets],
            dtype=np.float32,
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
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 512,
                "overlap_tokens": 0,
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "dense": {"batch_size": 4},
        }
    )


def make_unit(
    unit_id: str,
    *,
    segmentation_plan: str,
    index: int,
    start: int,
    end: int,
    text: str,
    parent_section_id: str | None = None,
) -> RetrievalUnitRecord:
    return RetrievalUnitRecord(
        retrieval_unit_id=unit_id,
        document_id="paper-1",
        analysis_set="cross_model_core",
        segmentation_plan=segmentation_plan,
        unit_index=index,
        span=CharacterSpan(start=start, end=end),
        text=text,
        token_count=2,
        scope_token_start=index * 2,
        scope_token_end=index * 2 + 2,
        parent_section_id=parent_section_id,
        overlapping_parent_section_ids=[parent_section_id] if parent_section_id else [],
    )


def write_inputs(config: AppConfig) -> None:
    text = "alpha beta gamma delta"
    write_documents_jsonl(
        [
            DocumentRecord(
                document_id="paper-1",
                split="test",
                title="Paper",
                abstract="",
                text=text,
                sections=[],
                paragraphs=[],
                queries=[],
            )
        ],
        config.paths.processed_dir / "documents.jsonl",
    )
    write_models_jsonl(
        [
            make_unit(
                "continuous-1",
                segmentation_plan="continuous",
                index=0,
                start=0,
                end=len(text),
                text=text,
            )
        ],
        retrieval_unit_dir(config, 512) / "continuous_units.jsonl",
    )
    write_models_jsonl(
        [
            make_unit(
                "section-1",
                segmentation_plan="section_bounded",
                index=0,
                start=0,
                end=10,
                text="alpha beta",
                parent_section_id="parent-1",
            ),
            make_unit(
                "section-2",
                segmentation_plan="section_bounded",
                index=1,
                start=11,
                end=len(text),
                text="gamma delta",
                parent_section_id="parent-1",
            ),
        ],
        retrieval_unit_dir(config, 512) / "section_bounded_units.jsonl",
    )
    write_models_jsonl(
        [
            TopLevelSectionRecord(
                parent_section_id="parent-1",
                document_id="paper-1",
                analysis_set="cross_model_core",
                heading="Body",
                span=CharacterSpan(start=0, end=len(text)),
                text=text,
            )
        ],
        retrieval_unit_dir(config, 512) / "top_level_sections.jsonl",
    )
    write_models_jsonl(
        [
            PreparedQueryRecord(
                query_id="query-1",
                document_id="paper-1",
                split="test",
                analysis_set="cross_model_core",
                question="What follows beta?",
            )
        ],
        retrieval_unit_dir(config, 512) / "queries.jsonl",
    )


@pytest.mark.parametrize(
    ("condition", "expected_ids", "contextual"),
    [
        (RetrievalCondition.FIXED_DENSE, ["continuous-1"], False),
        (RetrievalCondition.SECTION_ISOLATED, ["section-1", "section-2"], False),
        (RetrievalCondition.SECTION_CONSTRAINED, ["section-1", "section-2"], True),
        (RetrievalCondition.GLOBAL, ["section-1", "section-2"], True),
    ],
)
def test_dense_conditions_write_passage_and_shared_query_embeddings(
    tmp_path: Path,
    condition: RetrievalCondition,
    expected_ids: list[str],
    contextual: bool,
) -> None:
    config = make_config(tmp_path)
    write_inputs(config)
    runtime = FakeRuntime()

    manifest = build_dense_encoding(
        config,
        condition=condition,
        model_key=EmbeddingModel.GRANITE,
        chunk_size=512,
        runtime_factory=lambda _config, _model: runtime,
    )

    assert manifest["unit_count"] == len(expected_ids)
    assert manifest["query_count"] == 1
    document_path = (
        encoding_dir(config, 512)
        / condition.value
        / "granite"
        / manifest["documents"][0]["file"]
    )
    with np.load(document_path, allow_pickle=False) as payload:
        assert payload["retrieval_unit_ids"].tolist() == expected_ids
        assert payload["embeddings"].shape == (len(expected_ids), 3)

    query_path = config.paths.encoding_dir / "queries" / "granite" / "queries.npz"
    with np.load(query_path, allow_pickle=False) as payload:
        assert payload["query_ids"].tolist() == ["query-1"]
        assert payload["embeddings"].shape == (1, 3)

    assert bool(runtime.contextual_calls) is contextual
    assert runtime.independent_calls[-1] == ("query", ["What follows beta?"])


def test_target_span_pooling_uses_overlapping_content_tokens_only() -> None:
    hidden = torch.tensor(
        [
            [
                [100.0, 100.0],
                [1.0, 3.0],
                [3.0, 5.0],
                [7.0, 9.0],
                [200.0, 200.0],
            ]
        ]
    )
    attention = torch.ones((1, 5), dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.long)
    offsets = torch.tensor([[[0, 0], [0, 5], [6, 10], [11, 16], [0, 0]]])

    pooled = _pool_target_spans(
        hidden,
        attention_mask=attention,
        special_tokens_mask=special,
        offset_mapping=offsets,
        targets=[TargetSpan("u1", 0, 10), TargetSpan("u2", 11, 16)],
        normalize=False,
    )

    assert torch.equal(pooled[0], torch.tensor([2.0, 4.0]))
    assert torch.equal(pooled[1], torch.tensor([7.0, 9.0]))


def test_query_cache_rejects_changed_questions(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_inputs(config)
    runtime = FakeRuntime()
    def factory(_config, _model):
        return runtime

    build_dense_encoding(
        config,
        condition=RetrievalCondition.FIXED_DENSE,
        model_key=EmbeddingModel.GRANITE,
        chunk_size=512,
        runtime_factory=factory,
    )
    write_models_jsonl(
        [
            PreparedQueryRecord(
                query_id="query-1",
                document_id="paper-1",
                split="test",
                analysis_set="cross_model_core",
                question="Which token follows beta?",
            )
        ],
        retrieval_unit_dir(config, 512) / "queries.jsonl",
    )

    with pytest.raises(RuntimeError, match="Cached query embeddings"):
        build_dense_encoding(
            config,
            condition=RetrievalCondition.FIXED_DENSE,
            model_key=EmbeddingModel.GRANITE,
            chunk_size=512,
            runtime_factory=factory,
        )


def test_passage_cache_rejects_changed_global_context(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_inputs(config)
    runtime = FakeRuntime()
    def factory(_config, _model):
        return runtime

    build_dense_encoding(
        config,
        condition=RetrievalCondition.GLOBAL,
        model_key=EmbeddingModel.GRANITE,
        chunk_size=512,
        runtime_factory=factory,
    )
    write_documents_jsonl(
        [
            DocumentRecord(
                document_id="paper-1",
                split="test",
                title="Paper",
                abstract="",
                text="alpha beta gamma delta appendix",
                sections=[],
                paragraphs=[],
                queries=[],
            )
        ],
        config.paths.processed_dir / "documents.jsonl",
    )

    with pytest.raises(RuntimeError, match="retrieval inputs"):
        build_dense_encoding(
            config,
            condition=RetrievalCondition.GLOBAL,
            model_key=EmbeddingModel.GRANITE,
            chunk_size=512,
            runtime_factory=factory,
        )


def test_dense_attention_backend_is_part_of_cache_fingerprint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_inputs(config)
    first_runtime = FakeRuntime()
    build_dense_encoding(
        config,
        condition=RetrievalCondition.FIXED_DENSE,
        model_key=EmbeddingModel.GRANITE,
        chunk_size=512,
        runtime_factory=lambda _config, _model: first_runtime,
    )

    changed = config.model_copy(deep=True)
    changed.dense.attn_implementation = "eager"
    with pytest.raises(RuntimeError, match="Re-run with --overwrite"):
        build_dense_encoding(
            changed,
            condition=RetrievalCondition.FIXED_DENSE,
            model_key=EmbeddingModel.GRANITE,
            chunk_size=512,
            runtime_factory=lambda _config, _model: FakeRuntime(),
        )


def test_dense_attention_backend_validation(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.dense.attn_implementation == "sdpa"

    flex_payload = config.model_dump(mode="python")
    flex_payload["dense"]["attn_implementation"] = "flex_attention"
    flex_config = AppConfig.model_validate(flex_payload)
    assert flex_config.dense.attn_implementation == "flex_attention"

    invalid_payload = config.model_dump(mode="python")
    invalid_payload["dense"]["attn_implementation"] = "not-a-backend"
    with pytest.raises(ValueError, match="attn_implementation"):
        AppConfig.model_validate(invalid_payload)


def test_partial_passage_cache_resumes_after_interruption(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    documents = [
        DocumentRecord(
            document_id="paper-1",
            split="test",
            title="Paper 1",
            abstract="",
            text="alpha beta",
            sections=[],
            paragraphs=[],
            queries=[],
        ),
        DocumentRecord(
            document_id="paper-2",
            split="test",
            title="Paper 2",
            abstract="",
            text="gamma delta",
            sections=[],
            paragraphs=[],
            queries=[],
        ),
    ]
    write_documents_jsonl(documents, config.paths.processed_dir / "documents.jsonl")

    units = [
        RetrievalUnitRecord(
            retrieval_unit_id="paper-1-section-1",
            document_id="paper-1",
            analysis_set="cross_model_core",
            segmentation_plan="section_bounded",
            unit_index=0,
            span=CharacterSpan(start=0, end=len("alpha beta")),
            text="alpha beta",
            token_count=2,
            scope_token_start=0,
            scope_token_end=2,
            parent_section_id="paper-1-parent",
            overlapping_parent_section_ids=["paper-1-parent"],
        ),
        RetrievalUnitRecord(
            retrieval_unit_id="paper-2-section-1",
            document_id="paper-2",
            analysis_set="cross_model_core",
            segmentation_plan="section_bounded",
            unit_index=0,
            span=CharacterSpan(start=0, end=len("gamma delta")),
            text="gamma delta",
            token_count=2,
            scope_token_start=0,
            scope_token_end=2,
            parent_section_id="paper-2-parent",
            overlapping_parent_section_ids=["paper-2-parent"],
        ),
    ]
    write_models_jsonl(
        units,
        retrieval_unit_dir(config, 512) / "section_bounded_units.jsonl",
    )
    write_models_jsonl(
        [
            PreparedQueryRecord(
                query_id="query-1",
                document_id="paper-1",
                split="test",
                analysis_set="cross_model_core",
                question="Question one?",
            ),
            PreparedQueryRecord(
                query_id="query-2",
                document_id="paper-2",
                split="test",
                analysis_set="cross_model_core",
                question="Question two?",
            ),
        ],
        retrieval_unit_dir(config, 512) / "queries.jsonl",
    )

    class FailOnSecondDocument(FakeRuntime):
        def encode_contextual(self, text: str, targets: list[TargetSpan]) -> np.ndarray:
            if text == "gamma delta":
                raise RuntimeError("synthetic interruption")
            return super().encode_contextual(text, targets)

    failing_runtime = FailOnSecondDocument()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_dense_encoding(
            config,
            condition=RetrievalCondition.GLOBAL,
            model_key=EmbeddingModel.GRANITE,
            chunk_size=512,
            runtime_factory=lambda _config, _model: failing_runtime,
        )

    output_dir = encoding_dir(config, 512) / "global" / "granite"
    partial_manifest = output_dir / "partial_manifest.json"
    assert partial_manifest.exists()
    assert len(list((output_dir / "documents").glob("*.npz"))) == 1

    resumed_runtime = FakeRuntime()
    manifest = build_dense_encoding(
        config,
        condition=RetrievalCondition.GLOBAL,
        model_key=EmbeddingModel.GRANITE,
        chunk_size=512,
        runtime_factory=lambda _config, _model: resumed_runtime,
    )

    assert manifest["document_count"] == 2
    assert manifest["cached_document_count"] == 1
    assert len(resumed_runtime.contextual_calls) == 1
    assert resumed_runtime.contextual_calls[0][0] == "gamma delta"
    assert not partial_manifest.exists()
