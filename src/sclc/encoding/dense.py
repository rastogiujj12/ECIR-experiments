from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sclc.config import AppConfig, DenseEncodingConfig, ModelProfileConfig
from sclc.data.io import read_documents_jsonl
from sclc.data.context_windows import build_length_matched_windows, load_canonical_tokenizer
from sclc.data.retrieval_unit_io import (
    read_prepared_queries,
    read_retrieval_units,
    read_top_level_sections,
)
from sclc.data.schema import (
    DocumentRecord,
    PreparedQueryRecord,
    RetrievalUnitRecord,
    TopLevelSectionRecord,
)
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import encoding_dir, retrieval_unit_dir

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TaskKind = Literal["passage", "query"]


@dataclass(frozen=True)
class TargetSpan:
    retrieval_unit_id: str
    start: int
    end: int


@dataclass(frozen=True)
class DenseConditionSpec:
    unit_filename: str
    segmentation_plan: str
    context_scope: str
    independent_units: bool


class EmbeddingRuntime(Protocol):
    dimension: int

    def set_task(self, task: TaskKind) -> None: ...

    def encode_independent(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_contextual(self, text: str, targets: Sequence[TargetSpan]) -> np.ndarray: ...


RuntimeFactory = Callable[[AppConfig, EmbeddingModel], EmbeddingRuntime]


def condition_spec(condition: RetrievalCondition) -> DenseConditionSpec:
    match condition:
        case RetrievalCondition.FIXED_DENSE:
            return DenseConditionSpec(
                unit_filename="continuous_units.jsonl",
                segmentation_plan="continuous",
                context_scope="retrieval_unit",
                independent_units=True,
            )
        case RetrievalCondition.SECTION_ISOLATED:
            return DenseConditionSpec(
                unit_filename="section_bounded_units.jsonl",
                segmentation_plan="section_bounded",
                context_scope="retrieval_unit",
                independent_units=True,
            )
        case RetrievalCondition.SECTION_CONSTRAINED:
            return DenseConditionSpec(
                unit_filename="section_bounded_units.jsonl",
                segmentation_plan="section_bounded",
                context_scope="top_level_section",
                independent_units=False,
            )
        case RetrievalCondition.GLOBAL:
            return DenseConditionSpec(
                unit_filename="section_bounded_units.jsonl",
                segmentation_plan="section_bounded",
                context_scope="complete_document",
                independent_units=False,
            )
        case RetrievalCondition.LENGTH_MATCHED:
            return DenseConditionSpec(
                unit_filename="section_bounded_units.jsonl",
                segmentation_plan="section_bounded",
                context_scope="length_matched_linear_window",
                independent_units=False,
            )
        case RetrievalCondition.BM25:
            raise ValueError("BM25 does not use the dense encoding pipeline")
        case _:
            raise ValueError(f"Unsupported dense condition: {condition!r}")


def _model_config(config: AppConfig, model: EmbeddingModel) -> ModelProfileConfig:
    return config.models.granite if model is EmbeddingModel.GRANITE else config.models.jina


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unsupported dense dtype: {requested}")
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _last_hidden_state(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return cast(torch.Tensor, hidden)
    return cast(torch.Tensor, output[0])


def _content_mask(
    *,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
) -> torch.Tensor:
    non_empty_offsets = offset_mapping[..., 1] > offset_mapping[..., 0]
    return attention_mask.bool() & ~special_tokens_mask.bool() & non_empty_offsets


def _mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
    denominator = expanded.sum(dim=1).clamp_min(1.0)
    pooled = (hidden_states * expanded).sum(dim=1) / denominator
    if normalize:
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled


def _pool_target_spans(
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
    targets: Sequence[TargetSpan],
    normalize: bool,
) -> torch.Tensor:
    if hidden_states.shape[0] != 1:
        raise ValueError("Target-span pooling expects one contextual scope at a time")

    base_mask = _content_mask(
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        offset_mapping=offset_mapping,
    )[0]
    offsets = offset_mapping[0]
    vectors: list[torch.Tensor] = []

    for target in targets:
        if not 0 <= target.start < target.end:
            raise ValueError(f"Invalid target span for {target.retrieval_unit_id}")
        overlap = (offsets[:, 0] < target.end) & (offsets[:, 1] > target.start)
        mask = base_mask & overlap
        if not torch.any(mask):
            raise RuntimeError(
                f"No model tokens overlap retrieval unit {target.retrieval_unit_id}"
            )
        pooled = hidden_states[0, mask].mean(dim=0)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=0)
        vectors.append(pooled)

    return torch.stack(vectors, dim=0)


class TransformerEmbeddingRuntime:
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        model: Any,
        model_config: ModelProfileConfig,
        dense_config: DenseEncodingConfig,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.dense_config = dense_config
        self.device = device
        self.dimension = int(model.config.hidden_size)
        self._active_adapter: str | None = None

    def set_task(self, task: TaskKind) -> None:
        adapter = (
            self.model_config.passage_adapter
            if task == "passage"
            else self.model_config.query_adapter
        )
        if adapter is None or adapter == self._active_adapter:
            return
        if not hasattr(self.model, "set_adapter"):
            raise RuntimeError(
                f"{self.model_config.model_id} requires adapter {adapter!r}, but the "
                "loaded model does not expose set_adapter()."
            )
        self.model.set_adapter(adapter)
        self._active_adapter = adapter

    def _forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key not in {"offset_mapping", "special_tokens_mask"}
        }
        with torch.inference_mode():
            output = self.model(**inputs)
        return _last_hidden_state(output)

    def _check_lengths(self, attention_mask: torch.Tensor) -> None:
        lengths = attention_mask.sum(dim=1)
        maximum = int(lengths.max().item())
        if maximum > self.model_config.max_document_tokens:
            raise RuntimeError(
                f"Tokenized scope has {maximum} tokens, exceeding the configured limit "
                f"of {self.model_config.max_document_tokens} for "
                f"{self.model_config.model_id}. No truncation is permitted."
            )

    def encode_independent(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        batch_size = self.dense_config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            self._check_lengths(encoded["attention_mask"])
            hidden = self._forward(dict(encoded))
            mask = _content_mask(
                attention_mask=encoded["attention_mask"].to(hidden.device),
                special_tokens_mask=encoded["special_tokens_mask"].to(hidden.device),
                offset_mapping=encoded["offset_mapping"].to(hidden.device),
            )
            pooled = _mean_pool(
                hidden,
                mask,
                normalize=self.dense_config.normalize,
            )
            batches.append(pooled.float().cpu().numpy())

        return np.concatenate(batches, axis=0)

    def encode_contextual(self, text: str, targets: Sequence[TargetSpan]) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        self._check_lengths(encoded["attention_mask"])
        token_count = int(encoded["attention_mask"].sum().item())
        try:
            hidden = self._forward(dict(encoded))
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory while encoding one contextual scope of "
                f"{token_count} tokens with attention backend "
                f"{self.dense_config.attn_implementation!r} on {self.device}. "
                "Contextual encoding already uses batch size 1, so lowering "
                "dense.batch_size will not fix this failure. Keep truncation and "
                "windowing disabled; instead use dense.attn_implementation='sdpa' "
                "or 'flash_attention_2', or run this condition on a GPU with more "
                "VRAM."
            ) from exc
        pooled = _pool_target_spans(
            hidden,
            attention_mask=encoded["attention_mask"].to(hidden.device),
            special_tokens_mask=encoded["special_tokens_mask"].to(hidden.device),
            offset_mapping=encoded["offset_mapping"].to(hidden.device),
            targets=targets,
            normalize=self.dense_config.normalize,
        )
        return pooled.float().cpu().numpy()


def load_runtime(config: AppConfig, model_key: EmbeddingModel) -> EmbeddingRuntime:
    from transformers import AutoModel, AutoTokenizer

    model_config = _model_config(config, model_key)
    device = _resolve_device(config.dense.device)
    dtype = _resolve_dtype(config.dense.dtype, device)

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.tokenizer_trust_remote_code,
        "cache_dir": str(config.paths.hf_cache_dir),
        "use_fast": True,
    }
    if model_config.revision is not None:
        tokenizer_kwargs["revision"] = model_config.revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        **tokenizer_kwargs,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{model_config.model_id} did not load a fast tokenizer. "
            "Target-span pooling requires character offsets."
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.model_trust_remote_code,
        "cache_dir": str(config.paths.hf_cache_dir),
        "dtype": dtype,
    }
    if config.dense.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = config.dense.attn_implementation
    if model_config.revision is not None:
        model_kwargs["revision"] = model_config.revision
    model = AutoModel.from_pretrained(
        model_config.model_id,
        **model_kwargs,
    )

    adapter_source = model_config.adapter_source or model_config.model_id
    adapters = {
        adapter
        for adapter in (model_config.passage_adapter, model_config.query_adapter)
        if adapter is not None
    }
    if adapters:
        if not hasattr(model, "load_adapter"):
            raise RuntimeError(
                f"{model_config.model_id} is configured with task adapters, but the "
                "loaded model does not expose load_adapter()."
            )
        for adapter in sorted(adapters):
            adapter_kwargs: dict[str, Any] = {
                "adapter_name": adapter,
                "adapter_kwargs": {"subfolder": adapter},
            }
            adapter_revision = model_config.adapter_revision or model_config.revision
            if adapter_revision is not None:
                adapter_kwargs["revision"] = adapter_revision
            model.load_adapter(adapter_source, **adapter_kwargs)

    model.to(device)
    model.eval()
    return TransformerEmbeddingRuntime(
        tokenizer=tokenizer,
        model=model,
        model_config=model_config,
        dense_config=config.dense,
        device=device,
    )


def _safe_document_name(document_id: str) -> str:
    digest = hashlib.sha1(document_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"document_{digest}.npz"


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cast(Any, arrays))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _records_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        encoded = canonical.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _passage_input_fingerprint(
    *,
    condition: RetrievalCondition,
    units: Sequence[RetrievalUnitRecord],
    documents: dict[str, DocumentRecord] | None,
    parent_sections: dict[str, TopLevelSectionRecord] | None,
    config: AppConfig,
) -> str:
    records: list[dict[str, Any]] = []
    for unit in sorted(
        units,
        key=lambda item: (item.document_id, item.unit_index, item.retrieval_unit_id),
    ):
        records.append(
            {
                "kind": "unit",
                "retrieval_unit_id": unit.retrieval_unit_id,
                "document_id": unit.document_id,
                "analysis_set": unit.analysis_set,
                "segmentation_plan": unit.segmentation_plan,
                "unit_index": unit.unit_index,
                "span": [unit.span.start, unit.span.end],
                "text": unit.text,
                "parent_section_id": unit.parent_section_id,
            }
        )

    if condition is RetrievalCondition.GLOBAL:
        if documents is None:
            raise RuntimeError("Prepared documents are required for global encoding")
        for document_id in sorted({unit.document_id for unit in units}):
            document = documents.get(document_id)
            if document is None:
                raise RuntimeError(f"Prepared document {document_id} is unavailable")
            records.append(
                {
                    "kind": "document_context",
                    "document_id": document_id,
                    "text": document.text,
                }
            )
    elif condition is RetrievalCondition.SECTION_CONSTRAINED:
        if parent_sections is None:
            raise RuntimeError("Top-level sections are required for section encoding")
        parent_ids = sorted(
            {
                unit.parent_section_id
                for unit in units
                if unit.parent_section_id is not None
            }
        )
        for parent_id in parent_ids:
            parent = parent_sections.get(parent_id)
            if parent is None:
                raise RuntimeError(f"Top-level section {parent_id} is unavailable")
            records.append(
                {
                    "kind": "section_context",
                    "parent_section_id": parent_id,
                    "document_id": parent.document_id,
                    "span": [parent.span.start, parent.span.end],
                    "text": parent.text,
                }
            )
    elif condition is RetrievalCondition.LENGTH_MATCHED:
        if documents is None or parent_sections is None:
            raise RuntimeError("Documents and top-level sections are required for LMLC")
        for document_id in sorted({unit.document_id for unit in units}):
            document = documents.get(document_id)
            if document is None:
                raise RuntimeError(f"Prepared document {document_id} is unavailable")
            document_units = [unit for unit in units if unit.document_id == document_id]
            windows = build_length_matched_windows(
                document_text=document.text,
                units=document_units,
                parent_sections=parent_sections,
                tokenizer=load_canonical_tokenizer(config),
            )
            for unit in document_units:
                window = windows[unit.retrieval_unit_id]
                records.append(
                    {
                        "kind": "length_matched_context",
                        "retrieval_unit_id": unit.retrieval_unit_id,
                        "document_id": document_id,
                        "span": [window.span.start, window.span.end],
                        "canonical_token_count": window.canonical_token_count,
                        "text": window.text,
                        "target": [window.target.start, window.target.end],
                    }
                )

    return _records_fingerprint(records)


def _query_input_fingerprint(queries: Sequence[PreparedQueryRecord]) -> str:
    return _records_fingerprint(
        [
            {
                "query_id": query.query_id,
                "document_id": query.document_id,
                "analysis_set": query.analysis_set,
                "question": query.question,
            }
            for query in sorted(queries, key=lambda item: item.query_id)
        ]
    )


def _configuration_payload(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model_key: EmbeddingModel,
    passage_input_fingerprint: str,
    chunk_size: int,
) -> dict[str, Any]:
    model_config = _model_config(config, model_key)
    spec = condition_spec(condition)
    return {
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model_key.value,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "passage_adapter": model_config.passage_adapter,
        "adapter_source": model_config.adapter_source,
        "adapter_revision": model_config.adapter_revision,
        "segmentation_plan": spec.segmentation_plan,
        "context_scope": spec.context_scope,
        "pooling": config.dense.pooling,
        "normalize": config.dense.normalize,
        "compute_dtype": config.dense.dtype,
        "attn_implementation": config.dense.attn_implementation,
        "output_dtype": config.dense.output_dtype,
        "passage_input_fingerprint": passage_input_fingerprint,
    }


def _eligible_analysis_set(model_key: EmbeddingModel, analysis_set: str) -> bool:
    return model_key is EmbeddingModel.GRANITE or analysis_set == "cross_model_core"


def _load_units(
    config: AppConfig,
    *,
    spec: DenseConditionSpec,
    model_key: EmbeddingModel,
    chunk_size: int,
) -> list[RetrievalUnitRecord]:
    path = retrieval_unit_dir(config, chunk_size) / spec.unit_filename
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc chunk` first.")
    return [
        unit
        for unit in read_retrieval_units(path)
        if _eligible_analysis_set(model_key, unit.analysis_set)
    ]


def _load_queries(
    config: AppConfig,
    model_key: EmbeddingModel,
    *,
    chunk_size: int,
) -> list[PreparedQueryRecord]:
    path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc chunk` first.")
    return [
        query
        for query in read_prepared_queries(path)
        if _eligible_analysis_set(model_key, query.analysis_set)
    ]


def _load_documents(config: AppConfig) -> dict[str, DocumentRecord]:
    path = config.paths.processed_dir / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc prepare` first.")
    return {document.document_id: document for document in read_documents_jsonl(path)}


def _load_parent_sections(
    config: AppConfig,
    *,
    chunk_size: int,
) -> dict[str, TopLevelSectionRecord]:
    path = retrieval_unit_dir(config, chunk_size) / "top_level_sections.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc chunk` first.")
    return {section.parent_section_id: section for section in read_top_level_sections(path)}


def _local_target(
    unit: RetrievalUnitRecord,
    *,
    scope_start: int,
    scope_text: str,
) -> TargetSpan:
    start = unit.span.start - scope_start
    end = unit.span.end - scope_start
    if not 0 <= start < end <= len(scope_text):
        raise RuntimeError(
            f"Retrieval unit {unit.retrieval_unit_id} falls outside its encoding scope"
        )
    if scope_text[start:end] != unit.text:
        raise RuntimeError(
            f"Scope text does not recover retrieval unit {unit.retrieval_unit_id}"
        )
    return TargetSpan(unit.retrieval_unit_id, start, end)


def _encode_document_units(
    *,
    condition: RetrievalCondition,
    spec: DenseConditionSpec,
    document_id: str,
    units: Sequence[RetrievalUnitRecord],
    runtime: EmbeddingRuntime,
    documents: dict[str, DocumentRecord] | None,
    parent_sections: dict[str, TopLevelSectionRecord] | None,
    config: AppConfig,
) -> tuple[np.ndarray, int]:
    ordered = sorted(units, key=lambda unit: unit.unit_index)
    if spec.independent_units:
        return runtime.encode_independent([unit.text for unit in ordered]), len(ordered)

    if condition is RetrievalCondition.GLOBAL:
        if documents is None or document_id not in documents:
            raise RuntimeError(f"Prepared document {document_id} is unavailable")
        document = documents[document_id]
        targets = [
            _local_target(unit, scope_start=0, scope_text=document.text)
            for unit in ordered
        ]
        return runtime.encode_contextual(document.text, targets), 1

    if condition is RetrievalCondition.SECTION_CONSTRAINED:
        if parent_sections is None:
            raise RuntimeError("Top-level section records are unavailable")
        units_by_parent: dict[str, list[RetrievalUnitRecord]] = defaultdict(list)
        for unit in ordered:
            if unit.parent_section_id is None:
                raise RuntimeError(
                    f"Section-bounded unit {unit.retrieval_unit_id} has no parent section"
                )
            units_by_parent[unit.parent_section_id].append(unit)

        vectors_by_id: dict[str, np.ndarray] = {}
        for parent_id, parent_units in units_by_parent.items():
            parent = parent_sections.get(parent_id)
            if parent is None:
                raise RuntimeError(f"Top-level section {parent_id} is unavailable")
            targets = [
                _local_target(
                    unit,
                    scope_start=parent.span.start,
                    scope_text=parent.text,
                )
                for unit in parent_units
            ]
            vectors = runtime.encode_contextual(parent.text, targets)
            for unit, vector in zip(parent_units, vectors, strict=True):
                vectors_by_id[unit.retrieval_unit_id] = vector

        stacked = np.stack(
            [vectors_by_id[unit.retrieval_unit_id] for unit in ordered],
            axis=0,
        )
        return stacked, len(units_by_parent)

    if condition is RetrievalCondition.LENGTH_MATCHED:
        if documents is None or parent_sections is None:
            raise RuntimeError("Documents and top-level sections are required for LMLC")
        document = documents.get(document_id)
        if document is None:
            raise RuntimeError(f"Prepared document {document_id} is unavailable")
        canonical_tokenizer = load_canonical_tokenizer(config)
        windows = build_length_matched_windows(
            document_text=document.text,
            units=list(ordered),
            parent_sections=parent_sections,
            tokenizer=canonical_tokenizer,
        )

        # Adjacent target units often share the same window. Encode each unique
        # context once, then scatter the pooled target vectors back to unit order.
        grouped: dict[tuple[int, int], list[RetrievalUnitRecord]] = defaultdict(list)
        for unit in ordered:
            window = windows[unit.retrieval_unit_id]
            grouped[(window.span.start, window.span.end)].append(unit)

        vectors_by_id: dict[str, np.ndarray] = {}
        for key, group in grouped.items():
            first = windows[group[0].retrieval_unit_id]
            targets = [windows[unit.retrieval_unit_id].target for unit in group]
            target_spans = [
                TargetSpan(unit.retrieval_unit_id, target.start, target.end)
                for unit, target in zip(group, targets, strict=True)
            ]
            vectors = runtime.encode_contextual(first.text, target_spans)
            for unit, vector in zip(group, vectors, strict=True):
                vectors_by_id[unit.retrieval_unit_id] = vector

        return (
            np.stack([vectors_by_id[unit.retrieval_unit_id] for unit in ordered], axis=0),
            len(grouped),
        )

    raise ValueError(f"Unexpected contextual condition: {condition.value}")


def _infer_dimension_from_document_files(document_dir: Path) -> int | None:
    for path in sorted(document_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"]
            if embeddings.ndim == 2:
                return int(embeddings.shape[1])
    return None


def _validate_cached_passage_file(
    path: Path,
    *,
    expected_unit_ids: Sequence[str],
) -> int:
    try:
        with np.load(path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"]
            unit_ids = payload["retrieval_unit_ids"].tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Cached passage embedding file {path} is invalid. "
            "Re-run with --overwrite."
        ) from exc
    if unit_ids != list(expected_unit_ids):
        raise RuntimeError(
            f"Cached passage embedding file {path} contains different retrieval units. "
            "Re-run with --overwrite."
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_unit_ids):
        raise RuntimeError(
            f"Cached passage embedding file {path} has an invalid shape. "
            "Re-run with --overwrite."
        )
    return int(embeddings.shape[1])


def _validate_cached_query_file(
    path: Path,
    *,
    expected_query_ids: Sequence[str],
) -> int:
    try:
        with np.load(path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"]
            query_ids = payload["query_ids"].tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Cached query embedding file {path} is invalid. Re-run with --overwrite."
        ) from exc
    if query_ids != list(expected_query_ids):
        raise RuntimeError(
            f"Cached query embedding file {path} contains different queries. "
            "Re-run with --overwrite."
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_query_ids):
        raise RuntimeError(
            f"Cached query embedding file {path} has an invalid shape. "
            "Re-run with --overwrite."
        )
    return int(embeddings.shape[1])


def _build_query_encoding(
    config: AppConfig,
    *,
    model_key: EmbeddingModel,
    queries: Sequence[PreparedQueryRecord],
    runtime: EmbeddingRuntime | None,
    runtime_factory: RuntimeFactory,
    overwrite: bool,
) -> tuple[dict[str, Any], EmbeddingRuntime | None]:
    model_config = _model_config(config, model_key)
    output_dir = config.paths.encoding_dir / "queries" / model_key.value
    output_path = output_dir / "queries.npz"
    manifest_path = output_dir / "manifest.json"
    configuration = {
        "model_key": model_key.value,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "query_adapter": model_config.query_adapter,
        "adapter_source": model_config.adapter_source,
        "adapter_revision": model_config.adapter_revision,
        "pooling": config.dense.pooling,
        "normalize": config.dense.normalize,
        "compute_dtype": config.dense.dtype,
        "attn_implementation": config.dense.attn_implementation,
        "output_dtype": config.dense.output_dtype,
        "query_input_fingerprint": _query_input_fingerprint(queries),
        "analysis_sets": (
            ["cross_model_core", "granite_extended"]
            if model_key is EmbeddingModel.GRANITE
            else ["cross_model_core"]
        ),
    }
    fingerprint = _fingerprint(configuration)
    existing = _read_json(manifest_path)

    if output_path.exists() and not overwrite:
        if existing is None or existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached query embeddings at {output_path} do not match the current "
                "configuration. Re-run with --overwrite."
            )
        cached_dimension = _validate_cached_query_file(
            output_path,
            expected_query_ids=[query.query_id for query in queries],
        )
        if int(existing.get("embedding_dimension", -1)) != cached_dimension:
            raise RuntimeError(
                f"Cached query manifest at {manifest_path} does not match its data file. "
                "Re-run with --overwrite."
            )
        return existing, runtime

    if runtime is None:
        runtime = runtime_factory(config, model_key)
    runtime.set_task("query")
    embeddings = runtime.encode_independent([query.question for query in queries])
    if len(embeddings) != len(queries):
        raise RuntimeError("Query encoder returned an unexpected number of vectors")

    dtype = np.float32 if config.dense.output_dtype == "float32" else np.float16
    _write_npz(
        output_path,
        embeddings=embeddings.astype(dtype, copy=False),
        query_ids=np.asarray([query.query_id for query in queries], dtype=np.str_),
        document_ids=np.asarray([query.document_id for query in queries], dtype=np.str_),
    )
    manifest = {
        "schema_version": 2,
        "kind": "dense_queries",
        "model_key": model_key.value,
        "model_id": model_config.model_id,
        "query_count": len(queries),
        "embedding_dimension": int(embeddings.shape[1]) if len(embeddings) else runtime.dimension,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "file": output_path.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, runtime


def build_dense_encoding(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model_key: EmbeddingModel,
    chunk_size: int,
    overwrite: bool = False,
    runtime_factory: RuntimeFactory = load_runtime,
) -> dict[str, Any]:
    """Encode one dense condition with controlled target-span mean pooling."""
    if not condition.is_dense:
        raise ValueError("build_dense_encoding requires a dense retrieval condition")

    spec = condition_spec(condition)
    units = _load_units(config, spec=spec, model_key=model_key, chunk_size=chunk_size)
    queries = _load_queries(config, model_key, chunk_size=chunk_size)
    units_by_document: dict[str, list[RetrievalUnitRecord]] = defaultdict(list)
    for unit in units:
        units_by_document[unit.document_id].append(unit)

    documents = _load_documents(config) if condition in {RetrievalCondition.GLOBAL, RetrievalCondition.LENGTH_MATCHED} else None
    parent_sections = (
        _load_parent_sections(config, chunk_size=chunk_size)
        if condition in {RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.LENGTH_MATCHED}
        else None
    )
    passage_input_fingerprint = _passage_input_fingerprint(
        condition=condition,
        units=units,
        documents=documents,
        parent_sections=parent_sections,
        config=config,
    )
    configuration = _configuration_payload(
        config,
        condition=condition,
        model_key=model_key,
        passage_input_fingerprint=passage_input_fingerprint,
        chunk_size=chunk_size,
    )
    fingerprint = _fingerprint(configuration)
    output_dir = encoding_dir(config, chunk_size) / condition.value / model_key.value
    document_dir = output_dir / "documents"
    manifest_path = output_dir / "manifest.json"
    partial_manifest_path = output_dir / "partial_manifest.json"
    existing_manifest = _read_json(manifest_path)
    existing_partial_manifest = _read_json(partial_manifest_path)
    has_cached_document_files = document_dir.exists() and any(document_dir.glob("*.npz"))

    if (
        existing_manifest is not None
        and existing_manifest.get("configuration_fingerprint") != fingerprint
        and not overwrite
    ):
        raise RuntimeError(
            f"Cached {condition.value}/{model_key.value} embeddings do not match the "
            "current configuration or retrieval inputs. Re-run with --overwrite."
        )

    if has_cached_document_files and existing_manifest is None and not overwrite:
        if existing_partial_manifest is None:
            raise RuntimeError(
                f"Cached passage embeddings exist at {document_dir}, but both the final "
                "and partial manifests are missing. Re-run with --overwrite."
            )
        if existing_partial_manifest.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Partial cached {condition.value}/{model_key.value} embeddings do not "
                "match the current configuration or retrieval inputs. Re-run with "
                "--overwrite."
            )

    if overwrite:
        if document_dir.exists():
            for cached_path in document_dir.glob("*.npz"):
                cached_path.unlink()
        for stale_path in (manifest_path, partial_manifest_path):
            if stale_path.exists():
                stale_path.unlink()
        existing_manifest = None
        existing_partial_manifest = None

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_manifest = {
        "schema_version": 1,
        "kind": "dense_passages_partial",
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model_key.value,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "completed_document_count": 0,
        "last_completed_document_id": None,
    }
    if existing_partial_manifest is not None and not overwrite:
        partial_manifest.update(
            {
                "completed_document_count": int(
                    existing_partial_manifest.get("completed_document_count", 0)
                ),
                "last_completed_document_id": existing_partial_manifest.get(
                    "last_completed_document_id"
                ),
            }
        )
    _write_json_atomic(partial_manifest_path, partial_manifest)

    runtime: EmbeddingRuntime | None = None
    manifest_documents: list[dict[str, Any]] = []
    total_units = 0
    cached_documents = 0
    embedding_dimension = (
        int(existing_manifest["embedding_dimension"])
        if existing_manifest and "embedding_dimension" in existing_manifest
        else _infer_dimension_from_document_files(document_dir)
    )

    for document_id in tqdm(sorted(units_by_document), desc=f"Encoding {condition.value}"):
        document_units = sorted(
            units_by_document[document_id], key=lambda unit: unit.unit_index
        )
        output_path = document_dir / _safe_document_name(document_id)
        if output_path.exists() and not overwrite:
            cached_dimension = _validate_cached_passage_file(
                output_path,
                expected_unit_ids=[unit.retrieval_unit_id for unit in document_units],
            )
            if embedding_dimension is None:
                embedding_dimension = cached_dimension
            elif cached_dimension != embedding_dimension:
                raise RuntimeError(
                    f"Cached passage embedding dimension changed in {output_path}. "
                    "Re-run with --overwrite."
                )
            cached_documents += 1
            manifest_documents.append(
                {
                    "document_id": document_id,
                    "analysis_set": document_units[0].analysis_set,
                    "file": str(output_path.relative_to(output_dir)),
                    "unit_count": len(document_units),
                    "status": "cached",
                }
            )
            total_units += len(document_units)
            partial_manifest["completed_document_count"] = len(manifest_documents)
            partial_manifest["last_completed_document_id"] = document_id
            _write_json_atomic(partial_manifest_path, partial_manifest)
            continue

        if runtime is None:
            runtime = runtime_factory(config, model_key)
        runtime.set_task("passage")
        embeddings, scope_count = _encode_document_units(
            condition=condition,
            spec=spec,
            document_id=document_id,
            units=document_units,
            runtime=runtime,
            documents=documents,
            parent_sections=parent_sections,
            config=config,
        )
        if embeddings.shape[0] != len(document_units):
            raise RuntimeError(
                f"Encoder returned {embeddings.shape[0]} vectors for "
                f"{len(document_units)} units in {document_id}"
            )
        if embedding_dimension is None:
            embedding_dimension = int(embeddings.shape[1])
        elif embeddings.shape[1] != embedding_dimension:
            raise RuntimeError("Embedding dimension changed within one encoding run")

        dtype = np.float32 if config.dense.output_dtype == "float32" else np.float16
        _write_npz(
            output_path,
            embeddings=embeddings.astype(dtype, copy=False),
            retrieval_unit_ids=np.asarray(
                [unit.retrieval_unit_id for unit in document_units], dtype=np.str_
            ),
            unit_indices=np.asarray(
                [unit.unit_index for unit in document_units], dtype=np.int32
            ),
        )
        manifest_documents.append(
            {
                "document_id": document_id,
                "analysis_set": document_units[0].analysis_set,
                "file": str(output_path.relative_to(output_dir)),
                "unit_count": len(document_units),
                "scope_count": scope_count,
                "status": "written",
            }
        )
        total_units += len(document_units)
        partial_manifest["completed_document_count"] = len(manifest_documents)
        partial_manifest["last_completed_document_id"] = document_id
        _write_json_atomic(partial_manifest_path, partial_manifest)

    query_manifest, runtime = _build_query_encoding(
        config,
        model_key=model_key,
        queries=queries,
        runtime=runtime,
        runtime_factory=runtime_factory,
        overwrite=overwrite,
    )
    if embedding_dimension is None:
        embedding_dimension = int(query_manifest["embedding_dimension"])

    manifest = {
        "schema_version": 2,
        "kind": "dense_passages",
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model_key.value,
        "model_id": _model_config(config, model_key).model_id,
        "segmentation_plan": spec.segmentation_plan,
        "context_scope": spec.context_scope,
        "document_count": len(units_by_document),
        "unit_count": total_units,
        "query_count": len(queries),
        "embedding_dimension": embedding_dimension,
        "cached_document_count": cached_documents,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "query_encoding_manifest": str(
            (config.paths.encoding_dir / "queries" / model_key.value / "manifest.json")
            .relative_to(config.paths.encoding_dir)
        ),
        "documents": manifest_documents,
    }
    _write_json_atomic(manifest_path, manifest)
    if partial_manifest_path.exists():
        partial_manifest_path.unlink()
    return manifest


__all__ = [
    "DenseConditionSpec",
    "EmbeddingRuntime",
    "TargetSpan",
    "TransformerEmbeddingRuntime",
    "build_dense_encoding",
    "condition_spec",
    "load_runtime",
]
