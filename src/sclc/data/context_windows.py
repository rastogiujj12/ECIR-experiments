from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sclc.data.schema import CharacterSpan, RetrievalUnitRecord, TopLevelSectionRecord


class OffsetTokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def load_canonical_tokenizer(config: Any) -> OffsetTokenizer:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": config.chunking.tokenizer_trust_remote_code,
        "cache_dir": str(config.paths.hf_cache_dir),
        "use_fast": True,
    }
    if config.chunking.canonical_tokenizer_revision is not None:
        kwargs["revision"] = config.chunking.canonical_tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(config.chunking.canonical_tokenizer, **kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError("The canonical tokenizer must provide fast character offsets")
    return tokenizer


@dataclass(frozen=True)
class ContextWindow:
    unit_id: str
    span: CharacterSpan
    text: str
    target: CharacterSpan
    canonical_token_count: int
    relative_position_preserved: bool


def _tokenise(tokenizer: OffsetTokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return list(encoded["input_ids"]), [tuple(value) for value in encoded["offset_mapping"]]


def _valid_offsets(offsets: list[tuple[int, int]]) -> list[int]:
    return [index for index, (start, end) in enumerate(offsets) if end > start]


def build_length_matched_windows(
    *,
    document_text: str,
    units: list[RetrievalUnitRecord],
    parent_sections: dict[str, TopLevelSectionRecord],
    tokenizer: OffsetTokenizer,
) -> dict[str, ContextWindow]:
    """Build non-structural context windows matched to each unit's parent-section length.

    The window is a contiguous slice of the complete document whose canonical
    token length equals the target unit's parent-section length. The target is
    centred in that window where possible, so the control does not simply recover
    the structural parent section. The window may cross top-level section boundaries.
    """
    _, offsets = _tokenise(tokenizer, document_text)
    valid = _valid_offsets(offsets)
    if len(valid) != len(offsets):
        raise RuntimeError("Canonical tokenizer emitted zero-length offsets")
    if not valid:
        return {}

    char_to_token = offsets
    results: dict[str, ContextWindow] = {}

    def token_interval(span: CharacterSpan) -> tuple[int, int]:
        overlapping = [
            index
            for index, (start, end) in enumerate(char_to_token)
            if start < span.end and end > span.start
        ]
        if not overlapping:
            raise RuntimeError(f"No canonical tokens overlap span {span}")
        return min(overlapping), max(overlapping) + 1

    for unit in units:
        if unit.parent_section_id is None:
            raise RuntimeError(f"LMLC requires a parent section for {unit.retrieval_unit_id}")
        parent = parent_sections.get(unit.parent_section_id)
        if parent is None:
            raise RuntimeError(f"Unknown parent section {unit.parent_section_id}")

        parent_start, parent_end = token_interval(parent.span)
        target_start, target_end = token_interval(unit.span)
        context_length = parent_end - parent_start
        if context_length <= 0:
            raise RuntimeError(f"Empty parent section for {parent.parent_section_id}")
        if target_end - target_start > context_length:
            raise RuntimeError(
                f"Target {unit.retrieval_unit_id} is longer than its parent section"
            )

        target_length = target_end - target_start
        # LMLC matches the *amount* of contextual input to the parent section,
        # but deliberately does not use the section boundary. The target is
        # centred in the linear window where possible so that the control gets
        # meaningful left/right context while remaining independent of structure.
        desired_start = target_start - max(0, (context_length - target_length) // 2)
        max_start = len(offsets) - context_length
        context_start_token = min(max(desired_start, 0), max_start)
        context_end_token = context_start_token + context_length

        if not (
            context_start_token <= target_start
            and target_end <= context_end_token
        ):
            raise RuntimeError(
                f"Could not place target {unit.retrieval_unit_id} inside its "
                "length-matched context"
            )

        context_start_char = offsets[context_start_token][0]
        context_end_char = offsets[context_end_token - 1][1]
        context_span = CharacterSpan(start=context_start_char, end=context_end_char)
        target = CharacterSpan(
            start=unit.span.start - context_start_char,
            end=unit.span.end - context_start_char,
        )
        context_text = document_text[context_span.start : context_span.end]

        relative_position_preserved = context_start_token == desired_start
        results[unit.retrieval_unit_id] = ContextWindow(
            unit_id=unit.retrieval_unit_id,
            span=context_span,
            text=context_text,
            target=target,
            canonical_token_count=context_length,
            relative_position_preserved=relative_position_preserved,
        )

    return results
