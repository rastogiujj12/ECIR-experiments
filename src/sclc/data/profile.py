from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sclc.config import AppConfig, ModelProfileConfig
from sclc.data.io import read_documents_jsonl

console = Console()


def load_tokenizer(model: ModelProfileConfig, cache_dir: Path) -> PreTrainedTokenizerBase:
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": model.tokenizer_trust_remote_code,
        "cache_dir": str(cache_dir),
        "use_fast": True,
    }
    if model.revision is not None:
        tokenizer_kwargs["revision"] = model.revision
    tokenizer = AutoTokenizer.from_pretrained(model.model_id, **tokenizer_kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{model.model_id} did not load a fast tokenizer. "
            "Fast offset mappings are required later for target-span pooling."
        )
    return tokenizer


def token_count(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    encoded: dict[str, Any] = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def _eligibility(
    granite_tokens: int,
    jina_tokens: int,
    granite_limit: int,
    jina_limit: int,
) -> str:
    if granite_tokens > granite_limit:
        return "excluded_too_long"
    if jina_tokens > jina_limit:
        return "granite_extended"
    return "cross_model_core"


def profile_documents(config: AppConfig) -> pd.DataFrame:
    documents_path = config.paths.processed_dir / "documents.jsonl"
    documents = list(read_documents_jsonl(documents_path))

    console.print("[bold]Loading Granite tokenizer...[/bold]")
    granite_tokenizer = load_tokenizer(config.models.granite, config.paths.hf_cache_dir)

    console.print("[bold]Loading Jina tokenizer...[/bold]")
    jina_tokenizer = load_tokenizer(config.models.jina, config.paths.hf_cache_dir)

    rows: list[dict[str, Any]] = []
    for document in tqdm(documents, desc="Profiling complete papers"):
        granite_tokens = token_count(granite_tokenizer, document.text)
        jina_tokens = token_count(jina_tokenizer, document.text)
        rows.append(
            {
                "document_id": document.document_id,
                "split": document.split,
                "title": document.title,
                "character_count": len(document.text),
                "section_count": len(document.sections),
                "paragraph_count": len(document.paragraphs),
                "question_count": len(document.queries),
                "usable_question_count": document.usable_question_count,
                "granite_tokens": granite_tokens,
                "jina_tokens": jina_tokens,
                "eligibility_group": _eligibility(
                    granite_tokens=granite_tokens,
                    jina_tokens=jina_tokens,
                    granite_limit=config.models.granite.max_document_tokens,
                    jina_limit=config.models.jina.max_document_tokens,
                ),
            }
        )

    del granite_tokenizer
    del jina_tokenizer
    gc.collect()
    return pd.DataFrame(rows)


def write_profile_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "document_lengths.csv", index=False)
    frame.to_parquet(output_dir / "document_lengths.parquet", index=False)

    grouped = (
        frame.groupby("eligibility_group", dropna=False)
        .agg(
            documents=("document_id", "count"),
            usable_questions=("usable_question_count", "sum"),
            median_granite_tokens=("granite_tokens", "median"),
            max_granite_tokens=("granite_tokens", "max"),
            median_jina_tokens=("jina_tokens", "median"),
            max_jina_tokens=("jina_tokens", "max"),
        )
        .reset_index()
    )

    summary = {
        "total_documents": int(len(frame)),
        "total_usable_questions": int(frame["usable_question_count"].sum()),
        "groups": grouped.to_dict(orient="records"),
    }
    with (output_dir / "profile_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
