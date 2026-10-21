#!/usr/bin/env python3
"""Exploratory mechanism analysis for the QASPER ECIR experiment.

This script performs A-D using *existing* experiment outputs only. It does not
run encoding, retrieval, or evaluation.

A  Section-length dependence of SCLC - LMLC.
B  Proximity of supporting evidence to a top-level section boundary.
C  Dependence on the project's manually coded QASPER query type.
D  Dependence on the project's evidence-structure classification.

Important methodological choices:
- The confirmatory population is the common ``cross_model_core`` test split.
- SCLC/LMLC/GLC query-level metrics are required to have identical query sets;
  the script never silently drops unmatched queries during merging.
- The exact top-level section records saved by the 128-token experiment are
  reused.  The 256/512 section records are checked for structural identity.
- LMLC parent-section token lengths are recovered with the *same* project
  ``build_length_matched_windows`` implementation used by the experiment.
- Subgroup confidence intervals use the project's own paired document
  bootstrap, so the exploratory effects use the same statistical machinery as
  the confirmatory comparisons.
- Partial query-type coding is allowed for this exploratory analysis even when
  the main experiment configuration requires complete coding. Only coded
  queries are included in C.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sclc.config import load_config
from sclc.data.context_windows import build_length_matched_windows, load_canonical_tokenizer
from sclc.data.io import read_documents_jsonl
from sclc.data.retrieval_unit_io import (
    read_prepared_queries,
    read_retrieval_units,
    read_top_level_sections,
)
from sclc.data.query_types import ALLOWED_QUERY_TYPES
from sclc.evaluation.statistics import paired_document_bootstrap
from sclc.paths import evaluation_dir, global_query_types_path, retrieval_unit_dir

CONFIG_PATH = Path("configs/qasper_full.yaml")
OUT_DIR = Path("outputs/qasper/mechanism_analysis")
SIZES = (128, 256, 512)
MODELS = ("granite", "jina")
PRIMARY_SET = "cross_model_core"
PRIMARY_SPLIT = "test"
EFFECTS = {
    "sclc_minus_lmlc": ("ndcg_sclc", "ndcg_lmlc"),
    "sclc_minus_glc": ("ndcg_sclc", "ndcg_glc"),
}


def spearman_rho(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rho without SciPy."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < 3:
        return math.nan
    x_rank = pd.Series(x_arr).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y_arr).rank(method="average").to_numpy(dtype=float)
    if x_rank.std(ddof=1) == 0.0 or y_rank.std(ddof=1) == 0.0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def rank_quartiles(series: pd.Series) -> pd.Series:
    labels = ["shortest", "short_short_mid", "long_mid", "longest"]
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = values.notna()
    if not valid.any():
        return result
    ranks = values.loc[valid].rank(method="average", pct=True)
    bucket = np.ceil(ranks.to_numpy(dtype=float) * len(labels)).astype(int)
    bucket = np.clip(bucket, 1, len(labels))
    result.loc[valid] = [labels[i - 1] for i in bucket]
    return result


def _summary(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    *,
    group_name: str,
    group: pd.DataFrame,
    seed: int,
    config,
) -> dict[str, object]:
    if group.empty:
        return {
            "group": group_name,
            "query_count": 0,
            "document_count": 0,
            "mean_effect": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
        }
    observed, lower, upper, p_value, _ = paired_document_bootstrap(
        group,
        first_column=first_column,
        second_column=second_column,
        iterations=config.evaluation.bootstrap_iterations,
        confidence_level=config.evaluation.confidence_level,
        seed=seed,
    )
    return {
        "group": group_name,
        "query_count": int(group["query_id"].nunique()),
        "document_count": int(group["document_id"].nunique()),
        "mean_effect": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "exploratory_bootstrap_p": p_value,
    }


def load_effects(config) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    for size in SIZES:
        for model in MODELS:
            base = evaluation_dir(config, size)
            frames: dict[str, pd.DataFrame] = {}
            for condition, filename in {
                "sclc": "section_constrained",
                "lmlc": "length_matched",
                "glc": "global",
            }.items():
                path = base / filename / model / "query_metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(
                    path,
                    usecols=["query_id", "document_id", "split", "analysis_set", "ndcg_at_5"],
                    dtype={
                        "query_id": "string",
                        "document_id": "string",
                        "split": "string",
                        "analysis_set": "string",
                    },
                    keep_default_na=False,
                )
                for column in ("query_id", "document_id", "split", "analysis_set"):
                    frame[column] = frame[column].astype("string").str.strip()
                frame = frame[
                    (frame["analysis_set"] == PRIMARY_SET)
                    & (frame["split"] == PRIMARY_SPLIT)
                ].copy()
                if frame.empty:
                    raise RuntimeError(f"No {PRIMARY_SET}/{PRIMARY_SPLIT} rows in {path}")
                if frame["query_id"].duplicated().any():
                    raise RuntimeError(f"Duplicate query IDs in {path}")
                if frame[["query_id", "document_id"]].duplicated().any():
                    raise RuntimeError(f"Duplicate query/document pairs in {path}")
                frames[condition] = frame[["query_id", "document_id", "ndcg_at_5"]].rename(
                    columns={"ndcg_at_5": f"ndcg_{condition}"}
                )

            ids = {
                name: set(frame["query_id"].astype(str)) for name, frame in frames.items()
            }
            if not (ids["sclc"] == ids["lmlc"] == ids["glc"]):
                raise RuntimeError(
                    f"Mismatched query sets at size={size}, model={model}: "
                    f"SCLC={len(ids['sclc'])}, LMLC={len(ids['lmlc'])}, GLC={len(ids['glc'])}"
                )

            merged = frames["sclc"]
            merged = merged.merge(frames["lmlc"], on=["query_id", "document_id"], how="left", validate="one_to_one")
            merged = merged.merge(frames["glc"], on=["query_id", "document_id"], how="left", validate="one_to_one")
            if merged[["ndcg_sclc", "ndcg_lmlc", "ndcg_glc"]].isna().any().any():
                raise RuntimeError(f"Metric values are missing after merge at size={size}, model={model}")

            merged["size"] = size
            merged["model"] = model
            merged["delta_sclc_lmlc"] = merged["ndcg_sclc"] - merged["ndcg_lmlc"]
            merged["delta_sclc_glc"] = merged["ndcg_sclc"] - merged["ndcg_glc"]
            all_frames.append(merged)

    return pd.concat(all_frames, ignore_index=True)


def _load_partial_query_types(config, query_ids: set[str]) -> dict[str, str]:
    """Load only the existing coded subset; do not require complete coding."""
    path = global_query_types_path(config)
    if not path.exists():
        return {}
    frame = pd.read_csv(
        path,
        dtype={"query_id": "string", "query_type": "string"},
        keep_default_na=False,
    )
    required = {"query_id", "query_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["query_id"].duplicated().any():
        duplicates = frame.loc[frame["query_id"].duplicated(), "query_id"].tolist()
        raise ValueError(f"Duplicate query IDs in {path}: {duplicates[:10]}")
    result: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        query_id = str(row.query_id).strip()
        query_type = str(row.query_type).strip().lower().replace("-", "_").replace(" ", "_")
        if not query_id or query_id not in query_ids:
            continue
        if query_type not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"Unsupported query type {query_type!r} for {query_id}")
        result[query_id] = query_type
    return result


def _load_parent_sections(config, document_ids: set[str]):
    paths = [retrieval_unit_dir(config, size) / "top_level_sections.jsonl" for size in SIZES]
    if any(not path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Missing top-level-section artifacts: {missing}")

    records_by_size = []
    for path in paths:
        records = {
            record.parent_section_id: record
            for record in read_top_level_sections(path)
            if record.document_id in document_ids
        }
        records_by_size.append(records)

    base = records_by_size[0]
    for size, records in zip(SIZES[1:], records_by_size[1:], strict=True):
        if set(records) != set(base):
            raise RuntimeError(f"Top-level section IDs differ between 128 and {size}")
        for section_id, expected in base.items():
            actual = records[section_id]
            fingerprint_expected = (
                expected.document_id,
                expected.heading,
                expected.span.start,
                expected.span.end,
                tuple(expected.source_section_ids),
            )
            fingerprint_actual = (
                actual.document_id,
                actual.heading,
                actual.span.start,
                actual.span.end,
                tuple(actual.source_section_ids),
            )
            if fingerprint_expected != fingerprint_actual:
                raise RuntimeError(
                    f"Top-level section structure differs for {section_id} between 128 and {size}"
                )
    return base


def _prepare_section_metadata(documents, parent_sections):
    """Build the exact mappings required by the project's evidence classifier.

    Evidence annotations are expressed as paragraph IDs, while each paragraph
    points to a source section ID. The evidence classifier therefore needs a
    paragraph_id -> top_level_parent_id mapping. We keep a separate
    section_id -> top_level_parent_id mapping for boundary/parent lookup.
    """
    paragraph_to_parent: dict[str, dict[str, str]] = defaultdict(dict)
    section_to_parent: dict[str, dict[str, str]] = defaultdict(dict)
    parents_by_document: dict[str, dict[str, object]] = defaultdict(dict)

    for parent_id, parent in parent_sections.items():
        parents_by_document[parent.document_id][parent_id] = parent
        for source_section_id in parent.source_section_ids:
            section_to_parent[parent.document_id][source_section_id] = parent_id

    if documents is None:
        raise ValueError("Documents are required to map evidence paragraphs to parent sections")

    for document in documents:
        section_map = section_to_parent.get(document.document_id, {})
        paragraph_map = paragraph_to_parent[document.document_id]
        for paragraph in document.paragraphs:
            parent_id = section_map.get(paragraph.section_id)
            if parent_id is not None:
                paragraph_map[paragraph.paragraph_id] = parent_id

    return paragraph_to_parent, section_to_parent, parents_by_document


def _evidence_metadata(query, document, paragraph_to_parent, section_to_parent, parents_by_id):
    # Importing here keeps the dependency localized and uses the project's exact classifier.
    from sclc.analysis.evidence_structure import classify_evidence_structure

    classification = classify_evidence_structure(query, paragraph_to_parent)
    primary_id = classification["primary_evidence_set_id"]
    primary_set = next(es for es in query.evidence_sets if es.evidence_set_id == primary_id)
    paragraph_lookup = {paragraph.paragraph_id: paragraph for paragraph in document.paragraphs}
    primary_paragraphs = [
        paragraph_lookup[pid] for pid in primary_set.paragraph_ids if pid in paragraph_lookup
    ]
    primary_parent_ids = sorted({section_to_parent[p.section_id] for p in primary_paragraphs})

    section_length_tokens = math.nan
    min_boundary_distance_chars = math.nan
    min_boundary_relative_distance = math.nan
    if len(primary_parent_ids) == 1 and primary_paragraphs:
        parent = parents_by_id[primary_parent_ids[0]]
        distances = []
        relative = []
        parent_length_chars = max(1, parent.span.end - parent.span.start)
        for paragraph in primary_paragraphs:
            left = max(0, paragraph.span.start - parent.span.start)
            right = max(0, parent.span.end - paragraph.span.end)
            distance = min(left, right)
            distances.append(distance)
            relative.append(distance / parent_length_chars)
        min_boundary_distance_chars = float(min(distances))
        min_boundary_relative_distance = float(min(relative))

    return {
        "query_type": None,
        "evidence_structure": classification["evidence_structure"],
        "primary_evidence_set_id": primary_id,
        "primary_section_count": len(primary_parent_ids),
        "primary_section_ids": "|".join(primary_parent_ids),
        "evidence_paragraph_count": len(primary_paragraphs),
        "section_length_tokens": section_length_tokens,
        "boundary_distance_chars": min_boundary_distance_chars,
        "boundary_relative_distance": min_boundary_relative_distance,
        "cross_section_required": bool(classification["cross_section_required"]),
        "cross_section_possible": bool(classification["cross_section_possible"]),
        "minimal_structure_ambiguous": bool(classification["minimal_structure_ambiguous"]),
    }


def _exact_parent_token_lengths(config, document_ids: set[str], parent_sections):
    units_path = retrieval_unit_dir(config, 128) / "section_bounded_units.jsonl"
    if not units_path.exists():
        raise FileNotFoundError(units_path)

    one_unit_by_parent: dict[tuple[str, str], object] = {}
    for unit in read_retrieval_units(units_path):
        if unit.document_id not in document_ids or unit.parent_section_id is None:
            continue
        one_unit_by_parent.setdefault((unit.document_id, unit.parent_section_id), unit)

    parent_keys = {
        (parent.document_id, parent.parent_section_id)
        for parent in parent_sections.values()
    }
    unit_keys = set(one_unit_by_parent)
    missing = sorted(parent_keys.difference(unit_keys))
    if missing:
        examples = missing[:5]
        raise RuntimeError(
            "Saved section-bounded units do not cover all saved top-level sections: "
            f"{len(missing)} parent sections are missing; examples={examples}"
        )

    units_by_doc: dict[str, list[object]] = defaultdict(list)
    for (doc_id, _), unit in one_unit_by_parent.items():
        units_by_doc[doc_id].append(unit)

    tokenizer = load_canonical_tokenizer(config)
    documents_path = config.paths.processed_dir / "documents.jsonl"
    lengths: dict[tuple[str, str], int] = {}
    for document in read_documents_jsonl(documents_path):
        if document.document_id not in document_ids:
            continue
        units = units_by_doc.get(document.document_id, [])
        parents = {
            parent_id: parent
            for parent_id, parent in parent_sections.items()
            if parent.document_id == document.document_id
        }
        windows = build_length_matched_windows(
            document_text=document.text,
            units=units,
            parent_sections=parents,
            tokenizer=tokenizer,
        )
        for unit in units:
            lengths[(document.document_id, unit.parent_section_id)] = int(
                windows[unit.retrieval_unit_id].canonical_token_count
            )
    if len(lengths) != len(parent_sections):
        raise RuntimeError(
            f"Recovered {len(lengths)} parent token lengths for {len(parent_sections)} parent sections"
        )
    return lengths


def build_metadata(config, query_ids: set[str], effects: pd.DataFrame) -> pd.DataFrame:
    prepared_path = retrieval_unit_dir(config, 128) / "queries.jsonl"
    documents_path = config.paths.processed_dir / "documents.jsonl"
    if not prepared_path.exists():
        raise FileNotFoundError(prepared_path)
    if not documents_path.exists():
        raise FileNotFoundError(documents_path)

    prepared = {
        query.query_id: query
        for query in read_prepared_queries(prepared_path)
        if query.query_id in query_ids
    }
    missing = sorted(query_ids.difference(prepared))
    if missing:
        raise RuntimeError(f"{len(missing)} evaluation queries are missing from {prepared_path}; examples={missing[:5]}")

    document_ids = {query.document_id for query in prepared.values()}
    parent_sections = _load_parent_sections(config, document_ids)
    relevant_documents = []
    for document in read_documents_jsonl(documents_path):
        if document.document_id in document_ids:
            relevant_documents.append(document)
    if {document.document_id for document in relevant_documents} != document_ids:
        missing_documents = sorted(document_ids.difference(document.document_id for document in relevant_documents))
        raise RuntimeError(
            f"Could not load {len(missing_documents)} required documents; examples={missing_documents[:5]}"
        )
    paragraph_to_parent, section_to_parent, parents_by_document = _prepare_section_metadata(
        relevant_documents, parent_sections
    )
    parent_lengths = _exact_parent_token_lengths(config, document_ids, parent_sections)
    query_types = _load_partial_query_types(config, query_ids)

    query_ids_by_document: dict[str, list[str]] = defaultdict(list)
    for query in prepared.values():
        query_ids_by_document[query.document_id].append(query.query_id)

    rows: list[dict[str, object]] = []
    found: set[str] = set()
    for document in read_documents_jsonl(documents_path):
        relevant = query_ids_by_document.get(document.document_id)
        if not relevant:
            continue
        paragraph_map = paragraph_to_parent.get(document.document_id, {})
        section_map = section_to_parent.get(document.document_id, {})
        if not paragraph_map or not section_map:
            raise RuntimeError(f"No paragraph/section parent mapping for {document.document_id}")
        parents = parents_by_document[document.document_id]
        for query_id in relevant:
            section_map_for_document = section_to_parent.get(document.document_id, {})
            meta = _evidence_metadata(
                prepared[query_id],
                document,
                paragraph_map,
                section_map_for_document,
                parents,
            )
            if meta["primary_section_count"] == 1:
                parent_id = meta["primary_section_ids"]
                meta["section_length_tokens"] = parent_lengths[(document.document_id, parent_id)]
            meta["query_id"] = query_id
            meta["document_id"] = document.document_id
            meta["query_type"] = query_types.get(query_id, "uncoded")
            rows.append(meta)
            found.add(query_id)

    missing_after = sorted(query_ids.difference(found))
    if missing_after:
        raise RuntimeError(f"Could not build metadata for {len(missing_after)} queries; examples={missing_after[:5]}")
    meta = pd.DataFrame(rows)
    for key in ("query_id", "document_id", "query_type"):
        meta[key] = meta[key].astype("string").str.strip()
    if meta[["query_id", "document_id"]].duplicated().any():
        raise RuntimeError("Metadata has duplicate query/document pairs")
    return meta


def grouped_effects(frame: pd.DataFrame, group_column: str, first: str, second: str, config, seed_base: int):
    rows = []
    for idx, (value, group) in enumerate(frame.groupby(group_column, dropna=False)):
        name = "<NA>" if pd.isna(value) else str(value)
        rows.append(_summary(frame, first, second, group_name=name, group=group, seed=seed_base + idx, config=config))
    return pd.DataFrame(rows)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = load_config(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = load_effects(config)
    query_ids = set(effects["query_id"].astype(str))
    metadata = build_metadata(config, query_ids, effects)

    effects["query_id"] = effects["query_id"].astype("string").str.strip()
    effects["document_id"] = effects["document_id"].astype("string").str.strip()
    metadata["query_id"] = metadata["query_id"].astype("string").str.strip()
    metadata["document_id"] = metadata["document_id"].astype("string").str.strip()

    merged = effects.merge(
        metadata,
        on=["query_id", "document_id"],
        how="left",
        validate="many_to_one",
    )
    required_metadata = [
        "evidence_structure",
        "primary_section_count",
        "section_length_tokens",
        "boundary_distance_chars",
        "boundary_relative_distance",
    ]
    missing_rows = merged[required_metadata].isna().all(axis=1)
    if missing_rows.any():
        examples = merged.loc[missing_rows, ["query_id", "document_id"]].head(10).to_dict("records")
        raise RuntimeError(f"Missing structural metadata for {int(missing_rows.sum())} rows; examples={examples}")
    merged.to_csv(OUT_DIR / "query_effects.csv", index=False)

    # A: section length.
    a = merged.dropna(subset=["section_length_tokens"]).copy()
    a["section_length_quartile"] = rank_quartiles(a["section_length_tokens"])
    rows = []
    for (size, model), group in a.groupby(["size", "model"]):
        for effect_name, (first, second) in EFFECTS.items():
            summary = grouped_effects(group, "section_length_quartile", first, second, config, 10_000 + int(size))
            summary.insert(0, "size", size)
            summary.insert(1, "model", model)
            summary.insert(2, "effect", effect_name)
            rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "A_section_length.csv", index=False)

    # B: minimum boundary proximity of any supporting evidence paragraph, within one primary section.
    b = merged.dropna(subset=["boundary_relative_distance"]).copy()
    b["boundary_proximity_quartile"] = rank_quartiles(b["boundary_relative_distance"])
    rows = []
    for (size, model), group in b.groupby(["size", "model"]):
        for effect_name, (first, second) in EFFECTS.items():
            summary = grouped_effects(group, "boundary_proximity_quartile", first, second, config, 20_000 + int(size))
            summary.insert(0, "size", size)
            summary.insert(1, "model", model)
            summary.insert(2, "effect", effect_name)
            rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "B_boundary_proximity.csv", index=False)

    # Correlations are exploratory and are not used as confirmatory significance tests.
    corr_rows = []
    for (size, model), group in a.groupby(["size", "model"]):
        for variable in ("section_length_tokens", "boundary_relative_distance"):
            subset = group[[variable, "delta_sclc_lmlc"]].dropna()
            corr_rows.append({
                "size": size,
                "model": model,
                "variable": variable,
                "n": len(subset),
                "spearman_rho": spearman_rho(subset[variable], subset["delta_sclc_lmlc"]),
            })
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "A_B_spearman.csv", index=False)

    # C: partial manually coded query-type subset.
    coded = merged[merged["query_type"].isin(ALLOWED_QUERY_TYPES)].copy()
    rows = []
    for (size, model), group in coded.groupby(["size", "model"]):
        for effect_name, (first, second) in EFFECTS.items():
            summary = grouped_effects(group, "query_type", first, second, config, 30_000 + int(size))
            summary.insert(0, "size", size)
            summary.insert(1, "model", model)
            summary.insert(2, "effect", effect_name)
            rows.append(summary)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "C_query_type.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "C_query_type.csv", index=False)

    # D: exact project evidence-structure classifier.
    rows = []
    for (size, model), group in merged.groupby(["size", "model"]):
        for effect_name, (first, second) in EFFECTS.items():
            summary = grouped_effects(group, "evidence_structure", first, second, config, 40_000 + int(size))
            summary.insert(0, "size", size)
            summary.insert(1, "model", model)
            summary.insert(2, "effect", effect_name)
            rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "D_evidence_structure.csv", index=False)

    overview = []
    for (size, model), group in merged.groupby(["size", "model"]):
        overview.append({
            "size": size,
            "model": model,
            "queries": len(group),
            "documents": group["document_id"].nunique(),
            "single_section_primary_evidence_queries": int(group["section_length_tokens"].notna().sum()),
            "coded_query_type_queries": int(group["query_type"].isin(ALLOWED_QUERY_TYPES).sum()),
            "cross_section_primary_evidence_queries": int((group["primary_section_count"] > 1).sum()),
            "mean_sclc_minus_lmlc_query_weighted": float(group["delta_sclc_lmlc"].mean()),
            "mean_sclc_minus_glc_query_weighted": float(group["delta_sclc_glc"].mean()),
        })
    pd.DataFrame(overview).to_csv(OUT_DIR / "overview.csv", index=False)

    print("A-D mechanism analysis complete.")
    print(f"Outputs: {OUT_DIR}")
    print(f"Primary population: {PRIMARY_SET}/{PRIMARY_SPLIT}")
    print("No encoding/retrieval/evaluation was run.")


if __name__ == "__main__":
    main()
