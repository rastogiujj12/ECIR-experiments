from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("outputs/qasa/encodings")
DATA = Path("data/qasa/retrieval_units")
OUT = Path("outputs/qasa/analysis/cosine_scope")
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "section_isolated",
    "section_constrained",
    "global",
    "length_matched",
]
MODELS = ["jina", "granite"]
SIZES = [128, 256, 512]

# Cosine values pooled across all chunk pairs.
pair_values = defaultdict(list)

# Each document contributes one mean per relation.
document_rows = []

# Each section with >= 2 chunks contributes one mean.
section_rows = []

# Each chunk with >= 1 sibling contributes two similarity measures.
unit_rows = []

missing_metadata = 0
processed_archives = 0


def describe(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
        }
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
    }


for size in SIZES:
    unit_path = DATA / f"chunk_{size}" / "retrieval_units.csv"
    units = pd.read_csv(unit_path)

    # Embedding archives use section-bounded unit IDs.
    units = units.drop_duplicates("retrieval_unit_id")
    metadata = {}

    for row in units.to_dict("records"):
        rid = str(row["retrieval_unit_id"])
        section = row.get("parent_section_id")
        if pd.isna(section):
            section = None

        metadata[rid] = {
            "document_id": str(row["document_id"]),
            "parent_section_id": (
                str(section) if section is not None else None
            ),
        }

    for model in MODELS:
        for condition in CONDITIONS:
            enc_dir = (
                BASE / f"chunk_{size}" / condition / model / "documents"
            )

            archives = sorted(enc_dir.glob("document_*.npz"))
            if not archives:
                print(f"SKIP: no archives in {enc_dir}")
                continue

            for archive in archives:
                with np.load(archive, allow_pickle=False) as data:
                    vectors = data["embeddings"].astype(np.float64)
                    unit_ids = data["retrieval_unit_ids"].astype(str)

                # Compute true cosine similarities, irrespective of whether
                # stored vectors were already normalized.
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
                similarity = vectors @ vectors.T

                info = [metadata.get(uid) for uid in unit_ids]
                missing_metadata += sum(x is None for x in info)

                keep = [
                    i for i, item in enumerate(info)
                    if item is not None
                    and item["parent_section_id"] is not None
                ]
                if len(keep) < 2:
                    continue

                vectors = vectors[keep]
                similarity = similarity[np.ix_(keep, keep)]
                unit_ids = unit_ids[keep]
                info = [info[i] for i in keep]

                document_ids = {x["document_id"] for x in info}
                if len(document_ids) != 1:
                    raise ValueError(
                        f"Mixed documents in archive: {archive}"
                    )
                document_id = next(iter(document_ids))

                sections = [x["parent_section_id"] for x in info]
                key_base = (condition, model, size)

                # Compare every distinct pair of chunks within this paper.
                pair_indices = np.triu_indices(len(unit_ids), k=1)
                doc_pairs = defaultdict(list)

                for i, j in zip(*pair_indices):
                    relation = (
                        "within_section"
                        if sections[i] == sections[j]
                        else "across_sections"
                    )
                    value = float(similarity[i, j])
                    key = (*key_base, relation)
                    pair_values[key].append(value)
                    doc_pairs[relation].append(value)

                for relation, values in doc_pairs.items():
                    document_rows.append({
                        "condition": condition,
                        "model": model,
                        "chunk_size": size,
                        "document_id": document_id,
                        "relation": relation,
                        "mean_cosine": float(np.mean(values)),
                        "pair_count": len(values),
                    })

                # Calculate within-section pair similarity, giving every
                # section equal weight in the section-level summary.
                section_to_indices = defaultdict(list)
                for i, section in enumerate(sections):
                    section_to_indices[section].append(i)

                for section, indices in section_to_indices.items():
                    if len(indices) < 2:
                        continue

                    matrix = similarity[np.ix_(indices, indices)]
                    ii, jj = np.triu_indices(len(indices), k=1)
                    values = matrix[ii, jj].tolist()

                    section_rows.append({
                        "condition": condition,
                        "model": model,
                        "chunk_size": size,
                        "document_id": document_id,
                        "parent_section_id": section,
                        "chunk_count": len(indices),
                        "mean_pair_cosine": float(np.mean(values)),
                        "median_pair_cosine": float(np.median(values)),
                        "pair_count": len(values),
                    })

                    # For each chunk, measure its average and maximum
                    # similarity to another chunk in the same section.
                    for pos, global_idx in enumerate(indices):
                        neighbours = np.delete(matrix[pos], pos)
                        unit_rows.append({
                            "condition": condition,
                            "model": model,
                            "chunk_size": size,
                            "document_id": document_id,
                            "retrieval_unit_id": unit_ids[global_idx],
                            "parent_section_id": section,
                            "sibling_count": len(neighbours),
                            "mean_sibling_cosine": float(
                                np.mean(neighbours)
                            ),
                            "nearest_sibling_cosine": float(
                                np.max(neighbours)
                            ),
                        })

                processed_archives += 1


# Pair-weighted summaries.
pair_summary_rows = []
for key, values in sorted(pair_values.items()):
    condition, model, size, relation = key
    stats = describe(values)
    pair_summary_rows.append({
        "condition": condition,
        "model": model,
        "chunk_size": size,
        "relation": relation,
        **stats,
    })

pair_summary = pd.DataFrame(pair_summary_rows)
pair_summary.to_csv(OUT / "cosine_pair_summary.csv", index=False)

documents = pd.DataFrame(document_rows)
documents.to_csv(OUT / "cosine_document_summary.csv", index=False)

sections = pd.DataFrame(section_rows)
sections.to_csv(OUT / "cosine_section_summary.csv", index=False)

units_out = pd.DataFrame(unit_rows)
units_out.to_csv(OUT / "cosine_unit_summary.csv", index=False)

# Equal-document-weighted results: large papers don't dominate.
if not documents.empty:
    doc_summary = (
        documents.groupby(
            ["condition", "model", "chunk_size", "relation"]
        )
        .agg(
            documents=("document_id", "nunique"),
            mean_document_cosine=("mean_cosine", "mean"),
            median_document_cosine=("mean_cosine", "median"),
            document_sd=("mean_cosine", "std"),
        )
        .reset_index()
    )
    doc_summary.to_csv(
        OUT / "cosine_equal_document_summary.csv", index=False
    )

# Equal-section-weighted results.
if not sections.empty:
    section_summary = (
        sections.groupby(["condition", "model", "chunk_size"])
        .agg(
            sections=("parent_section_id", "count"),
            mean_section_cosine=("mean_pair_cosine", "mean"),
            median_section_cosine=("mean_pair_cosine", "median"),
            p10=("mean_pair_cosine", lambda x: x.quantile(0.10)),
            p90=("mean_pair_cosine", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    section_summary.to_csv(
        OUT / "cosine_equal_section_summary.csv", index=False
    )

# Chunk-level sibling similarity summary.
if not units_out.empty:
    unit_summary = (
        units_out.groupby(["condition", "model", "chunk_size"])
        .agg(
            chunks=("retrieval_unit_id", "count"),
            mean_sibling_cosine=("mean_sibling_cosine", "mean"),
            median_sibling_cosine=("mean_sibling_cosine", "median"),
            mean_nearest_sibling_cosine=(
                "nearest_sibling_cosine", "mean"
            ),
            median_nearest_sibling_cosine=(
                "nearest_sibling_cosine", "median"
            ),
        )
        .reset_index()
    )
    unit_summary.to_csv(
        OUT / "cosine_sibling_summary.csv", index=False
    )
    print("\n=== WITHIN-SECTION SIBLING SIMILARITY ===")
    print(unit_summary.round(4).to_string(index=False))

print("\n=== PAIRWISE COSINE SUMMARY ===")
print(pair_summary.round(4).to_string(index=False))

print(f"\nProcessed document archives: {processed_archives}")
print(f"Embedding IDs without metadata: {missing_metadata}")
print(f"\nSaved analysis files to: {OUT}")
