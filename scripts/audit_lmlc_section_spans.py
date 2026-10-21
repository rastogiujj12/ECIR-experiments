#!/usr/bin/env python3
"""Cross-check LMLC overlap diagnostics against saved top-level-section spans.

Run from repository root:
  python3 /path/to/audit_lmlc_section_spans.py \
    --root . \
    --details outputs/audit/lmlc_window_instrumentation/lmlc_window_details.csv \
    --config configs/qasper_full.yaml \
    --config configs/qasa.yaml \
    --config configs/doc2dial.yaml \
    --config configs/multidoc2dial.yaml

This reads existing CSV/JSONL/config artifacts only; no models or tokenizers are loaded.
Writes a concise TXT report and CSV of examples where logged overlap disagrees with
an independent intersection test against the saved section spans.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def bool_value(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def i(x: Any) -> int:
    return int(float(x))


def dataset_label(config_path: Path, config: Any) -> str:
    text = " ".join(
        [
            str(config_path),
            str(getattr(getattr(config, "dataset", None), "repo_id", "")),
            str(getattr(getattr(config, "paths", None), "processed_dir", "")),
        ]
    ).lower()
    for name in ("multidoc2dial", "doc2dial", "qasper", "qasa"):
        if name in text:
            return name
    return config_path.stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--outdir", type=Path, default=Path("outputs/audit/lmlc_section_span_crosscheck"))
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    root = args.root.resolve()
    details = args.details if args.details.is_absolute() else root / args.details
    if not details.exists():
        raise SystemExit(f"Details CSV not found: {details}")

    sys.path.insert(0, str(root / "src"))
    try:
        from sclc.config import load_config
        from sclc.data.retrieval_unit_io import read_top_level_sections
        from sclc.paths import retrieval_unit_dir
    except Exception as exc:
        raise SystemExit(
            "Could not import the project's SCLC modules. Run in the project environment. "
            f"Import error: {exc}"
        ) from exc

    configs: dict[str, tuple[Path, Any]] = {}
    warnings: list[str] = []
    for raw_path in args.config:
        config_path = raw_path if raw_path.is_absolute() else root / raw_path
        config_path = config_path.resolve()
        if not config_path.exists():
            warnings.append(f"Missing config: {config_path}")
            continue
        config = load_config(config_path)
        configs[dataset_label(config_path, config)] = (config_path, config)

    section_index: dict[tuple[str, int], dict[str, list[tuple[str, int, int]]]] = {}
    doc_section_counts: dict[tuple[str, int], dict[str, int]] = {}

    for dataset, (config_path, config) in configs.items():
        for size in args.sizes:
            path = retrieval_unit_dir(config, size) / "top_level_sections.jsonl"
            if not path.exists():
                warnings.append(f"{dataset} size={size}: missing {path}")
                continue
            sections = read_top_level_sections(path)
            per_doc: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
            for s in sections:
                per_doc[s.document_id].append(
                    (s.parent_section_id, int(s.span.start), int(s.span.end))
                )
            for doc, items in per_doc.items():
                items.sort(key=lambda x: (x[1], x[2], x[0]))
            section_index[(dataset, size)] = dict(per_doc)
            doc_section_counts[(dataset, size)] = {
                doc: len(items) for doc, items in per_doc.items()
            }

    # Aggregate rows. Recheck overlap independently using the persisted spans.
    grouped: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0, "docs": set(), "multi_section_docs": set(),
            "extend_parent": 0, "computed_overlap": 0, "logged_overlap": 0,
            "left_other_section": 0, "right_other_section": 0,
            "disagreements": 0, "overlap_chars": [],
            "computed_overlap_logged_false": 0,
        }
    )
    examples: list[dict[str, Any]] = []
    rows_seen = 0

    with details.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "dataset", "model", "chunk_size_tokens", "document_id",
            "retrieval_unit_id", "parent_section_id", "parent_char_start",
            "parent_char_end", "lmlc_window_char_start", "lmlc_window_char_end",
            "window_extends_outside_parent_span", "overlaps_nonparent_section",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Details CSV missing expected columns: {sorted(missing)}")

        for row in reader:
            rows_seen += 1
            dataset = row["dataset"]
            model = row["model"]
            size = i(row["chunk_size_tokens"])
            key = (dataset, model, size)
            start = i(row["lmlc_window_char_start"])
            end = i(row["lmlc_window_char_end"])
            parent_start = i(row["parent_char_start"])
            parent_end = i(row["parent_char_end"])
            parent_id = row["parent_section_id"]
            document_id = row["document_id"]
            logged_overlap = bool_value(row["overlaps_nonparent_section"])
            extends = bool_value(row["window_extends_outside_parent_span"])

            sections_for_dataset_size = section_index.get((dataset, size), {})
            sections = sections_for_dataset_size.get(document_id, [])
            section_count = len(sections)
            overlapping = [
                (sid, ss, se)
                for sid, ss, se in sections
                if sid != parent_id and start < se and end > ss
            ]
            computed_overlap = bool(overlapping)
            overlap_chars = sum(max(0, min(end, se) - max(start, ss)) for _, ss, se in overlapping)
            left_overlap = any(ss < parent_start and start < se and end > ss for _, ss, se in overlapping)
            right_overlap = any(se > parent_end and start < se and end > ss for _, ss, se in overlapping)

            agg = grouped[key]
            agg["n"] += 1
            agg["docs"].add(document_id)
            if section_count > 1:
                agg["multi_section_docs"].add(document_id)
            agg["extend_parent"] += int(extends)
            agg["computed_overlap"] += int(computed_overlap)
            agg["logged_overlap"] += int(logged_overlap)
            agg["left_other_section"] += int(left_overlap)
            agg["right_other_section"] += int(right_overlap)
            agg["disagreements"] += int(computed_overlap != logged_overlap)
            agg["computed_overlap_logged_false"] += int(computed_overlap and not logged_overlap)
            agg["overlap_chars"].append(overlap_chars)

            # Keep a small, useful sample whenever windows extend beyond parents
            # in multi-section documents, plus any disagreement with the saved flag.
            should_sample = (
                (extends and section_count > 1 and (left_overlap or right_overlap))
                or computed_overlap != logged_overlap
            )
            if should_sample and len(examples) < args.max_examples * 12:
                examples.append(
                    {
                        "dataset": dataset, "model": model, "chunk_size_tokens": size,
                        "document_id": document_id, "retrieval_unit_id": row["retrieval_unit_id"],
                        "parent_section_id": parent_id, "section_count_in_document": section_count,
                        "parent_span": f"[{parent_start},{parent_end})",
                        "window_span": f"[{start},{end})",
                        "extends_parent": extends,
                        "previous_section_ids_overlapped": ";".join(
                            sid for sid, ss, se in overlapping if se <= parent_start
                        ),
                        "next_section_ids_overlapped": ";".join(
                            sid for sid, ss, se in overlapping if ss >= parent_end
                        ),
                        "all_other_section_ids_overlapped": ";".join(sid for sid, _, _ in overlapping),
                        "computed_overlap": computed_overlap,
                        "logged_overlap": logged_overlap,
                        "overlap_chars": overlap_chars,
                    }
                )

    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    examples_path = outdir / "lmlc_section_overlap_examples.csv"
    report_path = outdir / "lmlc_section_span_crosscheck.txt"

    # Limit samples evenly across groups and sort for readable output.
    examples.sort(key=lambda r: (r["dataset"], r["model"], int(r["chunk_size_tokens"]), r["document_id"]))
    write_examples = examples[: args.max_examples * 12]
    if write_examples:
        with examples_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(write_examples[0].keys()))
            writer.writeheader()
            writer.writerows(write_examples)
    else:
        examples_path.write_text("", encoding="utf-8")

    lines = [
        "LMLC SAVED SECTION-SPAN CROSS-CHECK",
        "=" * 72,
        f"Rows scanned: {rows_seen:,}",
        f"Detail CSV: {details}",
        f"Example CSV: {examples_path}",
        "",
        "SUMMARY BY DATASET / MODEL / UNIT SIZE",
        "-" * 72,
        "dataset\tmodel\tsize\trows\tdocs\tmulti-section-docs\textends-parent\tcomputed-overlap\tlogged-overlap\tdisagreements\tleft-overlap\tright-overlap\tmedian-overlap-chars",
    ]
    for (dataset, model, size), a in sorted(grouped.items()):
        n = a["n"]
        lines.append(
            f"{dataset}\t{model}\t{size}\t{n}\t{len(a['docs'])}\t{len(a['multi_section_docs'])}\t"
            f"{a['extend_parent']/n:.6f}\t{a['computed_overlap']/n:.6f}\t{a['logged_overlap']/n:.6f}\t"
            f"{a['disagreements']}\t{a['left_other_section']/n:.6f}\t{a['right_other_section']/n:.6f}\t"
            f"{statistics.median(a['overlap_chars']) if a['overlap_chars'] else 0}"
        )
    if warnings:
        lines += ["", "WARNINGS", "-" * 72, *warnings]
    lines += [
        "",
        "INTERPRETATION GUIDE",
        "-" * 72,
        "- 'computed-overlap' is independently recomputed by intersecting each saved LMLC character window",
        "  with every saved top-level-section span from the same document, excluding the target's parent.",
        "- If computed-overlap is also zero, the result may be genuine; inspect multi-section document counts",
        "  and sample windows. If it is positive but logged-overlap is zero, the original diagnostic is wrong.",
        "- The fraction of windows extending beyond a parent is not itself evidence of crossing into another",
        "  section; the window may extend into unsectioned gaps or beyond the document's only parent span.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Cross-check complete. Report: {report_path}")
    print(f"Examples: {examples_path}")
    print(f"Rows scanned: {rows_seen:,}")
    return 0


def write_examples(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
