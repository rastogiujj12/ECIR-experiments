#!/usr/bin/env python3
"""Read-only inventory of statistical methods, comparison families, and saved outputs.

Run from the project root:
  python3 /path/to/audit_statistical_design.py --root .

Writes:
  outputs/audit/statistical_design/statistical_design_inventory.txt
  outputs/audit/statistical_design/statistical_output_inventory.csv

This does not rerun retrieval or statistical tests and does not modify experiment outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_DIRS = ("src/sclc/analysis", "src/sclc/evaluation", "scripts")
SOURCE_EXTS = {".py"}
SOURCE_TERMS = re.compile(
    r"paired_document_bootstrap|holm_adjust|holm|multipletests|bonferroni|"
    r"bootstrap_iterations|comparison_family|family_size|family|p.adjust|"
    r"confidence.interval|confidence_interval|p_value|pvalue|significan|"
    r"effect_name|comparisons_within|analysis_set|sample_scope",
    re.IGNORECASE,
)
OUTPUT_TERMS = re.compile(
    r"comparison|bootstrap|stat|effect|summary|metric|aggregate|infer|"
    r"paired|confidence|signific|interaction|mechanism|retrieval_unit_size|"
    r"cross_section_challenge|manifest",
    re.IGNORECASE,
)
MANIFEST_TERMS = re.compile(
    r"kind|analysis|condition|model|chunk|query_count|document_count|"
    r"bootstrap|holm|comparison|correction|metric|files|configuration|"
    r"sample_scope|analysis_set|fingerprint|schema_version",
    re.IGNORECASE,
)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "site-packages"}


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def trim(v: Any, limit: int = 500) -> str:
    s = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def walk_files(folder: Path, suffixes: set[str]):
    if not folder.exists():
        return
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            if any(part in SKIP_DIRS for part in path.relative_to(folder).parts):
                continue
        except ValueError:
            pass
        yield path


def source_audit(root: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for folder_name in SOURCE_DIRS:
        folder = root / folder_name
        for path in walk_files(folder, SOURCE_EXTS) or []:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            indices = [i for i, line in enumerate(lines) if SOURCE_TERMS.search(line)]
            if not indices:
                continue
            # Keep snippets compact: merge matches that lie within 4 lines.
            groups: list[list[int]] = []
            for idx in indices:
                if not groups or idx - groups[-1][-1] > 5:
                    groups.append([idx])
                else:
                    groups[-1].append(idx)
            snippets = []
            for group in groups[:18]:
                lo = max(0, group[0] - 3)
                hi = min(len(lines), group[-1] + 4)
                snippets.append(
                    {
                        "start_line": lo + 1,
                        "end_line": hi,
                        "text": "\n".join(f"{n + 1:5}: {lines[n]}" for n in range(lo, hi)),
                    }
                )
            hits.append(
                {
                    "path": rel(path, root),
                    "line_count": len(lines),
                    "match_count": len(indices),
                    "snippets": snippets,
                }
            )
    hits.sort(
        key=lambda x: (
            0 if any(n in x["path"].lower() for n in ("statistics", "comparison", "challenge", "retrieval_unit_size")) else 1,
            -x["match_count"],
            x["path"],
        )
    )
    return hits


def scan_csv(path: Path, root: Path) -> dict[str, Any]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            row_count = 0
            samples: list[dict[str, Any]] = []
            for row in reader:
                row_count += 1
                if len(samples) < 2:
                    samples.append({k: (v[:160] if isinstance(v, str) else v) for k, v in row.items() if k})
        return {
            "path": rel(path, root),
            "rows": row_count,
            "columns": columns,
            "sample_rows": samples,
            "error": None,
        }
    except Exception as exc:
        return {
            "path": rel(path, root), "rows": None, "columns": [],
            "sample_rows": [], "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_manifest(obj: Any) -> dict[str, Any]:
    keep: dict[str, Any] = {}

    def walk(node: Any, prefix: str = "", depth: int = 0) -> None:
        if depth > 4 or len(keep) >= 80:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                key_path = f"{prefix}.{k}" if prefix else str(k)
                if MANIFEST_TERMS.search(str(k)):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        keep[key_path] = v
                    elif isinstance(v, list):
                        keep[key_path] = {"length": len(v), "sample": v[:2]}
                    elif isinstance(v, dict):
                        # Avoid dumping entire file inventories.
                        keep[key_path] = {"keys": list(v.keys())[:20], "length": len(v)}
                if isinstance(v, dict):
                    walk(v, key_path, depth + 1)
        elif isinstance(node, list):
            for idx, value in enumerate(node[:3]):
                if isinstance(value, dict):
                    walk(value, f"{prefix}[{idx}]", depth + 1)

    walk(obj)
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/audit/statistical_design"))
    parser.add_argument("--max-csv", type=int, default=350)
    parser.add_argument("--max-manifests", type=int, default=250)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit("STATISTICAL DESIGN AND OUTPUT INVENTORY")
    emit("=" * 76)
    emit(f"Repository root: {root}")
    emit("Read-only audit: this inventories code and saved files; it does not rerun statistical tests.")
    emit("")

    # Source methods and family definitions.
    hits = source_audit(root)
    emit("1. SOURCE CODE: STATISTICAL PROCEDURES / COMPARISON FAMILIES")
    emit("-" * 76)
    emit(f"Source files with statistical-term matches: {len(hits)}")
    for item in hits:
        emit(f"\n### {item['path']} ({item['match_count']} matching lines; {item['line_count']} total)")
        for snip in item["snippets"]:
            emit(f"[lines {snip['start_line']}-{snip['end_line']}]")
            emit(snip["text"])

    # Output files, excluding known huge/non-statistical diagnostics and raw retrieval artifacts.
    csv_candidates = []
    outputs = root / "outputs"
    if outputs.exists():
        for path in walk_files(outputs, {".csv"}) or []:
            if OUTPUT_TERMS.search(path.name) or OUTPUT_TERMS.search(str(path.parent)):
                # Instrumentation CSVs are not statistical evidence.
                if "lmlc_window_instrumentation" in str(path) or "lmlc_section_span_crosscheck" in str(path):
                    continue
                csv_candidates.append(path)
    csv_candidates.sort(key=lambda p: (0 if any(k in str(p).lower() for k in ("analysis", "statistics", "comparison", "bootstrap")) else 1, str(p)))
    truncated_csvs = len(csv_candidates) > args.max_csv
    csv_candidates = csv_candidates[: args.max_csv]
    csv_records = [scan_csv(p, root) for p in csv_candidates]

    emit("\n\n2. SAVED CSV OUTPUTS")
    emit("-" * 76)
    emit(f"CSV files inventoried: {len(csv_records)}" + (" (limited by --max-csv)" if truncated_csvs else ""))
    for item in csv_records:
        emit(f"\n### {item['path']}")
        if item["error"]:
            emit(f"ERROR: {item['error']}")
            continue
        emit(f"Rows: {item['rows']}")
        emit(f"Columns: {', '.join(item['columns'])}")
        for i, row in enumerate(item["sample_rows"], 1):
            emit(f"Sample row {i}: {trim(row, 600)}")

    # Manifests likely to document statistical/evaluation runs.
    manifest_candidates = []
    if outputs.exists():
        for path in walk_files(outputs, {".json"}) or []:
            if path.name.lower() != "manifest.json":
                continue
            ptext = str(path).lower()
            if any(k in ptext for k in ("evaluation", "analysis", "stat", "retrieval_unit_size", "cross_section_challenge", "mechanism")):
                manifest_candidates.append(path)
    manifest_candidates.sort(key=str)
    truncated_manifests = len(manifest_candidates) > args.max_manifests
    manifest_candidates = manifest_candidates[: args.max_manifests]

    emit("\n\n3. STATISTICAL / EVALUATION MANIFESTS")
    emit("-" * 76)
    emit(f"Manifests inventoried: {len(manifest_candidates)}" + (" (limited by --max-manifests)" if truncated_manifests else ""))
    manifest_rows: list[dict[str, Any]] = []
    for path in manifest_candidates:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_manifest(obj)
            manifest_rows.append({"path": rel(path, root), "summary": summary})
            emit(f"\n### {rel(path, root)}")
            emit(trim(summary, 3000))
        except Exception as exc:
            emit(f"\n### {rel(path, root)}\nERROR: {type(exc).__name__}: {exc}")

    # Machine-readable inventory for follow-up analysis.
    inv_path = outdir / "statistical_output_inventory.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "rows", "columns", "sample_rows", "error"],
        )
        writer.writeheader()
        for item in csv_records:
            writer.writerow(
                {
                    "path": item["path"],
                    "rows": item["rows"],
                    "columns": json.dumps(item["columns"], ensure_ascii=False),
                    "sample_rows": json.dumps(item["sample_rows"], ensure_ascii=False),
                    "error": item["error"],
                }
            )

    meta_path = outdir / "statistical_manifest_inventory.json"
    meta_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "source_matches": hits,
                "csv_inventory": csv_records,
                "manifest_inventory": manifest_rows,
                "truncated_csv_inventory": truncated_csvs,
                "truncated_manifest_inventory": truncated_manifests,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    emit("\n\n4. INVENTORY FILES")
    emit("-" * 76)
    emit(f"Text report: {outdir / 'statistical_design_inventory.txt'}")
    emit(f"CSV inventory: {inv_path}")
    emit(f"Manifest/source machine-readable inventory: {meta_path}")
    if truncated_csvs:
        emit(f"NOTE: CSV inventory capped at {args.max_csv}; rerun with --max-csv to increase.")
    if truncated_manifests:
        emit(f"NOTE: manifest inventory capped at {args.max_manifests}; rerun with --max-manifests to increase.")

    report_path = outdir / "statistical_design_inventory.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Statistical design inventory complete.")
    print(f"Text report: {report_path}")
    print(f"CSV inventory: {inv_path}")
    print(f"Manifest inventory: {meta_path}")
    print(f"Source files with matches: {len(hits)}")
    print(f"CSV outputs inventoried: {len(csv_records)}")
    print(f"Manifests inventoried: {len(manifest_candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
