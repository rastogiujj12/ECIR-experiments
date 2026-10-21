#!/usr/bin/env python3
"""Audit the LMLC / length_matched implementation and saved artifacts.

Run from the repository root:
    python3 /path/to/audit_lmlc.py --root .

Writes:
    outputs/audit/lmlc_implementation_audit.txt
    outputs/audit/lmlc_implementation_audit.json

The script is intentionally read-only. It does not rebuild encodings or modify outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TEXT_EXTS = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".sh", ".txt", ".ini"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", "site-packages", "dist", "build"
}
TERMS = re.compile(
    r"length[_ -]?matched|\blmlc\b|section[_ -]?constrained|\bsclc\b|\bglc\b|"
    r"context[_ -]?(scope|window|length)|parent[_ -]?section|section[_ -]?id|"
    r"pool(ing|ed)?|offset_mapping|token[_ -]?(start|end|span|window)|"
    r"max[_ -]?sequence[_ -]?length|window[_ -]?size|chunk[_ -]?size|late[_ -]?chunk",
    re.IGNORECASE,
)
COND_ALIASES = {
    "length_matched": "length_matched", "length-matched": "length_matched", "lmlc": "length_matched",
    "section_constrained": "section_constrained", "section-constrained": "section_constrained", "sclc": "section_constrained",
    "section_isolated": "section_isolated", "section-isolated": "section_isolated",
    "global": "global", "glc": "global",
    "fixed_dense": "fixed_dense", "fixed-dense": "fixed_dense",
}
ARRAY_KEY_TERMS = re.compile(
    r"unit|chunk|retriev|doc|section|span|start|end|offset|token|embedding|vector|pool|index|id",
    re.IGNORECASE,
)


def short(value: Any, max_chars: int = 360) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = repr(value)
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def sha256_array(arr: Any) -> str:
    # numpy arrays are normalized to a contiguous byte representation.
    import numpy as np  # type: ignore
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def find_condition(path: Path) -> str | None:
    for part in path.parts:
        canon = COND_ALIASES.get(part.lower())
        if canon:
            return canon
    # Some output conventions put condition only in filename.
    name = path.name.lower()
    for alias, canon in COND_ALIASES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", name):
            return canon
    return None


def find_dataset(path: Path) -> str | None:
    known = {"qasper", "qasa", "doc2dial", "multidoc2dial"}
    for part in path.parts:
        if part.lower() in known:
            return part.lower()
    return None


def find_model(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts):
        if any(s in part for s in ("granite", "jina")):
            if "granite" in part:
                return "granite"
            if "jina" in part:
                return "jina"
    return None


def find_chunk_size(path: Path) -> str | None:
    for part in path.parts:
        m = re.search(r"(?:chunk[_ -]?)?(128|256|512)(?:tokens?)?", part.lower())
        if m:
            return m.group(1)
    return None


def manifest_summary(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    wanted = re.compile(
        r"condition|chunk|model|segment|context|scope|document|unit|token|dataset|schema|version|"
        r"retriev|window|pool|section|encoding|created|limit|excluded|filter|hash|path|file",
        re.IGNORECASE,
    )
    def walk(node: Any, prefix: str = "", depth: int = 0) -> None:
        if depth > 5 or len(out) >= 160:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if wanted.search(str(k)):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        out[key] = v
                    elif isinstance(v, list):
                        out[key] = {"type": "list", "length": len(v), "sample": v[:3]}
                    elif isinstance(v, dict):
                        out[key] = {"type": "object", "keys": list(v.keys())[:30]}
                walk(v, key, depth + 1)
        elif isinstance(node, list):
            for i, v in enumerate(node[:5]):
                walk(v, f"{prefix}[{i}]", depth + 1)
    walk(obj)
    return out


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Repository root (default: current directory)")
    ap.add_argument("--out", default=None, help="Output directory (default: ROOT/outputs/audit)")
    ap.add_argument("--context-lines", type=int, default=8, help="Lines before/after each code match (default: 8)")
    ap.add_argument("--max-files", type=int, default=100, help="Max source files included in code excerpts")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: repository root is not a directory: {root}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "repository_root": str(root),
        "audit_scope": "read-only static source scan + saved manifest/NPZ inventory; does not rerun encoders",
        "source_matches": [],
        "manifests": [],
        "encoding_artifacts": [],
        "comparisons": [],
        "cross_condition_array_comparisons": [],
        "limitations": [],
    }
    text_lines: list[str] = []
    def emit(s: str = "") -> None:
        text_lines.append(s)

    emit("LMLC / length_matched IMPLEMENTATION AUDIT")
    emit("=" * 72)
    emit(f"Repository root: {root}")
    emit("Purpose: determine what the current implementation and stored artifacts say about LMLC.")
    emit("This is a read-only audit; no model inference or encoding rebuild is performed.")
    emit()

    # Source/config scan. Prefer implementation/config/test files; do not scan large output trees.
    scan_roots = []
    for name in ("src", "tests", "test", "configs", "config", "scripts", "experiments", "notebooks"):
        p = root / name
        if p.exists() and p.is_dir():
            scan_roots.append(p)
    if not scan_roots:
        scan_roots = [root]
    source_hits = []
    for base in scan_roots:
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = content.splitlines()
            match_indices = [i for i, line in enumerate(lines) if TERMS.search(line)]
            if not match_indices:
                continue
            snippets = []
            occupied_until = -1
            for idx in match_indices:
                start = max(0, idx - args.context_lines)
                end = min(len(lines), idx + args.context_lines + 1)
                if start <= occupied_until:
                    if snippets:
                        prev = snippets[-1]
                        prev["end"] = max(prev["end"], end)
                        prev["indices"].append(idx)
                        occupied_until = prev["end"]
                    continue
                snippets.append({"start": start, "end": end, "indices": [idx]})
                occupied_until = end
            file_entry = {"path": safe_rel(p, root), "line_count": len(lines), "match_count": len(match_indices), "snippets": []}
            for sn in snippets:
                snippet_text = "\n".join(f"{n+1:5d}: {lines[n]}" for n in range(sn["start"], sn["end"]))
                file_entry["snippets"].append({"start_line": sn["start"] + 1, "end_line": sn["end"], "text": snippet_text})
            source_hits.append(file_entry)
    source_hits.sort(key=lambda x: (
        0 if any(k in x["path"].lower() for k in ("chunk", "encod", "pipeline", "retriev", "lmlc", "condition")) else 1,
        -x["match_count"], x["path"]
    ))
    source_hits = source_hits[: args.max_files]
    report["source_matches"] = source_hits
    emit("1. SOURCE / CONFIG / TEST REFERENCES")
    emit("-" * 72)
    if not source_hits:
        emit("No matches found in standard source/config/test directories.")
    for entry in source_hits:
        emit(f"\n### {entry['path']} ({entry['match_count']} matching lines; {entry['line_count']} total lines)")
        for sn in entry["snippets"]:
            emit(f"[lines {sn['start_line']}-{sn['end_line']}]\n{sn['text']}")

    # Find manifests and encoding files, excluding environments/build folders.
    manifest_paths = []
    npz_paths = []
    for base in (root / "outputs", root / "artifacts", root / "data"):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
                continue
            lower = str(p).lower()
            if p.name.lower() == "manifest.json":
                manifest_paths.append(p)
            elif p.suffix.lower() == ".npz" and (find_condition(p) or "encoding" in lower or "embedding" in lower):
                npz_paths.append(p)

    emit("\n\n2. MANIFESTS")
    emit("-" * 72)
    for p in sorted(set(manifest_paths)):
        if not (find_condition(p) or "encoding" in str(p).lower() or "encod" in str(p).lower()):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            report["manifests"].append({"path": safe_rel(p, root), "error": str(e)})
            emit(f"\n### {safe_rel(p, root)}\nERROR reading JSON: {e}")
            continue
        summary = manifest_summary(obj)
        entry = {
            "path": safe_rel(p, root), "dataset": find_dataset(p), "condition_from_path": find_condition(p),
            "model_from_path": find_model(p), "chunk_size_from_path": find_chunk_size(p),
            "summary": summary,
        }
        report["manifests"].append(entry)
        emit(f"\n### {entry['path']}")
        emit(f"Path inference: dataset={entry['dataset']}, condition={entry['condition_from_path']}, model={entry['model_from_path']}, chunk_size={entry['chunk_size_from_path']}")
        emit("Fields (filtered to audit-relevant keys):")
        emit(short(summary, 16000))

    emit("\n\n3. SAVED NPZ ENCODING ARTIFACTS")
    emit("-" * 72)
    npz_records = []
    np = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        emit("numpy is unavailable; skipped NPZ content inspection.")
    for p in sorted(set(npz_paths)):
        dataset, cond, model, chunk = find_dataset(p), find_condition(p), find_model(p), find_chunk_size(p)
        record: dict[str, Any] = {
            "path": safe_rel(p, root), "dataset": dataset, "condition": cond, "model": model,
            "chunk_size": chunk, "file_size_bytes": p.stat().st_size,
        }
        if np is not None:
            try:
                with np.load(p, allow_pickle=False) as data:
                    arrays = {}
                    for key in data.files:
                        arr = data[key]
                        item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                        if arr.nbytes <= 512 * 1024 * 1024:
                            item["sha256"] = sha256_array(arr)
                        if ARRAY_KEY_TERMS.search(key):
                            if arr.size <= 60:
                                item["sample"] = arr.tolist()
                            elif arr.ndim == 1:
                                item["sample"] = arr[:8].tolist()
                        arrays[key] = item
                    record["arrays"] = arrays
            except Exception as e:
                record["error"] = f"Could not inspect NPZ: {e}"
        npz_records.append(record)
    report["encoding_artifacts"] = npz_records
    if not npz_records:
        emit("No NPZ artifacts found under outputs/, artifacts/, or data/ using the path/name filters.")
    else:
        for r in npz_records:
            emit(f"\n### {r['path']}")
            emit(f"dataset={r['dataset']} condition={r['condition']} model={r['model']} chunk_size={r['chunk_size']} size={r['file_size_bytes']} bytes")
            if "error" in r:
                emit(r["error"])
            for key, info in r.get("arrays", {}).items():
                emit(f"  - {key}: {short(info, 800)}")

    # Compare NPZs across conditions on normalized relative identity within each dataset/model/chunk.
    emit("\n\n4. CROSS-CONDITION ARTIFACT COMPARISONS")
    emit("-" * 72)
    relevant = [r for r in npz_records if r.get("dataset") and r.get("condition") and r.get("model") and r.get("chunk_size")]
    # Normalize paths by removing the condition component, to align same document file where possible.
    def normalized_path(rel: str, condition: str) -> str:
        parts = Path(rel).parts
        removed = False
        kept = []
        for part in parts:
            if not removed and COND_ALIASES.get(part.lower()) == condition:
                removed = True
                continue
            kept.append(part)
        if not removed:
            # if condition is encoded only in filename, strip common aliases
            name = kept[-1] if kept else rel
            for alias, canon in COND_ALIASES.items():
                if canon == condition:
                    name = re.sub(re.escape(alias), "COND", name, flags=re.IGNORECASE)
            if kept:
                kept[-1] = name
        return str(Path(*kept))

    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in relevant:
        key = (r["dataset"], r["model"], r["chunk_size"])
        groups[key][r["condition"]].append(r)
    comparison_results = []
    for (dataset, model, chunk), cond_map in sorted(groups.items()):
        if "length_matched" not in cond_map:
            continue
        lmlc_items = {normalized_path(r["path"], "length_matched"): r for r in cond_map["length_matched"]}
        for other_cond in ("section_constrained", "section_isolated", "global", "fixed_dense"):
            other_items = {normalized_path(r["path"], other_cond): r for r in cond_map.get(other_cond, [])}
            common = sorted(set(lmlc_items) & set(other_items))
            result: dict[str, Any] = {
                "dataset": dataset, "model": model, "chunk_size": chunk, "comparison": f"length_matched vs {other_cond}",
                "lmlc_artifact_count": len(lmlc_items), "other_artifact_count": len(other_items), "matched_artifact_count": len(common),
                "array_comparisons": [],
            }
            for normalized in common:
                a = lmlc_items[normalized].get("arrays", {})
                b = other_items[normalized].get("arrays", {})
                common_keys = sorted(set(a) & set(b))
                for ak in common_keys:
                    ai, bi = a[ak], b[ak]
                    if "sha256" not in ai or "sha256" not in bi:
                        continue
                    if ai.get("shape") != bi.get("shape"):
                        status = "shape_differs"
                    elif ai["sha256"] == bi["sha256"]:
                        status = "identical_array"
                    else:
                        status = "same_shape_different_values"
                    result["array_comparisons"].append({
                        "artifact": normalized, "array_key": ak, "status": status,
                        "lmlc_shape": ai.get("shape"), "other_shape": bi.get("shape"),
                    })
            comparison_results.append(result)
            report["cross_condition_array_comparisons"].append(result)
            counts = defaultdict(int)
            for c in result["array_comparisons"]:
                counts[c["status"]] += 1
            emit(f"\n{dataset} / {model} / {chunk}: length_matched vs {other_cond}")
            emit(f"  matched NPZ artifacts: {len(common)} (LMLC={len(lmlc_items)}, other={len(other_items)})")
            emit(f"  common array key comparisons: {len(result['array_comparisons'])}; statuses={dict(counts)}")
            if len(common) == 0:
                emit("  NOTE: no path-aligned files; inspect manifests / output layout manually.")
            elif result["array_comparisons"]:
                examples = result["array_comparisons"][:8]
                for c in examples:
                    emit(f"    {c['status']}: {c['artifact']} :: {c['array_key']} {c.get('lmlc_shape')} vs {c.get('other_shape')}")
        report["comparisons"].append({
            "dataset": dataset, "model": model, "chunk_size": chunk,
            "conditions_present": sorted(cond_map.keys()),
            "condition_artifact_counts": {k: len(v) for k, v in cond_map.items()},
        })

    # Summary / explicit caveats.
    emit("\n\n5. WHAT THIS AUDIT CAN AND CANNOT ESTABLISH")
    emit("-" * 72)
    emit("The source excerpts should reveal how LMLC chooses its context window and how target-unit vectors are pooled.")
    emit("Manifest and NPZ comparisons can reveal whether conditions use the same document artifacts, array schemas, unit IDs/spans (if saved), and whether vectors are byte-identical or differ.")
    emit("Identical retrieval-unit IDs/spans do NOT prove that the contextual token window is length-matched or crosses section boundaries; that must be verified from the actual LMLC window-selection code and its inputs.")
    emit("Different embeddings do NOT prove that the intended context window was used; they only show that the saved arrays differ.")
    emit("If the current artifacts omit selected context token ranges, the next step is a small instrumentation run that records (document_id, target_unit_id, target token span, context token span, context token count, section-boundary crossing, pooling span) for a sample from each dataset/model/chunk size.")
    report["limitations"] = [
        "Static scan can miss dynamically constructed behavior or code outside scanned directories.",
        "Manifest keys and NPZ arrays are inferred from actual files; absent metadata cannot be reconstructed by this read-only audit.",
        "Identical unit IDs/spans do not prove the selected LMLC context window is correctly length-matched or crosses section boundaries.",
        "Different embeddings alone do not prove the intended context window was used.",
    ]

    txt_path = out_dir / "lmlc_implementation_audit.txt"
    json_path = out_dir / "lmlc_implementation_audit.json"
    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print("LMLC audit finished.")
    print(f"Text report: {txt_path}")
    print(f"Machine-readable report: {json_path}")
    print(f"Source files with matches: {len(source_hits)}")
    print(f"Manifests inspected: {len(report['manifests'])}")
    print(f"NPZ artifacts inspected: {len(npz_records)}")
    print("Please send back the TXT report first; the JSON helps if the report is large or we need deeper comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
