from __future__ import annotations

import argparse
from pathlib import Path

from sclc.config import ensure_output_directories, load_config
from sclc.data.io import document_summary_frame, evidence_alignment_frame, write_documents_jsonl
from sclc.data.qasa import iter_qasa_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official QASA into the SCLC DocumentRecord format.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="Downloaded QASA v1.1 answerable JSON.")
    parser.add_argument("--limit-papers", type=int, default=None, help="Prepare only the first N papers for a smoke test.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_directories(config)

    documents = iter_qasa_documents(config, json_path=args.json, split="test", limit=args.limit_papers)
    if not documents:
        raise RuntimeError("No QASA documents were prepared")

    write_documents_jsonl(documents, config.paths.processed_dir / "documents.jsonl")
    document_summary_frame(documents).to_csv(
        config.paths.processed_dir / "document_summary.csv", index=False
    )
    evidence_alignment_frame(documents).to_csv(
        config.paths.processed_dir / "evidence_alignment.csv", index=False
    )

    print(f"Prepared {len(documents)} QASA papers.")
    print(f"Usable questions: {sum(document.usable_question_count for document in documents)}")
    print(f"Outputs: {config.paths.processed_dir}")
    print("Next: run `sclc profile --config configs/qasa.yaml` and then `sclc sample --config configs/qasa.yaml`." )


if __name__ == "__main__":
    main()
