from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sclc.config import ensure_output_directories, load_config
from sclc.data.doc2dial import iter_doc2dial_documents
from sclc.data.io import document_summary_frame, evidence_alignment_frame, write_documents_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare official Doc2Dial in the existing SCLC DocumentRecord format."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Doc2Dial dialogue split to use. Validation is the fixed evaluation split.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_directories(config)

    documents = list(iter_doc2dial_documents(config, split=args.split))
    if not documents:
        raise RuntimeError("No usable Doc2Dial documents were produced")

    write_documents_jsonl(
        documents,
        config.paths.processed_dir / "documents.jsonl",
    )
    document_summary_frame(documents).to_csv(
        config.paths.processed_dir / "document_summary.csv",
        index=False,
    )
    evidence_alignment_frame(documents).to_csv(
        config.paths.processed_dir / "evidence_alignment.csv",
        index=False,
    )

    selected = pd.DataFrame(
        {
            "document_id": [document.document_id for document in documents],
            "analysis_set": ["cross_model_core"] * len(documents),
        }
    )
    selected["length_stratum"] = "all"
    selected.to_csv(
        config.paths.subset_dir / "selected_documents.csv",
        index=False,
    )

    print(f"Prepared {len(documents)} Doc2Dial documents ({args.split}).")
    print(f"Usable queries: {sum(document.usable_question_count for document in documents)}")


if __name__ == "__main__":
    main()
