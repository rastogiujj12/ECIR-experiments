import pandas as pd

from sclc.config import AppConfig
from sclc.data.sample import read_profile_csv, select_documents


def make_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 42},
            "paths": {
                "processed_dir": "data/processed",
                "profile_dir": "data/profiles",
                "subset_dir": "data/subsets",
                "hf_cache_dir": "cache/huggingface",
            },
            "dataset": {
                "repo_id": "allenai/qasper",
                "subset": "qasper",
                "splits": ["train", "validation", "test"],
            },
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 512,
                "overlap_tokens": 0,
            },
            "models": {
                "granite": {
                    "model_id": "granite",
                    "max_document_tokens": 32768,
                },
                "jina": {
                    "model_id": "jina",
                    "max_document_tokens": 8192,
                },
            },
            "sampling": {
                "core_documents": 6,
                "granite_extended_documents": 3,
                "minimum_usable_questions": 1,
                "length_bins": 3,
            },
        }
    )


def test_selection_is_document_level_and_reproducible() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "document_id": f"d{index}",
                "split": ["train", "validation", "test"][index % 3],
                "title": f"Paper {index}",
                "granite_tokens": 1000 + index * 500,
                "jina_tokens": 1000 + index * 500,
                "usable_question_count": 2,
                "eligibility_group": (
                    "cross_model_core" if index < 14 else "granite_extended"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    config = make_config()

    first = select_documents(config, frame)
    second = select_documents(config, frame)

    assert first["document_id"].tolist() == second["document_id"].tolist()
    assert (first["analysis_set"] == "cross_model_core").sum() == 6
    assert (first["analysis_set"] == "granite_extended").sum() == 3


def test_profile_csv_preserves_arxiv_like_document_ids(tmp_path) -> None:
    path = tmp_path / "document_lengths.csv"
    path.write_text(
        "document_id,split,granite_tokens\n"
        "1610.06510,train,1000\n"
        "1702.06700,test,2000\n",
        encoding="utf-8",
    )

    frame = read_profile_csv(path)

    assert frame["document_id"].tolist() == ["1610.06510", "1702.06700"]
