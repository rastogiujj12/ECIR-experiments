from pathlib import Path

from sclc.config import load_config


def test_multidoc2dial_uses_dataset_specific_paths() -> None:
    config = load_config(Path("configs/multidoc2dial.yaml"))

    assert str(config.paths.processed_dir).startswith("data/multidoc2dial/")
    assert str(config.paths.profile_dir).startswith("data/multidoc2dial/")
    assert str(config.paths.subset_dir).startswith("data/multidoc2dial/")
    assert str(config.paths.retrieval_unit_dir).startswith("data/multidoc2dial/")
    assert str(config.paths.encoding_dir).startswith("outputs/multidoc2dial/")
    assert str(config.paths.ranking_dir).startswith("outputs/multidoc2dial/")
    assert str(config.paths.evaluation_dir).startswith("outputs/multidoc2dial/")
    assert str(config.paths.analysis_dir).startswith("outputs/multidoc2dial/")
