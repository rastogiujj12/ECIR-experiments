from __future__ import annotations

from pathlib import Path

import typer

from sclc.commands import (
    build_units_command,
    challenge_analyse_command,
    compare_command,
    encode_command,
    evaluate_command,
    prepare_command,
    profile_command,
    qasper_audit_command,
    qasper_challenge_command,
    qasper_challenge_finalize_command,
    retrieval_unit_size_command,
    retrieve_command,
    sample_command,
)
from sclc.options import EmbeddingModel, RetrievalCondition

app = typer.Typer(
    name="sclc",
    help="Section-constrained late-chunking retrieval experiment.",
    no_args_is_help=True,
)


@app.command()
def prepare(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Load and reconstruct the QASPER papers."""
    prepare_command(config)


@app.command()
def profile(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Count full-document tokens and assign corpus eligibility groups."""
    profile_command(config)


@app.command()
def sample(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Select reproducible document-level samples stratified by length."""
    sample_command(config)


@app.command("build-units")
def build_units(
    retrieval_unit_size: int | None = typer.Option(
        None,
        "--retrieval-unit-size",
        min=1,
        help="Retrieval-unit size in canonical tokens; defaults to the configured primary size.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Construct canonical continuous and section-bounded retrieval units."""
    build_units_command(config, retrieval_unit_size=retrieval_unit_size)


@app.command()
def encode(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Representation condition to construct.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    retrieval_unit_size: int | None = typer.Option(
        None,
        "--retrieval-unit-size",
        min=1,
        help="Retrieval-unit size namespace; defaults to the configured primary size.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached representations that already exist.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Construct the lexical or dense representation for one condition."""
    encode_command(
        config,
        condition=condition,
        model=model,
        retrieval_unit_size=retrieval_unit_size,
        overwrite=overwrite,
    )


@app.command()
def retrieve(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Retrieval condition to rank.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    retrieval_unit_size: int | None = typer.Option(
        None,
        "--retrieval-unit-size",
        min=1,
        help="Retrieval-unit size namespace; defaults to the configured primary size.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached rankings.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Rank candidates within each query's associated paper."""
    retrieve_command(
        config,
        condition=condition,
        model=model,
        retrieval_unit_size=retrieval_unit_size,
        overwrite=overwrite,
    )


@app.command()
def evaluate(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Retrieval condition to evaluate.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    retrieval_unit_size: int | None = typer.Option(
        None,
        "--retrieval-unit-size",
        min=1,
        help="Retrieval-unit size namespace; defaults to the configured primary size.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached metric outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Evaluate rankings against alternative and union QASPER evidence."""
    evaluate_command(
        config,
        condition=condition,
        model=model,
        retrieval_unit_size=retrieval_unit_size,
        overwrite=overwrite,
    )


@app.command()
def compare(
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help=(
            "Restrict comparisons to one embedding model. Omit only after all "
            "configured models have been evaluated."
        ),
    ),
    retrieval_unit_size: int | None = typer.Option(
        None,
        "--retrieval-unit-size",
        min=1,
        help="Selected retrieval-unit size namespace.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached bootstrap comparisons.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Run document-level paired bootstrap comparisons and Holm correction."""
    compare_command(
        config, model=model, retrieval_unit_size=retrieval_unit_size, overwrite=overwrite
    )


@app.command("qasper-audit")
def qasper_audit(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild the complete QASPER evidence-structure audit.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Audit cross-section evidence across the complete prepared QASPER corpus."""
    qasper_audit_command(config, overwrite=overwrite)


@app.command("qasper-challenge")
def qasper_challenge(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild the frozen strict cross-section challenge files.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Freeze test-only challenge manifests and a blinded review sheet."""
    qasper_challenge_command(config, overwrite=overwrite)


@app.command("qasper-challenge-finalize")
def qasper_challenge_finalize(
    decisions: Path = typer.Option(
        Path("data/subsets_cross_section_challenge/review_decisions.csv"),
        "--decisions", exists=True, dir_okay=False, readable=True,
        help="Completed manual review decisions.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    config: Path = typer.Option(
        Path("configs/base.yaml"), exists=True, dir_okay=False, readable=True,
        help="Path to the primary YAML configuration.",
    ),
) -> None:
    """Validate the manual review and freeze the accepted challenge questions."""
    qasper_challenge_finalize_command(config, decisions_path=decisions, overwrite=overwrite)


@app.command("challenge-analyse")
def challenge_analyse(
    model: EmbeddingModel = typer.Option(EmbeddingModel.JINA, "--model", case_sensitive=False),
    accepted_queries: Path = typer.Option(
        Path("data/subsets_cross_section_challenge/review_decisions.csv"),
        "--accepted-queries", exists=True, dir_okay=False, readable=True,
        help="Reviewed challenge decisions containing the accepted query IDs.",
    ),
    retrieval_unit_sizes: str = typer.Option("128,256,512", "--retrieval-unit-sizes"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    config: Path = typer.Option(
        Path("configs/cross_section_challenge.yaml"), exists=True, dir_okay=False, readable=True,
        help="Path to the challenge YAML configuration.",
    ),
) -> None:
    """Analyse the accepted cross-section challenge questions."""
    try:
        parsed_sizes = tuple(int(v.strip()) for v in retrieval_unit_sizes.split(",") if v.strip())
    except ValueError as error:
        raise typer.BadParameter(
            "--retrieval-unit-sizes must be a comma-separated list of integers"
        ) from error
    if len(parsed_sizes) < 1:
        raise typer.BadParameter("--retrieval-unit-sizes must contain at least one size")
    challenge_analyse_command(
        config, accepted_queries_path=accepted_queries, model=model,
        retrieval_unit_sizes=parsed_sizes, overwrite=overwrite,
    )


@app.command("retrieval-unit-size")
def retrieval_unit_size(
    model: EmbeddingModel = typer.Option(
        ...,
        "--model",
        case_sensitive=False,
        help="Embedding model whose completed evaluations should be analysed.",
    ),
    retrieval_unit_sizes: str = typer.Option(
        "128,256,512",
        "--retrieval-unit-sizes",
        help="Comma-separated retrieval-unit sizes with completed dense evaluations.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached retrieval-unit-size analysis outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Analyse retrieval performance across 128, 256, and 512-token units."""
    try:
        parsed_sizes = tuple(
            int(value.strip())
            for value in retrieval_unit_sizes.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise typer.BadParameter(
            "--retrieval-unit-sizes must be a comma-separated list of integers"
        ) from error
    if len(parsed_sizes) < 2:
        raise typer.BadParameter("--retrieval-unit-sizes must contain at least two sizes")
    retrieval_unit_size_command(
        config,
        model=model,
        retrieval_unit_sizes=parsed_sizes,
        overwrite=overwrite,
    )




if __name__ == "__main__":
    app()
