from __future__ import annotations

from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from sclc.analysis.cross_section_challenge import (
    finalize_cross_section_challenge,
    freeze_cross_section_challenge,
)
from sclc.analysis.cross_section_challenge_analysis import analyse_cross_section_challenge
from sclc.analysis.qasper_collection import audit_qasper_collection
from sclc.analysis.retrieval_unit_size import analyse_retrieval_unit_size
from sclc.config import ensure_output_directories, load_config
from sclc.data.chunk_pipeline import construct_retrieval_units
from sclc.data.io import (
    document_summary_frame,
    evidence_alignment_frame,
    write_documents_jsonl,
)
from sclc.data.profile import profile_documents, write_profile_outputs
from sclc.data.qasper import iter_raw_records
from sclc.data.query_types import validate_query_type_coding
from sclc.data.reconstruct import reconstruct_document
from sclc.data.sample import read_profile_csv, select_documents
from sclc.encoding.pipeline import encode_condition
from sclc.evaluation.metrics import evaluate_condition
from sclc.evaluation.statistics import compare_conditions
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import (
    encoding_dir,
    evaluation_dir,
    ranking_dir,
    resolve_retrieval_unit_size,
    retrieval_unit_dir,
)
from sclc.retrieval.ranking import rank_condition

console = Console()


def prepare_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)

    documents = []
    for split, raw in tqdm(iter_raw_records(config), desc="Preparing QASPER papers"):
        documents.append(reconstruct_document(raw, split=split, config=config.document))

    documents_path = config.paths.processed_dir / "documents.jsonl"
    write_documents_jsonl(documents, documents_path)

    document_summary_frame(documents).to_csv(
        config.paths.processed_dir / "document_summary.csv",
        index=False,
    )
    evidence_alignment_frame(documents).to_csv(
        config.paths.processed_dir / "evidence_alignment.csv",
        index=False,
    )

    console.print(f"[green]Prepared {len(documents)} papers.[/green]")
    console.print(f"Documents: {documents_path}")


def profile_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    frame = profile_documents(config)
    write_profile_outputs(frame, config.paths.profile_dir)

    counts = frame["eligibility_group"].value_counts().to_dict()
    console.print("[green]Document profiling complete.[/green]")
    for group, count in counts.items():
        console.print(f"{group}: {count}")


def sample_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)

    profile_path = config.paths.profile_dir / "document_lengths.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} does not exist. Run `sclc profile` first.")

    profile = read_profile_csv(profile_path)
    selected = select_documents(config, profile)
    output_path = config.paths.subset_dir / "selected_documents.csv"
    selected.to_csv(output_path, index=False)

    console.print(f"[green]Selected {len(selected)} papers.[/green]")
    console.print(f"Subset manifest: {output_path}")


def build_units_command(
    config_path: Path, *, retrieval_unit_size: int | None
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_retrieval_unit_size = resolve_retrieval_unit_size(
        config, retrieval_unit_size
    )
    validation = construct_retrieval_units(
        config, chunk_size_tokens=resolved_retrieval_unit_size
    )

    counts = validation["counts"]
    console.print("[green]Retrieval-unit construction complete.[/green]")
    console.print(f"Continuous units: {counts['continuous_units']}")
    console.print(f"Section-bounded units: {counts['section_bounded_units']}")
    console.print(f"Usable queries: {counts['usable_queries']}")
    console.print(f"Validation: {validation['status']}")
    console.print(f"Outputs: {retrieval_unit_dir(config, resolved_retrieval_unit_size)}")


def encode_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    retrieval_unit_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_retrieval_unit_size = resolve_retrieval_unit_size(
        config, retrieval_unit_size
    )

    if condition.is_dense and model is not None:
        model_config = (
            config.models.granite
            if model is EmbeddingModel.GRANITE
            else config.models.jina
        )
        if model_config.revision is None:
            console.print(
                "[yellow]Warning: the model revision is not pinned. Set an immutable "
                "Hugging Face commit hash before the final dissertation run.[/yellow]"
            )

    try:
        result = encode_condition(
            config,
            condition=condition,
            model=model,
            chunk_size=resolved_retrieval_unit_size,
            overwrite=overwrite,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise

    console.print(
        "[green]Encoding complete for "
        f"{condition.value} at {resolved_retrieval_unit_size} tokens.[/green]"
    )
    console.print(f"Documents: {result['document_count']}")
    console.print(f"Retrieval units: {result['unit_count']}")
    if "query_count" in result:
        console.print(f"Queries: {result['query_count']}")
    if condition is RetrievalCondition.BM25:
        output_path = encoding_dir(config, resolved_retrieval_unit_size) / "bm25"
    else:
        assert model is not None
        output_path = (
            encoding_dir(config, resolved_retrieval_unit_size) / condition.value / model.value
        )
    console.print(f"Outputs: {output_path}")


def retrieve_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    retrieval_unit_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_retrieval_unit_size = resolve_retrieval_unit_size(
        config, retrieval_unit_size
    )
    validate_query_type_coding(
        config, model, chunk_size=resolved_retrieval_unit_size
    )
    result = rank_condition(
        config,
        condition=condition,
        model=model,
        chunk_size=resolved_retrieval_unit_size,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Retrieval complete for {condition.value} at "
        f"{resolved_retrieval_unit_size} tokens.[/green]"
    )
    console.print(f"Queries: {result['query_count']}")
    console.print(f"Ranked results: {result['result_count']}")
    if condition is RetrievalCondition.BM25:
        output_path = ranking_dir(config, resolved_retrieval_unit_size) / condition.value
    else:
        if model is None:
            raise ValueError(f"--model is required for {condition.value}")
        output_path = (
            ranking_dir(config, resolved_retrieval_unit_size)
            / condition.value
            / model.value
        )
    console.print(f"Outputs: {output_path}")


def evaluate_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    retrieval_unit_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_retrieval_unit_size = resolve_retrieval_unit_size(
        config, retrieval_unit_size
    )
    result = evaluate_condition(
        config,
        condition=condition,
        model=model,
        chunk_size=resolved_retrieval_unit_size,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Evaluation complete for {condition.value} at "
        f"{resolved_retrieval_unit_size} tokens.[/green]"
    )
    console.print(f"Queries: {result['query_count']}")
    console.print(f"Documents: {result['document_count']}")
    console.print(f"Classified queries: {result['classified_query_count']}")
    if condition is RetrievalCondition.BM25:
        output_path = evaluation_dir(config, resolved_retrieval_unit_size) / condition.value
    else:
        if model is None:
            raise ValueError(f"--model is required for {condition.value}")
        output_path = (
            evaluation_dir(config, resolved_retrieval_unit_size)
            / condition.value
            / model.value
        )
    console.print(f"Outputs: {output_path}")


def compare_command(
    config_path: Path,
    *,
    model: EmbeddingModel | None,
    retrieval_unit_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_retrieval_unit_size = resolve_retrieval_unit_size(
        config, retrieval_unit_size
    )
    selected_models = None if model is None else (model,)
    result = compare_conditions(
        config,
        chunk_size=resolved_retrieval_unit_size,
        models=selected_models,
        overwrite=overwrite,
    )
    console.print("[green]Pairwise statistical comparisons complete.[/green]")
    console.print(f"Comparisons: {result['comparison_count']}")
    comparison_path = evaluation_dir(config, resolved_retrieval_unit_size) / "comparisons"
    if model is not None:
        comparison_path = comparison_path / model.value
    console.print(f"Outputs: {comparison_path}")


def qasper_challenge_command(config_path: Path, *, overwrite: bool) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = freeze_cross_section_challenge(config, overwrite=overwrite)
    console.print("[green]Strict cross-section challenge frozen.[/green]")
    counts = result["counts"]
    console.print(f"All strict test questions: {counts['all_strict_test']}")
    console.print(
        f"Cross-model complete: {counts['cross_model_complete']}"
    )
    console.print(f"Cross-model unseen: {counts['cross_model_unseen']}")
    console.print(f"Granite extension: {counts['granite_extension']}")
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'qasper_cross_section_challenge'}"
    )



def qasper_challenge_finalize_command(
    config_path: Path, *, decisions_path: Path, overwrite: bool
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = finalize_cross_section_challenge(
        config, decisions_path=decisions_path, overwrite=overwrite
    )
    counts = result["counts"]
    console.print("[green]Cross-section challenge review finalised.[/green]")
    console.print(f"Reviewed: {counts['reviewed']}")
    console.print(f"Accepted: {counts['accepted']}")
    console.print(f"Accepted cross-model: {counts['accepted_cross_model']}")
    console.print(f"Accepted Granite extension: {counts['accepted_granite_extension']}")


def challenge_analyse_command(
    config_path: Path,
    *,
    accepted_queries_path: Path,
    model: EmbeddingModel,
    retrieval_unit_sizes: tuple[int, ...],
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = analyse_cross_section_challenge(
        config,
        accepted_queries_path=accepted_queries_path,
        model=model,
        chunk_sizes=retrieval_unit_sizes,
        overwrite=overwrite,
    )
    console.print("[green]Cross-section challenge analysis complete.[/green]")
    console.print(f"Model: {model.value}")
    console.print(f"Retrieval-unit sizes: {result['chunk_sizes']}")
    console.print(f"Comparisons: {result['comparison_count']}")

def qasper_audit_command(config_path: Path, *, overwrite: bool) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = audit_qasper_collection(config, overwrite=overwrite)
    console.print("[green]Full QASPER evidence audit complete.[/green]")
    console.print(f"Prepared papers: {result['total_documents']}")
    console.print(f"Usable questions: {result['classified_query_count']}")
    console.print(
        "Strict cross-section questions: "
        f"{result['strict_cross_section_query_count']}"
    )
    console.print(
        "Eligible new cross-model strict candidates: "
        f"{result['eligible_new_cross_model_strict_candidate_count']}"
    )
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'qasper_collection_audit'}"
    )


def retrieval_unit_size_command(
    config_path: Path,
    *,
    model: EmbeddingModel,
    retrieval_unit_sizes: tuple[int, ...],
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = analyse_retrieval_unit_size(
        config,
        model=model,
        chunk_sizes=retrieval_unit_sizes,
        overwrite=overwrite,
    )
    console.print("[green]Retrieval-unit-size analysis complete.[/green]")
    console.print(f"Model: {model.value}")
    console.print(f"Retrieval-unit sizes: {result['chunk_sizes']}")
    console.print(f"Queries: {result['query_count']}")
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'retrieval_unit_size' / model.value}"
    )
