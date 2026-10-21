from pathlib import Path
from textwrap import dedent

from sclc.config import load_config
from sclc.data.qasa import QASAItem, _parse_qasa_html, _normalise_arxiv_id
from sclc.data.relevance import prepare_queries


def _config(tmp_path: Path):
    path = tmp_path / "qasa.yaml"
    path.write_text(
        dedent(
            """
            project:
              seed: 42
            paths:
              processed_dir: data/qasa/processed
              profile_dir: data/qasa/profiles
              subset_dir: data/qasa/subsets
              retrieval_unit_dir: data/qasa/retrieval_units
              encoding_dir: outputs/qasa/encodings
              ranking_dir: outputs/qasa/rankings
              evaluation_dir: outputs/qasa/evaluation
              analysis_dir: outputs/qasa/analysis
              hf_cache_dir: cache/huggingface
            dataset:
              repo_id: lgresearch/QASA
              splits: [test]
              qasa_paper_cache_dir: data/qasa/papers
            document:
              include_title: true
              include_abstract: true
              abstract_heading: Abstract
              hierarchy_separator: " ::: "
              heading_separator: "\\n\\n"
              paragraph_separator: "\\n\\n"
              section_separator: "\\n\\n"
              remove_reference_sections: true
              reference_headings: [references, bibliography, works cited, reference]
            chunking:
              canonical_tokenizer: ibm-granite/granite-embedding-311m-multilingual-r2
              supported_chunk_sizes: [128, 256, 512]
            models:
              granite:
                model_id: ibm-granite/granite-embedding-311m-multilingual-r2
                max_document_tokens: 32768
              jina:
                model_id: jinaai/jina-embeddings-v3-hf
                max_document_tokens: 8192
                passage_adapter: retrieval_passage
                query_adapter: retrieval_query
            sampling:
              core_documents: 10000
              granite_extended_documents: 0
              minimum_usable_questions: 1
              length_bins: 3
            evaluation:
              confirmatory_split: test
              require_query_types: false
            """
        ),
        encoding="utf-8",
    )
    return load_config(path)


def test_normalise_arxiv_id_strips_version_suffix():
    assert _normalise_arxiv_id("2301.00122v3") == "2301.00122"
    assert _normalise_arxiv_id("https://arxiv.org/abs/2301.00122v3") == "2301.00122"


def test_partial_evidence_question_is_excluded_from_retrieval(tmp_path: Path):
    html = """
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>Matched evidence.</p></div>
      </section>
    </body></html>
    """
    document = _parse_qasa_html(
        paper_id="test",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="test",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=("Matched evidence.", "Missing evidence."),
            )
        ],
    )
    query = document.queries[0]
    assert not query.has_usable_textual_evidence
    assert prepare_queries(document, "core") == []


def test_partial_evidence_is_not_accepted_when_only_one_item_aligns(tmp_path: Path):
    """Guard the same rule through the public document summary property."""
    html = """
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Intro</h2>
        <div class='ltx_para' id='S1.p1'><p>Alpha evidence.</p></div>
      </section>
    </body></html>
    """
    document = _parse_qasa_html(
        paper_id="test2",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="test2",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=("Alpha evidence.", "Beta evidence."),
            )
        ],
    )
    assert document.usable_question_count == 0


def test_invalid_cached_source_is_discarded_and_refetched(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    paper_id = "2301.00122"
    cache_dir = tmp_path / "papers"
    source_dir = cache_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / f"{paper_id}.v1.source").write_bytes(b"not an arxiv source")

    source = (
        b"\\documentclass{article}\n\\begin{document}\n"
        b"\\section{Introduction}\nCache-safe evidence.\n"
        b"\\end{document}\n"
    )

    class Response:
        def __init__(self, *, text: str = "", content: bytes = b""):
            self.text = text
            self.content = content

        def raise_for_status(self):
            return None

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/e-print/" in url:
            return Response(content=source)
        raise RuntimeError("html unavailable")

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])

    selected = qasa.fetch_paper_html(
        paper_id=paper_id,
        url=f"https://ar5iv.labs.arxiv.org/html/{paper_id}",
        cache_dir=cache_dir,
        evidence_texts=("Cache-safe evidence.",),
        title_hint="Test",
    )
    assert "Cache-safe evidence." in selected
    assert any("e-print" in url for url in calls)


def test_pdf_fallback_defaults_to_v1_when_version_discovery_fails(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    class Response:
        def __init__(self, *, text: str = "", content: bytes = b""):
            self.text = text
            self.content = content

        def raise_for_status(self):
            return None

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/pdf/2301.00122v1" in url:
            return Response(content=b"fake-pdf")
        raise RuntimeError("unavailable")

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [])
    monkeypatch.setattr(qasa, "_looks_like_qasa_pdf", lambda data: True)
    monkeypatch.setattr(qasa, "_pdf_evidence_score", lambda data, evidence: (1, 1, ""))
    monkeypatch.setattr(
        qasa,
        "_pdf_to_qasa_html",
        lambda data, paper_id, title_hint: (
            '<html><body><section id="S1"><h2 class="ltx_title">1 Intro</h2>'
            '<div class="ltx_para" id="S1.p0"><p>Fallback evidence.</p></div>'
            '</section></body></html>'
        ),
    )

    selected = qasa.fetch_paper_html(
        paper_id="2301.00122",
        url="https://ar5iv.labs.arxiv.org/html/2301.00122",
        cache_dir=tmp_path,
        evidence_texts=("Fallback evidence.",),
        title_hint="Test",
    )
    assert "Fallback evidence." in selected
    assert any("2301.00122v1" in url for url in calls)


def test_fuzzy_single_paragraph_requires_an_exact_lexical_anchor(tmp_path: Path):
    import sclc.data.qasa as qasa

    candidate = (
        "alpha beta gamma delta noise epsilon zeta eta theta noise "
        "iota kappa lambda mu"
    )
    evidence = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    html = f"""
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Intro</h2>
        <div class='ltx_para' id='S1.p0'><p>{candidate}</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="anchor",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="anchor",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=(evidence,),
            )
        ],
    )
    item = document.queries[0].answers[0].evidence[0]
    assert item.alignment_status == "unmatched"
    assert item.matched_paragraph_ids == []


def test_qasa_evidence_does_not_match_title_or_abstract(tmp_path: Path):
    import sclc.data.qasa as qasa

    html = """
    <html><body>
      <h1 class='ltx_title_document'>Evidence wording.</h1>
      <div class='ltx_abstract'>Evidence wording.</div>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>Body wording.</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="bodyonly",
        html=html,
        split="test",
        title_hint="Evidence wording.",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="bodyonly",
                paper_url=None,
                title="Evidence wording.",
                question="What?",
                answer="",
                evidence_texts=("Evidence wording.",),
            )
        ],
    )
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "unmatched"
    assert document.usable_question_count == 0


def test_duplicate_exact_evidence_is_ambiguous_and_not_usable(tmp_path: Path):
    import sclc.data.qasa as qasa

    html = """
    <html><body>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>Repeated evidence.</p></div>
        <div class='ltx_para' id='S1.p2'><p>Repeated evidence.</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="ambiguous",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="ambiguous",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=("Repeated evidence.",),
            )
        ],
    )
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_ambiguous"
    assert document.usable_question_count == 0


def test_html_evidence_score_is_paragraph_local():
    import sclc.data.qasa as qasa

    html = """
    <html><body>
      <section id='S1'><h2 class='ltx_title'>1 Intro</h2>
        <div class='ltx_para' id='S1.p1'><p>First half.</p></div>
        <div class='ltx_para' id='S1.p2'><p>second half.</p></div>
      </section>
    </body></html>
    """
    assert qasa._html_evidence_score(html, ["First half. second half."]) == (0, 1)


def test_pdf_evidence_score_is_block_local():
    import fitz
    import sclc.data.qasa as qasa

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox((72, 72, 520, 110), "First half.", fontsize=10)
    page.insert_textbox((72, 130, 520, 170), "second half.", fontsize=10)
    data = pdf.tobytes()
    pdf.close()
    assert qasa._pdf_evidence_score(data, ["First half. second half."])[0] == 0


def test_pdf_heading_detector_accepts_common_unnumbered_headings():
    import sclc.data.qasa as qasa

    assert qasa._is_top_level_pdf_heading("Introduction")
    assert qasa._is_top_level_pdf_heading("Methods")
    assert qasa._is_top_level_pdf_heading("Results:")
    assert not qasa._is_top_level_pdf_heading("This sentence describes the methods used.")




def test_download_qasa_redownloads_invalid_existing_file(tmp_path: Path, monkeypatch):
    import importlib.util
    import json

    script_path = Path(__file__).parents[1] / "scripts" / "download_qasa.py"
    spec = importlib.util.spec_from_file_location("download_qasa", script_path)
    downloader = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(downloader)

    output = tmp_path / "testset.json"
    output.write_text("partial download", encoding="utf-8")

    payload = {
        "0": {
            "paper_id": "2301.00122",
            "question_id": 0,
            "question": "What?",
            "evidential_info": [{"context": "Evidence."}],
            "title": "x" * 1800,
        }
    }

    class Response:
        content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self):
            return None

    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr("sys.argv", ["download_qasa.py", str(output)])
    downloader.main()
    assert json.loads(output.read_text(encoding="utf-8"))["0"]["question"] == "What?"


def test_qasa_pandoc_source_path_preserves_abstract_and_nested_list_paragraphs(tmp_path: Path, monkeypatch):
    import subprocess
    import sclc.data.qasa as qasa

    source = b"dummy"

    class Completed:
        returncode = 0
        stdout = """<html><body>
        <p>This abstract should be restored from source.</p>
        <h1>Introduction</h1>
        <p>First paragraph.</p>
        <ul><li><p>Evidence paragraph inside a list item with distinctive wording.</p></li></ul>
        <p>Third paragraph.</p>
        <h1>Method</h1>
        <p>Method paragraph.</p>
        </body></html>"""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    files = {"main.tex": r"""\documentclass{article}
\begin{document}
\begin{abstract}This abstract should be restored from source.\end{abstract}
\section{Introduction}
First paragraph.
\begin{itemize}
\item Evidence paragraph inside a list item with distinctive wording.
\end{itemize}
Third paragraph.
\section{Method}
Method paragraph.
\end{document}
"""}

    import io, tarfile
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        raw = files["main.tex"].encode("utf-8")
        info = tarfile.TarInfo("main.tex")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))

    output = qasa._source_to_qasa_html(
        archive.getvalue(), paper_id="test", title_hint="Test"
    )
    assert "This abstract should be restored from source." in output
    assert "Evidence paragraph inside a list item with distinctive wording." in output
    assert output.count('class="ltx_para"') == 4


def test_qasa_fetch_tries_ar5iv_org_alternate(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    valid = """<html><head><meta name=\"qasa-source\" content=\"1\"></head><body><section id='S1'><h2 class='ltx_title'>1 Intro</h2>
    <div class='ltx_para' id='S1.p1'><p>Evidence paragraph.</p></div></section></body></html>"""

    class Response:
        def __init__(self, text: str):
            self.text = text
            self.content = text.encode()

        def raise_for_status(self):
            return None

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "ar5iv.labs.arxiv.org" in url:
            return Response("not usable")
        if "ar5iv.org" in url:
            return Response(valid)
        raise AssertionError(url)

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    selected = qasa.fetch_paper_html(
        paper_id="1234.5678",
        url=None,
        cache_dir=tmp_path,
        evidence_texts=("Evidence paragraph.",),
        title_hint="Test",
    )
    assert "Evidence paragraph." in selected
    assert any("ar5iv.labs.arxiv.org" in url for url in calls)
    assert any("ar5iv.org" in url for url in calls)


def test_evidence_alignment_frame_exposes_matched_ids_and_question_status(tmp_path: Path):
    import sclc.data.qasa as qasa
    from sclc.data.io import evidence_alignment_frame

    html = """<html><body>
      <section id='S1'><h2 class='ltx_title'>1 Intro</h2>
        <div class='ltx_para' id='S1.p1'><p>Evidence paragraph.</p></div>
      </section>
    </body></html>"""
    document = qasa._parse_qasa_html(
        paper_id="frame-test",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="frame-test",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=("Evidence paragraph.",),
            )
        ],
    )
    frame = evidence_alignment_frame([document])
    row = frame.iloc[0]
    assert row["matched_paragraph_ids"] == "frame-test:paragraph:00001"
    assert bool(row["question_usable"])
