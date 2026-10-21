from pathlib import Path
from textwrap import dedent

from sclc.config import load_config
from sclc.data.qasa import QASAItem, _parse_qasa_html, iter_qasa_items


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


def test_iter_qasa_items_supports_gold_context_ids_and_text():
    raw = [
        {
            "id": "q1",
            "arxiv_id": "1707.01083",
            "question": "What is the effect?",
            "answer": "It improves performance.",
            "gold_ctxs": ["1707.01083_all_0", "1707.01083_all_1"],
            "ctxs": [
                {"id": "1707.01083_all_0", "text": "First evidence paragraph."},
                {"id": "1707.01083_all_1", "text": "Second evidence paragraph."},
            ],
        }
    ]
    items = iter_qasa_items(raw)
    assert len(items) == 1
    assert items[0].paper_id == "1707.01083"
    assert items[0].evidence_texts == (
        "First evidence paragraph.",
        "Second evidence paragraph.",
    )


def test_parse_qasa_html_preserves_top_level_sections_and_aligns_evidence(tmp_path: Path):
    html = """
    <html><head><title>Test Paper</title></head><body>
      <h1 class='ltx_title_document'>Test Paper</h1>
      <div class='ltx_abstract'>A short abstract.</div>
      <section id='S1'>
        <h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para'><p>First evidence paragraph.</p></div>
        <section id='S1.SS1'>
          <h3 class='ltx_title'>1.1 Details</h3>
          <div class='ltx_para'><p>Second evidence paragraph.</p></div>
        </section>
      </section>
      <section id='S2'>
        <h2 class='ltx_title'>2 Method</h2>
        <div class='ltx_para'><p>Another paragraph.</p></div>
      </section>
    </body></html>
    """
    config = _config(tmp_path)

    document = _parse_qasa_html(
        paper_id="1707.01083",
        html=html,
        split="test",
        title_hint="Test Paper",
        config=config,
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="1707.01083",
                paper_url=None,
                title="Test Paper",
                question="What is the effect?",
                answer="It works.",
                evidence_texts=("Second evidence paragraph.",),
            )
        ],
    )

    assert len(document.sections) == 3  # abstract + S1 + S2
    assert document.sections[1].heading == "1 Introduction"
    assert len(document.paragraphs) == 5
    assert document.queries[0].has_usable_textual_evidence
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"
    assert len(evidence.matched_paragraph_ids) == 1
    paragraph = next(
        p for p in document.paragraphs if p.paragraph_id == evidence.matched_paragraph_ids[0]
    )
    assert paragraph.section_id == document.sections[1].section_id

def test_iter_qasa_items_supports_official_qasa_v1_1_shape():
    raw = {
        "0": {
            "paper_id": "1707.01083",
            "title": "Example",
            "question_id": 0,
            "question": "What is the effect?",
            "question_section": "Introduction",
            "question_trigger_sentence": "The authors discuss the issue.",
            "question_type": "surface",
            "evidential_info": [
                {
                    "paragraph": "First evidence paragraph.",
                    "rationale": "This explains the effect."
                },
                {
                    "paragraph": "Second evidence paragraph.",
                    "rationale": "This provides supporting detail."
                }
            ],
            "composition": "True",
            "arxiv_id": "1707.01083",
            "s2orc_url": "https://example.invalid/s2orc",
            "arxiv_url": "https://arxiv.org/abs/1707.01083",
        }
    }
    items = iter_qasa_items(raw)
    assert len(items) == 1
    assert items[0].evidence_texts == (
        "First evidence paragraph.",
        "Second evidence paragraph.",
    )
    # The adapter should use the ar5iv HTML route for full-text parsing.
    assert items[0].paper_url == "https://ar5iv.labs.arxiv.org/html/1707.01083"


def test_iter_qasa_items_does_not_treat_rationale_as_source_evidence():
    raw = {
        "0": {
            "paper_id": "1707.01083",
            "question_id": 0,
            "question": "What is the effect?",
            "evidential_info": [
                {"rationale": "Generated explanation only."},
            ],
        }
    }
    items = iter_qasa_items(raw)
    assert len(items) == 1
    assert items[0].evidence_texts == ()


def test_qasa_evidence_alignment_tolerates_inline_html_whitespace(tmp_path: Path):
    html = """
    <html><head><title>Whitespace</title></head><body>
      <h1 class='ltx_title_document'>Whitespace</h1>
      <section id='S1'>
        <h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>humans.<span>However</span>, most language models...</p></div>
      </section>
    </body></html>
    """
    config = _config(tmp_path)
    document = _parse_qasa_html(
        paper_id="2210.01241",
        html=html,
        split="test",
        title_hint="Whitespace",
        config=config,
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="2210.01241",
                paper_url=None,
                title="Whitespace",
                question="Why?",
                answer="",
                evidence_texts=("humans.However, most language models...",),
            )
        ],
    )
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"
    assert len(evidence.matched_paragraph_ids) == 1


def test_qasa_evidence_alignment_maps_contiguous_multi_paragraph_span(tmp_path: Path):
    html = """
    <html><head><title>Span</title></head><body>
      <h1 class='ltx_title_document'>Span</h1>
      <section id='S1'>
        <h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>First source paragraph.</p></div>
        <div class='ltx_para' id='S1.p2'><p>Second source paragraph.</p></div>
      </section>
    </body></html>
    """
    config = _config(tmp_path)
    document = _parse_qasa_html(
        paper_id="1707.01083",
        html=html,
        split="test",
        title_hint="Span",
        config=config,
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="1707.01083",
                paper_url=None,
                title="Span",
                question="What is covered?",
                answer="",
                evidence_texts=("First source paragraph. Second source paragraph.",),
            )
        ],
    )

    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_span"
    assert len(evidence.matched_paragraph_ids) == 2


def test_parse_qasa_html_handles_official_ar5iv_paragraph_ids(tmp_path: Path):
    html = """
    <html><head><title>Official Shape</title></head><body>
      <h1 class='ltx_title_document'>Official Shape</h1>
      <div class='ltx_abstract'>Abstract.</div>
      <section id='S1'>
        <h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>First evidence paragraph.</p></div>
        <section id='S1.SS1'>
          <h3 class='ltx_title'>1.1 Details</h3>
          <div class='ltx_para' id='S1.SS1.p1'><p>Second evidence paragraph.</p></div>
        </section>
      </section>
      <section id='S2'>
        <h2 class='ltx_title'>2 Method</h2>
        <div class='ltx_para' id='S2.p1'><p>Third paragraph.</p></div>
      </section>
    </body></html>
    """
    config = _config(tmp_path)
    document = _parse_qasa_html(
        paper_id="1707.01083",
        html=html,
        split="test",
        title_hint="Official Shape",
        config=config,
        query_items=[
            QASAItem(
                question_id="q1",
                paper_id="1707.01083",
                paper_url=None,
                title="Official Shape",
                question="What is the effect?",
                answer="",
                evidence_texts=("Second evidence paragraph.",),
            )
        ],
    )
    assert len(document.paragraphs) == 5  # title + abstract + three paper paragraphs
    assert document.sections[1].heading == "1 Introduction"
    assert document.sections[2].heading == "2 Method"
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"


def test_qasa_source_fallback_works_when_current_html_is_unavailable(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    source = _source_archive(
        {
            "main.tex": r"""\documentclass{article}
\begin{document}
\section{Introduction}
Historical QASA source text is available even when ar5iv HTML is unavailable.
\end{document}
"""
        }
    )

    class Response:
        def __init__(self, *, text: str = "", content: bytes = b""):
            self.text = text
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "/e-print/" in url:
            return Response(content=source)
        raise RuntimeError("current HTML unavailable")

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])

    selected = qasa.fetch_paper_html(
        paper_id="2301.00189",
        url="https://ar5iv.labs.arxiv.org/html/2301.00189",
        cache_dir=tmp_path,
        evidence_texts=(
            "Historical QASA source text is available even when ar5iv HTML is unavailable.",
        ),
        title_hint="Test",
    )

    assert "Historical QASA source text is available" in selected
    assert 'name="qasa-source"' in selected


def test_qasa_versioned_source_selection_prefers_exact_evidence_match(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    filler = 'x' * 1100
    current = f"""<html><body><div>{filler}</div><section id='S1'><h2 class='ltx_title'>1 Intro</h2>
    <div class='ltx_para' id='S1.p1'><p>Current wording.</p></div>
    </section></body></html>"""
    old = f"""<html><body><div>{filler}</div><section id='S1'><h2 class='ltx_title'>1 Intro</h2>
    <div class='ltx_para' id='S1.p1'><p>Historical wording from QASA.</p></div>
    </section></body></html>"""

    def fake_get(url, **kwargs):
        class Response:
            def __init__(self, text, status=200):
                self.text = text
                self.status_code = status
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError("HTTP error")
        if "2210.01241v1" in url:
            return Response(old)
        if "2210.01241" in url and "/abs/" not in url:
            return Response(current)
        return Response("<html><body>v1 v2</body></html>")

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])

    selected = qasa.fetch_paper_html(
        paper_id="2210.01241",
        url="https://ar5iv.labs.arxiv.org/html/2210.01241",
        cache_dir=tmp_path,
        evidence_texts=("Historical wording from QASA.",),
    )
    assert "Historical wording from QASA." in selected



def test_qasa_pdf_fallback_runs_when_html_and_source_are_unavailable(tmp_path: Path, monkeypatch):
    import fitz
    import sclc.data.qasa as qasa

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "1 INTRODUCTION", fontsize=12)
    page.insert_textbox(
        (72, 96, 520, 170),
        "PDF-only fallback evidence is available even when HTML and source reconstruction fail.",
        fontsize=10,
    )
    pdf_bytes = pdf.tobytes()
    pdf.close()

    def fake_get(url, **kwargs):
        if "/pdf/" in url and "v1" in url:
            class Response:
                text = ""
                content = pdf_bytes
                status_code = 200
                def raise_for_status(self):
                    return None
            return Response()
        raise RuntimeError("unavailable")

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])
    monkeypatch.setattr(qasa, "_looks_like_qasa_pdf", lambda data: True)
    monkeypatch.setattr(
        qasa,
        "_pdf_to_qasa_html",
        lambda data, paper_id, title_hint: (
            "<html><body><h1 class='ltx_title_document'>Test</h1>"
            "<section id='S1'><h2 class='ltx_title'>1 INTRODUCTION</h2>"
            "<div class='ltx_para' id='S1.p1'><p>PDF-only fallback evidence is available even when HTML and source reconstruction fail.</p></div>"
            "</section></body></html>"
        ),
    )

    selected = qasa.fetch_paper_html(
        paper_id="2301.00122",
        url="https://ar5iv.labs.arxiv.org/html/2301.00122",
        cache_dir=tmp_path,
        evidence_texts=(
            "PDF-only fallback evidence is available even when HTML and source reconstruction fail.",
        ),
    )

    assert "PDF-only fallback evidence" in selected

def test_qasa_versioned_pdf_fallback_recovers_historical_source(tmp_path: Path, monkeypatch):
    import fitz
    import sclc.data.qasa as qasa

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "1 INTRODUCTION", fontsize=12)
    page.insert_textbox(
        (72, 96, 520, 180),
        "Historical wording from QASA. This evidence is present in the older arXiv version.",
        fontsize=10,
    )
    page.insert_text((72, 220), "2 METHOD", fontsize=12)
    page.insert_textbox(
        (72, 244, 520, 320),
        "Method paragraph.",
        fontsize=10,
    )
    pdf.set_metadata({"subject": "x" * 12000})
    pdf.new_page()
    pdf_bytes = pdf.tobytes()
    pdf.close()

    filler = "x" * 1100
    current = f"""<html><body><div>{filler}</div><h1 class='ltx_title_document'>Test</h1>
    <section id='S1'><h2 class='ltx_title'>1 INTRODUCTION</h2>
    <div class='ltx_para' id='S1.p1'><p>Current wording.</p></div>
    </section></body></html>"""

    class Response:
        def __init__(self, *, text: str = "", content: bytes = b""):
            self.text = text
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "/pdf/" in url and "v1" in url:
            return Response(content=pdf_bytes)
        return Response(text=current)

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])

    selected = qasa.fetch_paper_html(
        paper_id="2210.01241",
        url="https://ar5iv.labs.arxiv.org/html/2210.01241",
        cache_dir=tmp_path,
        evidence_texts=(
            "Historical wording from QASA. This evidence is present in the older arXiv version.",
        ),
    )

    assert "Historical wording from QASA" in selected
    document = qasa._parse_qasa_html(
        paper_id="2210.01241",
        html=selected,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="2210.01241",
                paper_url=None,
                title="Test",
                question="Why?",
                answer="",
                evidence_texts=(
                    "Historical wording from QASA. This evidence is present in the older arXiv version.",
                ),
            )
        ],
    )
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"


def test_qasa_pdf_reconstruction_merges_text_across_figure_interrupt(tmp_path: Path):
    import fitz
    import sclc.data.qasa as qasa

    pdf = fitz.open()
    page0 = pdf.new_page()
    page0.insert_text((72, 72), "1 Introduction", fontsize=12)
    page0.insert_textbox(
        (72, 96, 520, 180),
        "The first part of the historical paragraph continues",
        fontsize=10,
    )

    page1 = pdf.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 240, 120))
    pix.clear_with(0xFFFFFF)
    page1.insert_image((160, 72, 400, 300), pixmap=pix)
    pix = None
    page1.insert_text((180, 315), "Figure 1: An illustrative diagram.", fontsize=8)
    page1.insert_textbox(
        (72, 360, 520, 430),
        "(RL) offers a natural continuation of the same paragraph.",
        fontsize=10,
    )

    pdf.set_metadata({"subject": "x" * 12000})
    pdf_bytes = pdf.tobytes()
    pdf.close()

    html = qasa._pdf_to_qasa_html(
        pdf_bytes,
        paper_id="2210.01241",
        title_hint="Test",
    )
    assert "The first part of the historical paragraph continues (RL) offers" in html
    assert "Figure 1: An illustrative diagram." not in html

    document = qasa._parse_qasa_html(
        paper_id="2210.01241",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="2210.01241",
                paper_url=None,
                title="Test",
                question="Why?",
                answer="",
                evidence_texts=(
                    "The first part of the historical paragraph continues (RL) offers a natural continuation of the same paragraph.",
                ),
            )
        ],
    )
    evidence = document.queries[0].answers[0].evidence[0]
    assert evidence.alignment_status == "matched_unique"


def _source_archive(files: dict[str, str]) -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_qasa_source_reconstruction_preserves_tex_paragraphs_and_excludes_figures():
    import sclc.data.qasa as qasa

    source = _source_archive(
        {
            "main.tex": r"""\documentclass{article}
\begin{document}
\begin{abstract}An abstract paragraph.\end{abstract}
\section{Introduction}
The first paragraph contains \textbf{important} wording and \citep{smith2020}.

The second paragraph follows the figure.
\begin{figure}
\caption{A distracting figure.}
\end{figure}
Still part of the second paragraph after the figure.
\section{Method}
A method paragraph.
\end{document}
""",
            "refs.bib": """@article{smith2020,\n  author = {Smith, Alice and Jones, Bob},\n  year = {2020}\n}\n""",
        }
    )
    html = qasa._source_to_qasa_html(source, paper_id="test", title_hint="Test")
    assert 'id="S1.p0"' in html
    assert "A distracting figure." not in html
    assert "important" in html
    assert "(Smith et al., 2020)" in html
    assert "The second paragraph follows the figure." in html


def test_qasa_source_fallback_can_beat_pdf_and_historical_html(tmp_path: Path, monkeypatch):
    import sclc.data.qasa as qasa

    current = """<html><body><div>{}</div><h1 class='ltx_title_document'>Test</h1>
    <section id='S1'><h2 class='ltx_title'>1 INTRODUCTION</h2>
    <div class='ltx_para' id='S1.p1'><p>Current wording.</p></div></section></body></html>""".format("x" * 1200)
    source = _source_archive(
        {"main.tex": r"""\documentclass{article}
\begin{document}
\section{Introduction}
Historical wording from the TeX source is preserved as one paragraph.
\end{document}
"""}
    )

    class Response:
        def __init__(self, *, text: str = "", content: bytes = b""):
            self.text = text
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "/e-print/" in url:
            return Response(content=source)
        return Response(text=current)

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    monkeypatch.setattr(qasa, "_discover_arxiv_versions", lambda *args, **kwargs: [1])

    selected = qasa.fetch_paper_html(
        paper_id="2210.01241",
        url="https://ar5iv.labs.arxiv.org/html/2210.01241",
        cache_dir=tmp_path,
        evidence_texts=("Historical wording from the TeX source is preserved as one paragraph.",),
        title_hint="Test",
    )
    assert "Historical wording from the TeX source is preserved as one paragraph." in selected
    assert 'name="qasa-source"' in selected or 'name="qasa-source"' in selected


def test_qasa_evidence_alignment_recovers_formatting_drift_with_fuzzy_span(tmp_path: Path):
    import sclc.data.qasa as qasa

    source = (
        "The ultimate aim of language technology is to interact with humans. "
        "However, most language models are trained without direct signals of human preference, "
        "with supervised target strings serving as a crude proxy. Automated metrics offer a "
        "promising compromise. Goodhart’s Law looms: particularly in the case of imperfect "
        "metrics that use neural networks, it is easy to find nonsense samples."
    )
    evidence = (
        "The ultimate aim of language technology is to interact with humans.However, most "
        "language models are trained without direct signals of human preference,with supervised "
        "target strings serving as (a sometimes crude) proxy.Automated metrics offer a promising "
        "compromise.Goodhart’s Law222Strathern (1997) paraphrases: When a measure becomes a "
        "target, it ceases to be a good measure.looms: particularly in the case of imperfect "
        "metrics that use neural networks, it is easy to find nonsense samples."
    )
    html = f"""
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>{source}</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="2210.01241",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="2210.01241",
                paper_url=None,
                title="Test",
                question="Why?",
                answer="",
                evidence_texts=(evidence,),
            )
        ],
    )
    item = document.queries[0].answers[0].evidence[0]
    assert item.alignment_status == "matched_fuzzy_unique"
    assert item.matched_paragraph_ids == ["p1"] or len(item.matched_paragraph_ids) == 1


def test_qasa_evidence_alignment_maps_fuzzy_span_to_multiple_paragraphs(tmp_path: Path):
    import sclc.data.qasa as qasa

    first = "The first source paragraph establishes the central retrieval setting."
    second = "The second source paragraph explains the observed effect in greater detail."
    evidence = (
        "The first source paragraph establishes the central retrieval setting. "
        "Inserted citation 2022. The second source paragraph explains the observed effect "
        "in greater detail."
    )
    html = f"""
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>{first}</p></div>
        <div class='ltx_para' id='S1.p2'><p>{second}</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="1707.01083",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="1707.01083",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=(evidence,),
            )
        ],
    )
    item = document.queries[0].answers[0].evidence[0]
    assert item.alignment_status == "matched_fuzzy_span"
    assert item.matched_paragraph_ids == ["p1", "p2"] or len(item.matched_paragraph_ids) == 2


def test_qasa_fuzzy_alignment_prefers_single_paragraph_over_adjacent_window(tmp_path: Path):
    import sclc.data.qasa as qasa

    first = (
        "Gaussian blur: Producing images via gradient ascent tends to produce examples "
        "with high frequency information. A useful regularization is thus to penalize high "
        "frequency information. We implement this as a Gaussian blur step."
    )
    second = (
        "Clipping pixels with small norm: The first two regularizations suppress high "
        "amplitude and high frequency information."
    )
    evidence = (
        "Gaussian blur: Producing images via gradient ascent tends to produce examples with "
        "high frequency information. A useful regularization is thus to penalize high frequency "
        "information. We implement this as a Gaussian blur step (Nguyen et al., 2014)."
    )
    html = f"""
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>{first}</p></div>
        <div class='ltx_para' id='S1.p2'><p>{second}</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="test",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="test",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=(evidence,),
            )
        ],
    )
    item = document.queries[0].answers[0].evidence[0]
    assert item.alignment_status == "matched_fuzzy_unique"
    assert len(item.matched_paragraph_ids) == 1


def test_qasa_fuzzy_alignment_recovers_noncontiguous_extraction_fragments(tmp_path: Path):
    import sclc.data.qasa as qasa

    first = (
        "The first source paragraph establishes the central retrieval setting with enough "
        "repeated lexical material to anchor the match robustly."
    )
    middle = "This unrelated intervening figure fragment contains completely different words."
    third = (
        "The third source paragraph explains the observed effect in greater detail with "
        "enough repeated lexical material to anchor the match robustly."
    )
    evidence = f"{first} {third}"
    html = f"""
    <html><body>
      <h1 class='ltx_title_document'>Test</h1>
      <section id='S1'><h2 class='ltx_title'>1 Introduction</h2>
        <div class='ltx_para' id='S1.p1'><p>{first}</p></div>
        <div class='ltx_para' id='S1.p2'><p>{middle}</p></div>
        <div class='ltx_para' id='S1.p3'><p>{third}</p></div>
      </section>
    </body></html>
    """
    document = qasa._parse_qasa_html(
        paper_id="test",
        html=html,
        split="test",
        title_hint="Test",
        config=_config(tmp_path),
        query_items=[
            qasa.QASAItem(
                question_id="q1",
                paper_id="test",
                paper_url=None,
                title="Test",
                question="What?",
                answer="",
                evidence_texts=(evidence,),
            )
        ],
    )
    item = document.queries[0].answers[0].evidence[0]
    assert item.alignment_status == "matched_fuzzy_scattered"
    assert len(item.matched_paragraph_ids) == 2


def test_iter_qasa_items_preserves_s2orc_id_for_non_arxiv_records():
    raw = [{
        "paper_id": "paper_156",
        "s2orc_id": "247185306",
        "arxiv_url": None,
        "title": "Example",
        "question_id": 3,
        "question": "What differs?",
        "evidential_info": [{"context": "Evidence paragraph."}],
    }]
    items = iter_qasa_items(raw)
    assert len(items) == 1
    assert items[0].paper_id == "paper_156"
    assert items[0].s2orc_id == "247185306"


def test_semantic_scholar_open_access_pdf_uses_corpus_id(monkeypatch):
    import sclc.data.qasa as qasa

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"openAccessPdf": {"url": "https://example.org/paper.pdf"}}

    def fake_get(url, *, params, timeout, verify, headers):
        assert url.endswith("/paper/CorpusId:247185306")
        assert params["fields"] == "title,openAccessPdf,externalIds,url"
        return Response()

    monkeypatch.setattr(qasa.requests, "get", fake_get)
    assert qasa._semantic_scholar_open_access_pdf("247185306", timeout=30) == "https://example.org/paper.pdf"





def test_non_arxiv_cached_s2orc_pdf_is_used_before_semantic_scholar(tmp_path: Path, monkeypatch):
    import pymupdf
    import sclc.data.qasa as qasa

    pdf = pymupdf.open()
    for page_num in range(12):
        page = pdf.new_page()
        page.insert_text((72, 72), "1 Introduction", fontsize=14)
        filler = (
            "Cached non arXiv evidence paragraph with enough textual content to validate "
            "the local PDF fallback before any metadata lookup is attempted. "
            + " ".join(f"unique_{page_num}_{i}" for i in range(120))
        )
        page.insert_textbox((72, 100, 520, 740), filler, fontsize=9)
    data = pdf.tobytes()
    pdf.close()
    assert len(data) > 10_000

    cache = tmp_path / "papers"
    cache.mkdir()
    (cache / "paper_172.s2orc.pdf").write_bytes(data)

    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be required when cached S2ORC PDF exists")

    monkeypatch.setattr(qasa, "_download_response", fail_network)
    monkeypatch.setattr(qasa, "_semantic_scholar_open_access_pdf", fail_network)

    html = qasa.fetch_paper_html(
        paper_id="paper_172",
        url=None,
        cache_dir=cache,
        evidence_texts=(
            "Cached non arXiv evidence paragraph with enough textual content to validate "
            "the local PDF fallback before any metadata lookup is attempted.",
        ),
        title_hint="GPT-Critic",
        s2orc_id="251648921",
    )

    assert 'name="qasa-pdf-source"' in html
    assert "Cached non arXiv evidence paragraph" in html

