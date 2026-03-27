from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import arxiv
from PyPDF2 import PdfReader

from . import config
from .schemas import FetchError, PaperMeta

logger = logging.getLogger(__name__)


def _normalize_id(raw: str) -> str:
    """Extract bare arXiv ID from a URL or ID string."""
    raw = raw.strip().rstrip("/")
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", raw)
    if m:
        return m.group(1)
    raise FetchError(f"Cannot parse arXiv ID from: {raw!r}")


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyPDF2."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def fetch_paper(arxiv_id: str, force: bool = False) -> PaperMeta:
    """
    Download a paper from arXiv by ID.

    Writes:
      papers/{arxiv_id}.txt       — full text (or abstract fallback)
      papers/{arxiv_id}_meta.json — metadata

    Returns PaperMeta.
    """
    arxiv_id = _normalize_id(arxiv_id)
    text_path = config.PAPERS_DIR / f"{arxiv_id}.txt"
    meta_path = config.PAPERS_DIR / f"{arxiv_id}_meta.json"

    if not force and text_path.exists() and meta_path.exists():
        logger.info("Paper %s already fetched, skipping (use --force to re-fetch)", arxiv_id)
        with open(meta_path) as f:
            return PaperMeta.from_dict(json.load(f))

    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        raise FetchError(f"No paper found for arXiv ID: {arxiv_id}")

    result = results[0]
    meta = PaperMeta(
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[a.name for a in result.authors],
        abstract=result.summary,
        year=result.published.year if result.published else 0,
        categories=result.categories,
    )

    # Try to download PDF and extract text
    full_text = ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = result.download_pdf(dirpath=tmpdir)
            full_text = _extract_text_from_pdf(Path(pdf_path))
    except Exception as exc:
        logger.warning("PDF text extraction failed for %s: %s. Falling back to abstract.", arxiv_id, exc)

    if not full_text.strip():
        logger.info("Using abstract as paper text for %s", arxiv_id)
        full_text = f"Title: {meta.title}\n\nAbstract:\n{meta.abstract}"

    text_path.write_text(full_text, encoding="utf-8")
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    logger.info("Fetched paper %s: %s", arxiv_id, meta.title)
    return meta
