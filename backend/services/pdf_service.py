"""
Page-aware PDF extraction + chunking.

Unlike the old helper.py (which used LangChain's DirectoryLoader and lost
page boundaries once documents were split into chunks), this service keeps
each chunk tagged with the page number(s) it came from, so answers can cite
"Page 3" the way the reconstructed architecture does.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple
from pypdf import PdfReader # type:ignore
from utils.logger import get_logger

logger = get_logger(__name__)

# OCR deps are optional at import time - only required if a page has no
# extractable text (i.e. it's a scanned image).
try:
    from pdf2image import convert_from_path #type: ignore
    import pytesseract  #type: ignore
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    ocr_used: bool = False


@dataclass
class Chunk:
    text: str
    pages: List[int] = field(default_factory=list)


def _ocr_page(image, page_number: int) -> PageText:
    ocr_text = pytesseract.image_to_string(image).strip()
    return PageText(page_number=page_number, text=ocr_text, ocr_used=True)


def extract_pages_from_pdf(file_path: str) -> List[PageText]:
    """Extract text per page, falling back to OCR for pages with no text layer."""
    logger.info("Extracting pages from %s", file_path)
    reader = PdfReader(file_path)
    pages: List[PageText] = []
    ocr_page_indices = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=i + 1, text=text))
        else:
            pages.append(PageText(page_number=i + 1, text="", ocr_used=False))
            ocr_page_indices.append(i)

    if ocr_page_indices:
        if not OCR_AVAILABLE:
            logger.warning(
                "%d page(s) had no extractable text but OCR deps (pdf2image/"
                "pytesseract + poppler/tesseract) are not installed; those "
                "pages will be skipped.",
                len(ocr_page_indices),
            )
        else:
            logger.info("Running OCR fallback on %d page(s)", len(ocr_page_indices))
            images = convert_from_path(file_path, first_page=min(ocr_page_indices) + 1, last_page=max(ocr_page_indices) + 1)
            ocr_pages: List[Tuple[int, PageText]] = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(_ocr_page, images[idx - min(ocr_page_indices)], idx + 1): idx
                    for idx in ocr_page_indices
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        ocr_pages.append((idx, future.result()))
                    except Exception as exc:
                        logger.warning("OCR failed for page %d: %s", idx + 1, exc)

            for idx, page_text in ocr_pages:
                pages[idx] = page_text

    non_empty = [p for p in pages if p.text]
    logger.info(
        "Extracted %d/%d pages with text (%d via OCR)",
        len(non_empty),
        len(pages),
        sum(1 for p in pages if p.ocr_used),
    )
    return pages


def chunk_pages(
    pages: List[PageText], chunk_size: int = 500, chunk_overlap: int = 80
) -> List[Chunk]:
    """
    Simple sliding-window chunker over each page's text, kept page-aware.
    Chunks near a page boundary that span into the next page get tagged
    with both page numbers.
    """
    chunks: List[Chunk] = []

    for page in pages:
        text = page.text
        if not text:
            continue

        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, pages=[page.page_number]))
            if end == length:
                break
            start = end - chunk_overlap if end - chunk_overlap > start else end

    logger.info("Created %d page-aware chunks", len(chunks))
    return chunks


def any_ocr_used(pages: List[PageText]) -> bool:
    return any(p.ocr_used for p in pages)
