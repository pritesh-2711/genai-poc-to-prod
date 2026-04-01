"""Extraction pipeline for documents.

Pipeline order:
    1. LayoutExtractor  — Docling layout analysis (must run first)
    2. TextExtractor    — text/latex records from layout
    3. TableExtractor   — structured tables via PDFPlumber
    4. ImageExtractor   — PNG image bytes via Docling picture export

Example (standalone):
    from src.extraction import LayoutExtractor, TextExtractor, TableExtractor, ImageExtractor
    from src.extraction import ExtractionContext

    context = ExtractionContext(file_path="/path/to/doc.pdf")

    LayoutExtractor().extract(context)          # populates context.layout
    texts  = TextExtractor().extract(context)
    tables = TableExtractor().extract(context)
    images = ImageExtractor().extract(context)
"""

from .base import BaseExtractor, ExtractionContext, ExtractedRecord, LayoutResult
from .image import ImageExtractor
from .layout import LayoutExtractor
from .table import TableExtractor
from .text import TextExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionContext",
    "ExtractedRecord",
    "LayoutResult",
    "LayoutExtractor",
    "TextExtractor",
    "TableExtractor",
    "ImageExtractor",
]
