"""Base abstractions for the extraction pipeline.

Extraction order enforced by callers:
    1. LayoutExtractor  — single Docling pass; discovers all elements
    2. TextExtractor    — reads text/latex/url records from layout
    3. TableExtractor   — reconstructs tables from Docling table_cells
    4. ImageExtractor   — crops image regions via PyMuPDF + Docling bboxes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractedRecord:
    """A single content unit extracted from a document page.

    record_type: "text" | "table" | "image" | "url" | "latex"
    page:        1-indexed page number (None if unknown)
    bbox:        {"l", "t", "r", "b"} in Docling BOTTOMLEFT coordinates
    content:     type-specific payload —
                   text/url/latex → str
                   table          → {"dataframe", "markdown", "csv", "json", "raw_docling"}
                   image          → {"png_bytes", "classification", "caption", "raw_docling"}
    raw:         original raw string or Docling label before transformation
    """

    record_type: str
    page: Optional[int]
    bbox: Optional[dict]
    content: Any
    raw: Optional[str] = None


@dataclass
class LayoutResult:
    """All records from one Docling pass over a document.

    source:  absolute path to the source file
    records: all ExtractedRecord objects in document order
    """

    source: str
    records: list[ExtractedRecord] = field(default_factory=list)

    def by_type(self, record_type: str) -> list[ExtractedRecord]:
        """Return only records of the given type."""
        return [r for r in self.records if r.record_type == record_type]


@dataclass
class ExtractionContext:
    """Carries shared state through the extraction pipeline.

    file_path: absolute path to the document being processed
    layout:    populated by LayoutExtractor.extract(); required by all
               downstream extractors (Text, Table, Image)
    """

    file_path: str
    layout: Optional[LayoutResult] = None


class BaseExtractor(ABC):
    """Abstract base for all extractors."""

    @abstractmethod
    def extract(self, context: ExtractionContext) -> list[ExtractedRecord]:
        """Run extraction and return content records.

        Args:
            context: Carries the file path and any previously computed layout.

        Returns:
            List of ExtractedRecord objects produced by this extractor.
        """
