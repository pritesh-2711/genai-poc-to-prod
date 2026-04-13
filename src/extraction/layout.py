"""Layout extraction using Docling.

LayoutExtractor is the first and only Docling pass in the pipeline.
It produces a flat list of typed records (text, table, image, url, latex)
identical in structure to the notebook's run_docling() output.

Each record dict stored in ExtractedRecord.content preserves the raw
Docling item dict so that downstream extractors (TableExtractor,
ImageExtractor) can use it directly without re-running Docling.

Usage as standalone:
    result = LayoutExtractor().extract_layout("path/to/file.pdf")
    # result.records  — list[ExtractedRecord]

Usage in pipeline:
    context = ExtractionContext(file_path="path/to/file.pdf")
    LayoutExtractor().extract(context)   # populates context.layout
"""

import re
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import logging

from ..core.exceptions import ExtractionError
from .base import BaseExtractor, ExtractedRecord, ExtractionContext, LayoutResult

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL helper (shared with downstream extractors via import)
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(
    r"""(?ix)\b(
        (?:https?://|www\.)
        [^\s<>()\[\]{}"']+
    )"""
)


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    return list({m.group(1) for m in URL_PATTERN.finditer(text)})


# ---------------------------------------------------------------------------
# LayoutExtractor
# ---------------------------------------------------------------------------

class LayoutExtractor(BaseExtractor):
    """Runs Docling on a document once and produces all typed layout records.

    The resulting records carry the raw Docling item dict in ExtractedRecord.content
    for tables and images, so that TableExtractor and ImageExtractor can
    process them without re-running Docling.
    """

    def __init__(self, do_picture_classification: bool = False) -> None:
        self._do_picture_classification = do_picture_classification

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    def extract(self, context: ExtractionContext) -> list[ExtractedRecord]:
        """Run Docling and populate context.layout.

        Args:
            context: Only context.file_path is used.

        Returns:
            List of ExtractedRecord (also stored in context.layout.records).

        Raises:
            ExtractionError: If Docling conversion fails.
        """
        records = self._run_docling(context.file_path)
        context.layout = LayoutResult(source=context.file_path, records=records)
        return records

    def extract_layout(self, file_path: str) -> LayoutResult:
        """Convenience wrapper: run and return the LayoutResult directly."""
        context = ExtractionContext(file_path=file_path)
        self.extract(context)
        return context.layout  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core — mirrors run_docling() from the notebook
    # ------------------------------------------------------------------

    def _run_docling(self, pdf_path: str) -> list[ExtractedRecord]:
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_formula_enrichment = False
            pipeline_options.do_picture_classification = self._do_picture_classification

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(pdf_path)
            doc_dict = result.document.export_to_dict()
        except Exception as exc:
            _log.exception("Docling pipeline failed for '%s'", pdf_path)
            raise ExtractionError(
                f"Docling conversion failed for '{pdf_path}': {exc}"
            ) from exc

        records: list[ExtractedRecord] = []

        for key in ("texts", "tables", "pictures"):
            for item in doc_dict.get(key, []):
                records.extend(self._parse_item(key, item))

        return records

    def _parse_item(self, key: str, item: dict) -> list[ExtractedRecord]:
        label = str(item.get("label", "")).upper()
        page, bbox = self._parse_prov(item.get("prov", []))

        out: list[ExtractedRecord] = []

        if key == "texts":
            text = item.get("text", "") or ""
            if "FORMULA" in label:
                out.append(ExtractedRecord(
                    record_type="latex",
                    page=page,
                    bbox=bbox,
                    content=item.get("latex") or text,
                    raw=text,
                ))
            else:
                out.append(ExtractedRecord(
                    record_type="text",
                    page=page,
                    bbox=bbox,
                    content=text,
                    raw=text,
                ))
                for url in extract_urls_from_text(text):
                    out.append(ExtractedRecord(
                        record_type="url",
                        page=page,
                        bbox=bbox,
                        content=url,
                        raw=text,
                    ))

        elif key == "tables":
            out.append(ExtractedRecord(
                record_type="table",
                page=page,
                bbox=bbox,
                content=item,   # full Docling dict; reconstruct_table() uses data.table_cells
                raw="table",
            ))

        elif key == "pictures":
            out.append(ExtractedRecord(
                record_type="image",
                page=page,
                bbox=bbox,
                content=item,   # full Docling dict; has prov[].bbox + meta.classification
                raw="picture",
            ))

        return out

    @staticmethod
    def _parse_prov(prov: list[dict]) -> tuple[int | None, dict | None]:
        if not prov:
            return None, None
        first = prov[0]
        page = first.get("page_no")
        raw_bbox = first.get("bbox")
        bbox = (
            {k: raw_bbox.get(k) for k in ("l", "t", "r", "b")}
            if isinstance(raw_bbox, dict)
            else None
        )
        return page, bbox
