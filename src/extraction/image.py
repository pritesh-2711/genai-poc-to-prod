"""Image extraction and summarisation.

Two-stage design:

    Stage 1 — extract() [required, fast]
        Reads image records from context.layout (produced by LayoutExtractor).
        Crops each image region from the PDF using PyMuPDF (fitz) based on
        the Docling BOTTOMLEFT bbox, and attaches the PNG bytes.

        Returns one ExtractedRecord per image with:
            record_type = "image"
            page        = 1-indexed page number
            bbox        = Docling BOTTOMLEFT bbox dict
            content     = {
                "png_bytes":      bytes,
                "classification": dict | None,  # Docling picture classifier output
                "caption":        str  | None,
                "raw_docling":    dict,          # original Docling picture dict
            }

    Stage 2 — process_images() [optional, expensive — requires OpenAI client]
        Filters images by classifier confidence / type, then calls GPT-4o
        vision to produce a structured summary for each relevant image.
        Returns a list of summary dicts (page, type, description, …).

Usage:
    context = ExtractionContext(file_path=path)
    LayoutExtractor().extract(context)                  # must run first

    extractor = ImageExtractor()
    image_records = extractor.extract(context)          # crop + metadata

    # Optional vision summarisation:
    from openai import OpenAI
    summaries = extractor.process_images(context, OpenAI())
"""

import base64
import json
from typing import Optional

import fitz  # PyMuPDF

from ..core.exceptions import ExtractionError
from .base import BaseExtractor, ExtractedRecord, ExtractionContext

# ---------------------------------------------------------------------------
# Classification filter constants
# ---------------------------------------------------------------------------

PROCESS_TYPES = {
    "bar_chart", "pie_chart", "line_chart", "scatter_plot", "box_plot",
    "photograph", "full_page_image", "flow_chart",
}

MIN_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Module-level helpers (ported from notebook)
# ---------------------------------------------------------------------------

def get_bbox_from_content(content: dict) -> Optional[dict]:
    """Extract bbox from a raw Docling picture item dict."""
    prov = content.get("prov", [])
    if prov:
        return prov[0].get("bbox")
    return None


def should_process_image(image_content: dict) -> tuple[bool, str]:
    """Return (should_summarise, top_class) based on Docling classification.

    Args:
        image_content: Raw Docling picture item dict (ExtractedRecord.content
                       from a layout image record, or content["raw_docling"]).

    Returns:
        (True, class_name) if the image passes confidence and type filters,
        (False, class_name) otherwise.
    """
    predictions = (
        image_content
        .get("meta", {})
        .get("classification", {})
        .get("predictions", [])
    )
    if not predictions:
        return False, "unknown"

    top = max(predictions, key=lambda p: p["confidence"])
    top_class = top["class_name"]
    confidence = top["confidence"]

    if confidence < MIN_CONFIDENCE:
        return False, top_class
    return top_class in PROCESS_TYPES, top_class


def crop_image_from_pdf(pdf_path: str, page_no: int, bbox: dict) -> bytes:
    """Crop an image region from a PDF page and return PNG bytes.

    Docling stores bbox with coord_origin='BOTTOMLEFT', so coordinates are
    converted to PyMuPDF's top-left origin before cropping.

    Args:
        pdf_path: Absolute path to the PDF.
        page_no:  1-indexed page number (Docling convention).
        bbox:     Dict with keys l, t, r, b in Docling BOTTOMLEFT coords.

    Returns:
        PNG bytes of the cropped region at 2× resolution.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    page_height = page.rect.height

    rect = fitz.Rect(
        bbox["l"],
        page_height - bbox["t"],
        bbox["r"],
        page_height - bbox["b"],
    )

    mat = fitz.Matrix(2, 2)   # 2× resolution for better OCR/vision quality
    clip = page.get_pixmap(matrix=mat, clip=rect)
    doc.close()
    return clip.tobytes("png")


def get_image_context(layout_records: list, page_no: int, n_chars: int = 600) -> str:
    """Return up to n_chars of text from the same page as the image.

    Args:
        layout_records: List of ExtractedRecord from context.layout.records.
        page_no:        Page to collect text from.
        n_chars:        Maximum characters to return.

    Returns:
        Concatenated text string.
    """
    texts = [
        r.content
        for r in layout_records
        if r.record_type == "text" and r.page == page_no and isinstance(r.content, str)
    ]
    return " ".join(texts)[:n_chars]


def summarize_image(
    image_bytes: bytes,
    image_type: str,
    context: str,
    client,  # openai.OpenAI
) -> dict:
    """Call GPT-4o vision to produce a structured summary of an image.

    Args:
        image_bytes: PNG bytes of the image.
        image_type:  Classifier label (e.g. "bar_chart", "photograph").
        context:     Surrounding page text for grounding.
        client:      Initialised openai.OpenAI client.

    Returns:
        Dict with keys: description, document_intent, context_link,
        key_data_points. Falls back to {"raw_response": str} on parse error.
    """
    b64 = base64.standard_b64encode(image_bytes).decode()

    prompt = f"""You are analysing an image extracted from a corporate annual report PDF.

Image type (from classifier): {image_type}

Surrounding text from the same page:
{context}

Answer the following in JSON with exactly these keys:
{{
  "description": "What is visually shown in this image?",
  "document_intent": "What is the document trying to communicate through this image?",
  "context_link": "Is this image self-contained or does it directly support the surrounding text? How?",
  "key_data_points": ["list any numbers, labels, or named entities visible in the image"]
}}

Return only the JSON object, no extra text."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw}


# ---------------------------------------------------------------------------
# ImageExtractor
# ---------------------------------------------------------------------------

class ImageExtractor(BaseExtractor):
    """Crops image regions from a PDF using Docling bboxes (via PyMuPDF).

    Stage 1 — extract(): returns one ExtractedRecord per image with PNG bytes
    and Docling classification metadata.

    Stage 2 — process_images(): filters by classifier and calls GPT-4o vision
    to produce structured summaries. Requires an OpenAI client.
    """

    # ------------------------------------------------------------------
    # Stage 1 — BaseExtractor interface
    # ------------------------------------------------------------------

    def extract(self, context: ExtractionContext) -> list[ExtractedRecord]:
        """Crop all images from the PDF and return records with PNG bytes.

        Args:
            context: Must have context.layout populated by LayoutExtractor.

        Returns:
            List of ExtractedRecord with record_type "image".

        Raises:
            ExtractionError: If context.layout is None.
        """
        if context.layout is None:
            raise ExtractionError(
                "ImageExtractor requires context.layout. "
                "Run LayoutExtractor first."
            )

        records: list[ExtractedRecord] = []

        for layout_rec in context.layout.by_type("image"):
            rec = self._crop_record(layout_rec, context.file_path)
            if rec is not None:
                records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Stage 2 — optional vision summarisation
    # ------------------------------------------------------------------

    def process_images(
        self,
        context: ExtractionContext,
        openai_client,
    ) -> list[dict]:
        """Filter and summarise images using GPT-4o vision.

        Only images that pass the classifier confidence + type filter are
        sent to the vision API. All others are skipped with a log message.

        Args:
            context:       Must have context.layout populated.
            openai_client: Initialised openai.OpenAI instance.

        Returns:
            List of summary dicts, one per processed image, with keys:
            self_ref, page, type, description, document_intent,
            context_link, key_data_points.
        """
        if context.layout is None:
            raise ExtractionError(
                "process_images requires context.layout. "
                "Run LayoutExtractor first."
            )

        image_layout_records = context.layout.by_type("image")
        print(f"Total images found: {len(image_layout_records)}")

        results: list[dict] = []

        for layout_rec in image_layout_records:
            raw_docling = layout_rec.content   # full Docling picture dict
            page_no = layout_rec.page
            bbox = get_bbox_from_content(raw_docling)

            if bbox is None:
                print(f"  Skipping page {page_no} — no bbox found")
                continue

            should_run, top_class = should_process_image(raw_docling)
            if not should_run:
                print(f"  Skipping page {page_no} — type: {top_class}")
                continue

            print(f"  Processing page {page_no} — type: {top_class}")

            image_bytes = crop_image_from_pdf(context.file_path, page_no, bbox)
            ctx_text = get_image_context(context.layout.records, page_no)
            summary = summarize_image(image_bytes, top_class, ctx_text, openai_client)

            results.append({
                "self_ref": raw_docling.get("self_ref"),
                "page": page_no,
                "type": top_class,
                **summary,
            })

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _crop_record(
        self,
        layout_rec: ExtractedRecord,
        pdf_path: str,
    ) -> Optional[ExtractedRecord]:
        """Crop one image record. Returns None if bbox is missing."""
        raw_docling = layout_rec.content
        bbox = layout_rec.bbox or get_bbox_from_content(raw_docling)
        page_no = layout_rec.page

        if bbox is None or page_no is None:
            return None

        try:
            png_bytes = crop_image_from_pdf(pdf_path, page_no, bbox)
        except Exception:
            return None

        _, classification_meta = self._parse_classification(raw_docling)

        content = {
            "png_bytes": png_bytes,
            "classification": classification_meta,
            "caption": raw_docling.get("caption_text"),
            "raw_docling": raw_docling,
        }

        return ExtractedRecord(
            record_type="image",
            page=page_no,
            bbox=bbox,
            content=content,
            raw="picture",
        )

    @staticmethod
    def _parse_classification(raw_docling: dict) -> tuple[bool, Optional[dict]]:
        """Return (should_process, classification_dict) from the Docling dict."""
        should_run, top_class = should_process_image(raw_docling)
        predictions = (
            raw_docling.get("meta", {})
            .get("classification", {})
            .get("predictions", [])
        )
        if predictions:
            return should_run, {"top_class": top_class, "predictions": predictions}
        return should_run, None
