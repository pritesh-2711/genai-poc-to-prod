"""Text extraction — reads text and latex records from a LayoutResult.

TextExtractor is the second stage of the pipeline. It does not re-parse the
document; it reads the text-type and latex-type records that LayoutExtractor
already produced.

Usage:
    context = ExtractionContext(file_path=path, layout=layout_result)
    records = TextExtractor().extract(context)
"""

from ..core.exceptions import ExtractionError
from .base import BaseExtractor, ExtractedRecord, ExtractionContext


class TextExtractor(BaseExtractor):
    """Returns all text and latex records from the layout result.

    Requires context.layout to be populated (i.e. LayoutExtractor must run
    first).  Text records are returned as-is; no additional parsing is done
    here so this extractor is lightweight and dependency-free.
    """

    def extract(self, context: ExtractionContext) -> list[ExtractedRecord]:
        """Return text and latex records from the layout result.

        Args:
            context: Must have context.layout populated by LayoutExtractor.

        Returns:
            List of ExtractedRecord with record_type "text" or "latex".

        Raises:
            ExtractionError: If context.layout is None.
        """
        if context.layout is None:
            raise ExtractionError(
                "TextExtractor requires context.layout. "
                "Run LayoutExtractor first and set context.layout."
            )

        return [
            r for r in context.layout.records
            if r.record_type in ("text", "latex")
        ]
