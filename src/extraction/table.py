"""Table extraction from Docling layout records.

TableExtractor reads the table records produced by LayoutExtractor (which
carry the raw Docling item dict) and reconstructs each one into a pandas
DataFrame using the cell-grid approach from the notebook.

Usage:
    context = ExtractionContext(file_path=path)
    LayoutExtractor().extract(context)          # must run first
    records = TableExtractor().extract(context)

Each ExtractedRecord returned has:
    record_type = "table"
    page        = 1-indexed page number
    bbox        = Docling BOTTOMLEFT bbox dict (from layout record)
    content     = {
        "dataframe": pd.DataFrame,
        "markdown":  str,
        "csv":       str,
        "json":      str,    # JSON array of row objects
        "raw_docling": dict, # original Docling table dict (data.table_cells etc.)
    }
    raw         = "table"
"""

import pandas as pd

from ..core.exceptions import ExtractionError
from .base import BaseExtractor, ExtractedRecord, ExtractionContext


def reconstruct_table(table_content: dict) -> pd.DataFrame:
    """Reconstruct a pandas DataFrame from Docling's table content dict.

    Each cell in data.table_cells carries its grid position via
    start_row_offset_idx / start_col_offset_idx. Cells flagged as
    column_header determine the DataFrame column names.

    Args:
        table_content: Raw Docling table item dict (content field of a
                       table ExtractedRecord from LayoutExtractor).

    Returns:
        Reconstructed pandas DataFrame.
    """
    cells = table_content["data"]["table_cells"]

    num_rows = max(c["end_row_offset_idx"] for c in cells)
    num_cols = max(c["end_col_offset_idx"] for c in cells)

    grid = [[""] * num_cols for _ in range(num_rows)]
    header_rows: set[int] = set()

    for cell in cells:
        r = cell["start_row_offset_idx"]
        c = cell["start_col_offset_idx"]
        grid[r][c] = cell["text"]
        if cell.get("column_header"):
            header_rows.add(r)

    if header_rows:
        header_row_idx = max(header_rows)
        columns = grid[header_row_idx]
        data_rows = [grid[i] for i in range(num_rows) if i not in header_rows]
    else:
        columns = [f"col_{i}" for i in range(num_cols)]
        data_rows = grid

    return pd.DataFrame(data_rows, columns=columns)


class TableExtractor(BaseExtractor):
    """Reconstructs tables from Docling layout records into DataFrames.

    Requires context.layout to be populated (LayoutExtractor must run first).
    """

    def extract(self, context: ExtractionContext) -> list[ExtractedRecord]:
        """Reconstruct all table records from context.layout into DataFrames.

        Args:
            context: Must have context.layout populated by LayoutExtractor.

        Returns:
            List of ExtractedRecord with record_type "table".

        Raises:
            ExtractionError: If context.layout is None or reconstruction fails.
        """
        if context.layout is None:
            raise ExtractionError(
                "TableExtractor requires context.layout. "
                "Run LayoutExtractor first."
            )

        records: list[ExtractedRecord] = []

        for layout_rec in context.layout.by_type("table"):
            rec = self._build_record(layout_rec)
            if rec is not None:
                records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_record(self, layout_rec: ExtractedRecord) -> ExtractedRecord | None:
        table_dict = layout_rec.content
        try:
            df = reconstruct_table(table_dict)
        except Exception:
            return None  # skip malformed tables

        try:
            content = {
                "dataframe": df,
                "markdown": df.to_markdown(index=False),
                "csv": df.to_csv(index=False),
                "json": df.to_json(orient="records"),
                "raw_docling": table_dict,
            }
        except Exception:
            content = {
                "dataframe": df,
                "raw_docling": table_dict,
            }

        return ExtractedRecord(
            record_type="table",
            page=layout_rec.page,
            bbox=layout_rec.bbox,
            content=content,
            raw="table",
        )
