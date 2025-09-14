from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None


@dataclass
class TableData:
    columns: List[str]
    rows: List[List[str]]
    source_path: str
    title: str


def _read_delimited(path: Path, delimiter: Optional[str] = None, encoding: str = "utf-8") -> TableData:
    with path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if delimiter:
            dialect = csv.excel
            dialect.delimiter = delimiter  # type: ignore[attr-defined]
        else:
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
            except Exception:
                dialect = csv.excel
                dialect.delimiter = ","  # type: ignore[attr-defined]

        reader = csv.reader(f, dialect)
        try:
            columns = next(reader)
        except StopIteration:
            return TableData(columns=[], rows=[], source_path=str(path), title=path.name)
        rows = [[("" if c is None else str(c)) for c in r] for r in reader]
        coln = len(columns)
        rows = [(r + [""] * (coln - len(r)))[:coln] for r in rows]
        return TableData(columns=[str(c) for c in columns], rows=rows, source_path=str(path), title=path.name)


def read_csv(path: str | Path, *, delimiter: Optional[str] = None, encoding: str = "utf-8") -> TableData:
    return _read_delimited(Path(path), delimiter=delimiter, encoding=encoding)


def read_tsv(path: str | Path, *, encoding: str = "utf-8") -> TableData:
    return _read_delimited(Path(path), delimiter="\t", encoding=encoding)


def read_pdf_tables(path: str | Path, max_pages: Optional[int] = None) -> List[TableData]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed; cannot extract tables from PDF.")
    out: List[TableData] = []
    p = Path(path)
    with pdfplumber.open(p) as pdf:
        pages = pdf.pages[:max_pages] if isinstance(max_pages, int) else pdf.pages
        for i, page in enumerate(pages, start=1):
            tables = page.extract_tables() or []
            for j, tbl in enumerate(tables, start=1):
                if not tbl:
                    continue
                columns = [str(c) for c in (tbl[0] or [])]
                body = tbl[1:] if len(tbl) > 1 else []
                if not any(columns):
                    ncols = max((len(r) for r in tbl), default=0)
                    columns = [f"col_{k+1}" for k in range(ncols)]
                    body = tbl
                coln = len(columns)
                rows = []
                for r in body:
                    rr = ["" if c is None else str(c) for c in r]
                    rr = (rr + [""] * (coln - len(rr)))[:coln]
                    rows.append(rr)
                out.append(
                    TableData(
                        columns=columns,
                        rows=rows,
                        source_path=str(p),
                        title=f"{p.name} (page {i}, table {j})",
                    )
                )
    return out


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]], max_width: int = 80) -> str:
    def clamp(s: str) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= max_width else s[: max_width - 1] + "…"

    head = "| " + " | ".join(clamp(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clamp(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def chunk_table(
    table: TableData,
    *,
    rows_per_chunk: int = 20,
    overlap: int = 0,
    max_markdown_cell: int = 80,
) -> List[Dict[str, Any]]:
    rows = table.rows
    out: List[Dict[str, Any]] = []
    if rows_per_chunk <= 0:
        rows_per_chunk = len(rows) or 1
    step = max(1, rows_per_chunk - overlap)
    for start in range(0, len(rows), step):
        end = min(len(rows), start + rows_per_chunk)
        slice_rows = rows[start:end]
        md = _markdown_table(table.columns, slice_rows, max_width=max_markdown_cell)
        out.append(
            {
                "text": md,
                "meta": {
                    "kind": "table",
                    "columns": table.columns,
                    "row_from": start + 1,
                    "row_to": end,
                    "total_rows": len(rows),
                    "source_path": table.source_path,
                    "title": table.title,
                },
            }
        )
        if end == len(rows):
            break
    return out


def extract_csv_tsv_chunks(
    path: str | Path,
    *,
    rows_per_chunk: int = 20,
    overlap: int = 0,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
) -> List[Dict[str, Any]]:
    p = Path(path)
    if delimiter == "\t" or p.suffix.lower() in {".tsv", ".tab"}:
        tbl = read_tsv(p, encoding=encoding)
    else:
        tbl = read_csv(p, delimiter=delimiter, encoding=encoding)
    if not tbl.columns:
        return []
    return chunk_table(tbl, rows_per_chunk=rows_per_chunk, overlap=overlap)


def extract_pdf_table_chunks(
    path: str | Path,
    *,
    rows_per_chunk: int = 20,
    overlap: int = 0,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for tbl in read_pdf_tables(path, max_pages=max_pages):
        chunks.extend(chunk_table(tbl, rows_per_chunk=rows_per_chunk, overlap=overlap))
    return chunks

