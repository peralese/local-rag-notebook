import csv
from pathlib import Path
from typing import Dict, List


def parse_csv_tsv(path: Path) -> List[Dict]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        try:
            headers = next(reader)
        except StopIteration:
            return rows
        headers = [h.strip() for h in headers]
        for idx, cells in enumerate(reader):
            # pad / trim row to header length
            if len(cells) < len(headers):
                cells = cells + [""] * (len(headers) - len(cells))
            if len(cells) > len(headers):
                cells = cells[: len(headers)]
            kv = "; ".join(f"{h}: {c}" for h, c in zip(headers, cells))
            rows.append(
                {
                    "doc_id": str(path),
                    "heading_path": f"{path.name} > Table",
                    "page_no": None,
                    "text": f"Row {idx}: {kv}",
                    "meta": {"file_path": str(path), "table": path.name},
                }
            )
    return rows
