from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from .clean import normalize_text


def parse_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = normalize_text(txt)
        sections.append(
            {
                "doc_id": str(path),
                "heading_path": f"{path.name} > Page {i}",
                "page_no": i,
                "text": txt,
                "meta": {"file_path": str(path)},
            }
        )
    return sections
