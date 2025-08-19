from pathlib import Path
import re
from typing import List, Dict
from .clean import normalize_text

def parse_md_or_txt(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)
    lines = text.splitlines()

    sections = []
    def push(title, txt):
        if txt:
            sections.append({
                "doc_id": str(path),
                "heading_path": f"{path.name} > {title}" if title else path.name,
                "page_no": None,
                "text": "\n".join(txt),
                "meta": {"file_path": str(path)}
            })

    title = ""
    buf = []
    for ln in lines:
        m = re.match(r"^\s*#+\s+(.*)", ln)
        if m:
            push(title, buf)
            title = m.group(1).strip()
            buf = []
        else:
            buf.append(ln)
    push(title, buf)

    if not sections:
        sections.append({
            "doc_id": str(path),
            "heading_path": path.name,
            "page_no": None,
            "text": text,
            "meta": {"file_path": str(path)}
        })
    return sections
