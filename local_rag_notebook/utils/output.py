from __future__ import annotations

import datetime
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-\s_]+", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s[:max_len].strip("-") or "query"


def infer_format(out_path: Optional[str], fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    if out_path:
        ext = Path(out_path).suffix.lower().lstrip(".")
        if ext in {"json", "md", "txt", "html", "htm"}:
            return "html" if ext in {"html", "htm"} else ext
    return "json"


def ensure_outpath(
    out_path: Optional[str], fmt: str, save_dir: Optional[str], question: str
) -> Path:
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    base_dir = Path(save_dir or "outputs")
    base_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_timestamp()}_{_slug(question)}.{ 'html' if fmt=='html' else fmt }"
    return base_dir / fname


def as_markdown(ans: Dict[str, Any]) -> str:
    q = ans.get("question", "")
    answer = ans.get("answer", "").strip()
    cites = ans.get("citations", [])
    timers = (ans.get("trace") or {}).get("timers_ms")
    lines: List[str] = []
    lines.append(f"# {q}\n")
    if answer:
        lines.append(answer)
        lines.append("")
    if cites:
        lines.append("## Citations")
        for c in cites:
            hp = (
                " > ".join(c.get("heading_path") or [])
                if isinstance(c.get("heading_path"), list)
                else (c.get("heading_path") or "")
            )
            pg = c.get("page_no") or "?"
            lines.append(f"- `{c.get('file')}` | {hp} | Page {pg}")
        lines.append("")
    if timers:
        lines.append("## Timers (ms)")
        lines.append("```json")
        lines.append(json.dumps(timers, indent=2))
        lines.append("```")
    return "\n".join(lines).strip() + "\n"


def as_text(ans: Dict[str, Any]) -> str:
    q = ans.get("question", "")
    answer = ans.get("answer", "").strip()
    cites = ans.get("citations", [])
    timers = (ans.get("trace") or {}).get("timers_ms")
    lines: List[str] = []
    lines.append(f"QUESTION: {q}")
    lines.append("")
    lines.append(answer)
    lines.append("")
    if cites:
        lines.append("CITATIONS:")
        for c in cites:
            hp = (
                " > ".join(c.get("heading_path") or [])
                if isinstance(c.get("heading_path"), list)
                else (c.get("heading_path") or "")
            )
            pg = c.get("page_no") or "?"
            lines.append(f"- {c.get('file')} | {hp} | Page {pg}")
        lines.append("")
    if timers:
        lines.append("TIMERS_MS: " + json.dumps(timers))
    return "\n".join(lines).strip() + "\n"


def as_html(ans: Dict[str, Any]) -> str:
    q = html.escape(ans.get("question", ""))
    answer = html.escape(ans.get("answer", "").strip()).replace("\n", "<br>")
    cites = ans.get("citations", [])
    timers = (ans.get("trace") or {}).get("timers_ms")

    def esc(x):
        return html.escape(str(x)) if x is not None else ""

    lines: List[str] = []
    lines.append("<!doctype html><html><head><meta charset='utf-8'>")
    lines.append(
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px} h1{font-size:1.6rem} code,pre{background:#f6f8fa;padding:2px 4px;border-radius:4px} .cites li{margin:6px 0}</style>"
    )
    lines.append("</head><body>")
    lines.append(f"<h1>{q}</h1>")
    if answer:
        lines.append(f"<div class='answer'>{answer}</div>")
    if cites:
        lines.append("<h2>Citations</h2><ul class='cites'>")
        for c in cites:
            hp = c.get("heading_path")
            if isinstance(hp, list):
                hp = " &gt; ".join([esc(h) for h in hp if h])
            else:
                hp = esc(hp or "")
            pg = esc(c.get("page_no") or "?")
            lines.append(f"<li><code>{esc(c.get('file'))}</code> | {hp} | Page {pg}</li>")
        lines.append("</ul>")
    if timers:
        lines.append("<h2>Timers (ms)</h2><pre><code>")
        lines.append(html.escape(json.dumps(timers, indent=2)))
        lines.append("</code></pre>")
    lines.append("</body></html>")
    return "\n".join(lines)


def write_output(
    question: str,
    payload: Dict[str, Any],
    out_path: Optional[str] = None,
    fmt: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Path:
    fmt2 = infer_format(out_path, fmt)
    target = ensure_outpath(out_path, fmt2, save_dir, question)
    # build a stable object
    obj = {
        "question": question,
        "answer": payload.get("answer"),
        "citations": payload.get("citations"),
        "trace": payload.get("trace"),
        "contexts": payload.get("contexts"),
    }
    if fmt2 == "json":
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt2 == "md":
        target.write_text(as_markdown(obj), encoding="utf-8")
    elif fmt2 == "txt":
        target.write_text(as_text(obj), encoding="utf-8")
    elif fmt2 == "html":
        target.write_text(as_html(obj), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt2}")
    return target
