# synthesizer.py
"""
Local-only, grounded LLM synthesis for the Local App Query Tool.

- Uses the llm/ adapter (Ollama by default) to call a local model (e.g., llama3.1:8b)
- Answers ONLY from provided context chunks with inline citations [C1], [C2], ...
- Will ABSTAIN if support is insufficient or citation checks fail
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from llm.factory import make_llm  # requires llm/base.py, llm/ollama.py, llm/factory.py

# ---------------------------
# Utilities
# ---------------------------

CITATION_TAG_RE = re.compile(r"\[C(\d+)\]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first top-level JSON object from a string.
    Avoids relying on regex recursion; uses a simple brace stack.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    # If we get here, braces were unbalanced
    raise ValueError("Unbalanced JSON braces in model output.")

def _json_loads_strict(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        return _extract_first_json_object(txt)

def pack_context(
    chunks: List[Dict[str, Any]],
    max_chars: int = 24_000
) -> List[Dict[str, str]]:
    """
    Packs retrieved chunks into a compact list for prompting.
    Each packed item gets an ID C1, C2, ...
    Expected chunk keys (best effort): text, title/file_name, path/uri/source.
    """
    out: List[Dict[str, str]] = []
    used = 0
    for i, ch in enumerate(chunks, start=1):
        text = (ch.get("text") or ch.get("content") or "").strip()
        if not text:
            continue
        item = {
            "id": f"C{i}",
            "title": ch.get("title") or ch.get("file_name") or f"Source {i}",
            "uri_or_path": ch.get("path") or ch.get("uri") or ch.get("source") or "",
            "text": text,
        }
        add = len(text)
        if used and used + add > max_chars:
            break
        out.append(item)
        used += add
    return out

def build_messages(query: str, packed: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Builds messages suitable for Llama 3.1 running in Ollama with format='json'.
    """
    system = (
        "You are a careful analyst. Use ONLY the provided CONTEXT to answer. "
        "If the answer is not fully supported by the CONTEXT, you MUST abstain. "
        "Return STRICT JSON ONLY that matches the schema. "
        "Every factual sentence in 'answer_markdown' must include at least one citation tag like [C1]. "
        "Do not include any text outside JSON."
    )

    user = f"""QUESTION:
{query}

CONTEXT (each item MUST be cited if used):
{json.dumps({"chunks": packed}, ensure_ascii=False)}

RESPONSE JSON SCHEMA:
{{
  "answer_markdown": string,      // Markdown with inline citations like [C1]
  "citations": [                  // Only include sources actually cited in answer_markdown
    {{"id": "C1", "title": string, "uri_or_path": string, "quote": string}}
  ],
  "support_coverage": number,     // 0.0..1.0 model self-estimate that claims are fully supported
  "abstain": boolean,
  "why": string                   // If abstain=true: short reason
}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def validate_citations(parsed: Dict[str, Any], strict: bool) -> Tuple[bool, str]:
    ans = (parsed.get("answer_markdown") or "").strip()
    cits = parsed.get("citations") or []
    ids = {str(c.get("id")) for c in cits if c.get("id")}

    # All tags in answer must be present in citations array
    tags = {f"C{m}" for m in CITATION_TAG_RE.findall(ans)}
    if not tags.issubset(ids):
        return False, "answer contains citation tags not present in citations array"

    if strict and ans:
        sentences = [s.strip() for s in SENT_SPLIT_RE.split(ans) if s.strip()]
        for s in sentences:
            if not CITATION_TAG_RE.search(s):
                return False, "strict mode: found a sentence without a citation tag"
    return True, ""

def decide_abstain(parsed: Dict[str, Any], avg_sim: float, threshold: float) -> Tuple[bool, str]:
    cov = float(parsed.get("support_coverage") or 0.0)
    blended = 0.5 * cov + 0.5 * max(0.0, min(1.0, float(avg_sim or 0.0)))
    if blended < threshold:
        return True, f"insufficient support (blended={blended:.2f} < threshold={threshold:.2f})"
    if parsed.get("abstain", False):
        return True, parsed.get("why", "model abstained")
    return False, ""

def _call_local_llm(
    messages: List[Dict[str, str]],
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    endpoint: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Calls the local LLM adapter and parses STRICT JSON from the response.
    """
    llm = make_llm(backend=backend, model=model, endpoint=endpoint, offline=True)
    raw = llm.chat_json(messages, temperature=temperature, max_tokens=max_tokens)
    return _json_loads_strict(raw)

# ---------------------------
# Public entry point
# ---------------------------

def synthesize_answer(
    query: str,
    retrieved: List[Dict[str, Any]],
    avg_sim: float,
    *,
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    endpoint: str = "http://localhost:11434",
    max_context_chars: int = 24_000,
    cite_n: int = 3,
    abstain_threshold: float = 0.70,
    strict_citations: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      - On success:
        {
          "abstain": False,
          "answer_markdown": "... [C1] ...",
          "citations": [{"id":"C1","title":"...","uri_or_path":"...","quote":"..."}]
        }
      - On abstain:
        {
          "abstain": True,
          "why": "reason",
          "snippets": [{"id":"C1","title":"...","uri_or_path":"...","text":"..."}]
        }
    """
    packed = pack_context(retrieved, max_chars=max_context_chars)
    if not packed:
        return {"abstain": True, "why": "no usable context provided", "snippets": []}

    messages = build_messages(query, packed)
    parsed = _call_local_llm(messages, backend=backend, model=model, endpoint=endpoint)

    ok, reason = validate_citations(parsed, strict=strict_citations)
    if not ok:
        return {"abstain": True, "why": f"citation validation failed: {reason}", "snippets": packed[:3]}

    # Trim citations to top-N unique by id, preserving order
    if cite_n and parsed.get("citations"):
        seen, trimmed = set(), []
        for c in parsed["citations"]:
            cid = str(c.get("id"))
            if cid and cid not in seen:
                trimmed.append(c)
                seen.add(cid)
            if len(trimmed) >= cite_n:
                break
        parsed["citations"] = trimmed

    do_abstain, why = decide_abstain(parsed, avg_sim=avg_sim, threshold=abstain_threshold)
    if do_abstain:
        return {"abstain": True, "why": why or parsed.get("why", ""), "snippets": packed[:3]}

    return {
        "abstain": False,
        "answer_markdown": parsed.get("answer_markdown", "").strip(),
        "citations": parsed.get("citations", []),
    }
