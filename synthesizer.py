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
from typing import Any, Dict, List, Tuple, Iterable

from llm.factory import make_llm  # requires llm/base.py, llm/ollama.py, llm/factory.py

# ---------------------------
# Utilities
# ---------------------------

CITATION_TAG_RE = re.compile(r"\[C(\d+)\]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Strip common wrappers around model JSON (markdown fences, leading junk)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_BOM_RE = re.compile(r"^\ufeff")


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first top-level JSON object from a string.
    Tolerates markdown fences, BOM, and trailing/leading noise.
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string")

    cleaned = _BOM_RE.sub("", text)
    cleaned = _CODE_FENCE_RE.sub("", cleaned)

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")

    depth = 0
    esc = False
    in_str = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "\\" and in_str:
            esc = not esc
            continue
        if ch == '"' and not esc:
            in_str = not in_str
        esc = False if ch != "\\" else esc

        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Unbalanced JSON braces in model output.")


def _json_loads_strict(txt: str) -> Dict[str, Any]:
    """
    Parse STRICT JSON. If full-string parse fails, locate the first full object.
    """
    try:
        return json.loads(_CODE_FENCE_RE.sub("", _BOM_RE.sub("", txt)))
    except Exception:
        return _extract_first_json_object(txt)


# -------- New: diversity + near-dup helpers --------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def pack_context(
    chunks: List[Dict[str, Any]],
    max_chars: int = 24_000,
    per_source_quota: int = 3,
    near_dup_jaccard: float = 0.90,
    compare_last_n: int = 3,
) -> List[Dict[str, str]]:
    """
    Pack retrieved chunks into a compact list for prompting with:
      - Source diversity via per_source_quota (limit per file/title)
      - Near-duplicate filtering using Jaccard token overlap
      - Character budget (max_chars)

    Each packed item gets an ID C1, C2, ...
    Input chunk keys (best effort): text, title/file_name, path/uri/source.
    """
    out: List[Dict[str, str]] = []
    used = 0
    per_src_count: dict[str, int] = {}
    kept_token_history: List[List[str]] = []  # parallel to 'out', token lists for last-N compare

    def _src_key(ch: Dict[str, Any]) -> str:
        return (ch.get("path") or ch.get("uri") or ch.get("source") or ch.get("title") or "").lower()

    for i, ch in enumerate(chunks, start=1):
        text = (ch.get("text") or ch.get("content") or "").strip()
        if not text:
            continue

        src = _src_key(ch)
        cnt = per_src_count.get(src, 0)
        if per_source_quota > 0 and cnt >= per_source_quota:
            # already have enough from this source; skip to diversify
            continue

        toks = _tokens(text)
        # Compare to the last-N kept chunks (cheap dedup)
        is_dup = False
        if near_dup_jaccard is not None and near_dup_jaccard > 0 and kept_token_history:
            for past in kept_token_history[-compare_last_n:] if compare_last_n > 0 else kept_token_history:
                if _jaccard(toks, past) >= near_dup_jaccard:
                    is_dup = True
                    break
        if is_dup:
            continue

        # budget check
        add = len(text)
        if used and used + add > max_chars:
            break

        item = {
            "id": f"C{len(out) + 1}",
            "title": ch.get("title") or ch.get("file_name") or ch.get("source") or f"Source {i}",
            "uri_or_path": ch.get("path") or ch.get("uri") or ch.get("source") or "",
            "text": text,
        }
        out.append(item)
        kept_token_history.append(toks)
        per_src_count[src] = cnt + 1
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


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s and s.strip()]


def validate_citations(parsed: Dict[str, Any], strict: bool) -> Tuple[bool, str]:
    """
    Validates that:
      1) Every citation tag in the answer exists in the citations array.
      2) If strict=True, every sentence contains at least one citation tag.
    Returns (ok, reason). `reason` highlights the first failure succinctly.
    """
    ans = (parsed.get("answer_markdown") or "").strip()
    cits = parsed.get("citations") or []
    ids = {str(c.get("id")) for c in cits if c.get("id")}

    # All tags in answer must be present in citations array
    tags_in_text = {f"C{m}" for m in CITATION_TAG_RE.findall(ans)}
    missing = tags_in_text - ids
    if missing:
        return False, f"answer contains citation tags not in citations array: {sorted(missing)}"

    if strict and ans:
        for s in _sentences(ans):
            if not CITATION_TAG_RE.search(s):
                preview = (s[:80] + "…") if len(s) > 80 else s
                return False, f"strict mode: sentence lacks citation tag: '{preview}'"
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
    # New packing behavior provides diversity + near-dup filtering
    packed = pack_context(
        retrieved,
        max_chars=max_context_chars,
        per_source_quota=3,
        near_dup_jaccard=0.90,
        compare_last_n=3,
    )
    if not packed:
        return {"abstain": True, "why": "no usable context provided", "snippets": []}

    messages = build_messages(query, packed)
    parsed = _call_local_llm(messages, backend=backend, model=model, endpoint=endpoint)

    ok, reason = validate_citations(parsed, strict=strict_citations)
    if not ok:
        return {
            "abstain": True,
            "why": f"citation validation failed: {reason}",
            "snippets": packed[:3],
        }

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

