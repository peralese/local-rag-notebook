def classify(query: str) -> str:
    ql = query.lower()
    if any(w in ql for w in ["list", "show all", "enumerate"]):
        return "list"
    if " vs " in ql or "compare" in ql or "difference" in ql:
        return "compare"
    if any(w in ql for w in ["sum", "average", "mean"]):
        return "compute"
    return "fact"
