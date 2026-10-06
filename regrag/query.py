"""Query understanding: which state(s) is the question about, and split comparisons."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

# Names, regulators, discoms and major cities that pin a question to a state.
STATE_ALIASES: dict[str, list[str]] = {
    "Gujarat": ["gujarat", "gerc", "geda", "dgvcl", "mgvcl", "pgvcl", "ugvcl", "torrent power",
                "ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar"],
    "Maharashtra": ["maharashtra", "merc", "msedcl", "mahadiscom", "mahavitaran", "mumbai", "pune",
                    "nagpur", "nashik", "best undertaking", "tata power mumbai", "adani electricity"],
    "Karnataka": ["karnataka", "kerc", "bescom", "mescom", "hescom", "gescom", "cesc mysore",
                  "bengaluru", "bangalore", "mysuru", "mysore", "hubli", "mangaluru"],
    "Tamil Nadu": ["tamil nadu", "tamilnadu", "tnerc", "tangedco", "tneb", "tnpdcl", "chennai",
                   "coimbatore", "madurai"],
    "Rajasthan": ["rajasthan", "rerc", "jvvnl", "avvnl", "jdvvnl", "jaipur", "jodhpur", "ajmer",
                  "udaipur"],
}
OTHER_STATES = ["andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "haryana",
                "himachal pradesh", "jharkhand", "kerala", "madhya pradesh", "manipur", "meghalaya", "mizoram",
                "nagaland", "odisha", "orissa", "punjab", "sikkim", "telangana", "tripura", "uttar pradesh",
                "uttarakhand", "west bengal", "delhi", "jammu", "kashmir", "ladakh", "puducherry", "chandigarh"]
CENTRAL_HINTS = ["central government", "ministry of power", "mop", "mnre", "green energy open access",
                 "geoa", "all india", "national", "centre"]
COMPARE_HINTS = re.compile(r"\b(compare|comparison|versus|vs\.?|difference|differ|both|each|all (five|5) states|across)\b", re.I)


@dataclass
class SubQuery:
    state: str | None      # None = no state detected -> search all
    text: str


def detect_states(query: str) -> list[str]:
    q = f" {query.lower()} "
    found = []
    for state, aliases in STATE_ALIASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", q) for a in aliases):
            found.append(state)
    if re.search(r"\ball (five|5) states\b|\bevery state\b|\beach state\b", q):
        found = list(config.STATES)
    return found


def unsupported_states(query: str) -> list[str]:
    """States named in the query that the corpus does not cover."""
    q = query.lower()
    return [s.title() for s in OTHER_STATES if re.search(rf"\b{re.escape(s)}\b", q)]


def out_of_scope_message(query: str) -> str | None:
    """Abstain up front when the question is only about states we have no documents for."""
    other = unsupported_states(query)
    if other and not detect_states(query):
        return (f"I can't answer this: the indexed regulations cover only {', '.join(config.STATES)} "
                f"and central rules, not {', '.join(other)}.")
    return None


def mentions_central(query: str) -> bool:
    q = query.lower()
    return any(re.search(rf"\b{re.escape(h)}\b", q) for h in CENTRAL_HINTS)


def _strip_states(query: str) -> str:
    out = query
    for aliases in STATE_ALIASES.values():
        for a in aliases:
            out = re.sub(rf"(?i)(?<![a-z]){re.escape(a)}(?![a-z])", "", out)
    # Drop connectors left dangling by the removal ("in  versus ?" -> "?"); repeat until stable.
    dangling = re.compile(r"(?i)\b(in|for|of|and|between|vs\.?|versus|compare|comparison)\s*(?=[,.?]|$|\band\b|\bvs\b|\bversus\b)")
    prev = None
    while prev != out:
        prev, out = out, dangling.sub("", out)
        out = re.sub(r"\s{2,}", " ", out)
    return re.sub(r"\s+([?,])", r"\1", out).strip(" ,")


def decompose(query: str) -> list[SubQuery]:
    """One sub-query per detected state. Single-state questions pass through unchanged."""
    states = detect_states(query)
    if not states:
        return [SubQuery(None, query)]
    if len(states) == 1:
        return [SubQuery(states[0], query)]
    core = (_strip_states(query) or query).rstrip("?. ").strip()
    # "Compare the limits" reads oddly as a single-state question: ask "What are the limits ... in Gujarat?"
    core = re.sub(r"(?i)^(compare|contrast)\s+", "What are ", core)
    return [SubQuery(s, f"{core} in {s}?") for s in states]


def is_comparison(query: str) -> bool:
    return len(detect_states(query)) > 1 or bool(COMPARE_HINTS.search(query))
