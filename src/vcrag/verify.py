"""Server-side citation verification.

The model must back every claim with a citation: a chunk id plus a verbatim
quote. The server accepts a citation only if the chunk was actually retrieved
for this question AND the quote appears in that chunk. No fragment, no citation;
no valid citation, no claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Chunk

MIN_QUOTE_CHARS = 12


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().strip(".,;:\"'«»")


@dataclass(frozen=True)
class CitationCheck:
    claim: int
    chunk_id: str
    quote: str
    ok: bool
    reason: str


def check_citations(claims: list[dict], retrieved: list[Chunk]) -> list[CitationCheck]:
    by_id = {c.id: _norm(c.text) for c in retrieved}
    out = []
    for i, claim in enumerate(claims):
        cites = claim.get("citations") or []
        if not cites:
            out.append(CitationCheck(i, "", "", False, "claim has no citation"))
        for cit in cites:
            cid, quote = cit.get("chunk_id", ""), cit.get("quote", "")
            q = _norm(quote)
            if cid not in by_id:
                reason, ok = "chunk was not retrieved for this question", False
            elif len(q) < MIN_QUOTE_CHARS:
                reason, ok = f"quote shorter than {MIN_QUOTE_CHARS} characters", False
            elif q not in by_id[cid]:
                reason, ok = "quote does not appear in the chunk", False
            else:
                reason, ok = "verified", True
            out.append(CitationCheck(i, cid, quote, ok, reason))
    return out


def verified_claims(claims: list[dict], checks: list[CitationCheck]) -> tuple[list[dict], list[dict]]:
    """Split claims into (kept with only their valid citations, dropped)."""
    kept, dropped = [], []
    for i, claim in enumerate(claims):
        good = [{"chunk_id": c.chunk_id, "quote": c.quote} for c in checks if c.claim == i and c.ok]
        (kept if good else dropped).append({**claim, "citations": good} if good else claim)
    return kept, dropped
