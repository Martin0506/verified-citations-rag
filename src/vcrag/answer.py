"""Question -> retrieve -> redact -> generate -> verify -> (retry) -> answer or human review."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .corpus import BM25, Chunk
from .llm import LLM
from .redact import Redactor
from .verify import CitationCheck, check_citations, verified_claims

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                            "required": ["chunk_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
        "insufficient_context": {"type": "boolean"},
    },
    "required": ["claims", "insufficient_context"],
    "additionalProperties": False,
}

SYSTEM = """You answer questions about internal policy documents.
Use only the sources provided. Split the answer into short factual claims.
Every claim must cite at least one source: give its chunk_id and copy a quote of at least
a few words verbatim from that chunk. Do not paraphrase inside "quote".
If the sources do not answer the question, return no claims and set insufficient_context to true.
Placeholders such as [EMAIL_1] stand for personal data; keep them as they are."""


@dataclass
class Answer:
    status: str                     # "answered" | "partial" | "needs_review" | "no_context"
    claims: list[dict]
    dropped: list[dict] = field(default_factory=list)
    checks: list[CitationCheck] = field(default_factory=list)
    attempts: int = 0
    sources: list[str] = field(default_factory=list)
    history: list[list[CitationCheck]] = field(default_factory=list)  # checks per attempt

    def render(self) -> str:
        if self.status in ("needs_review", "no_context"):
            return {"needs_review": "No verifiable answer. Sent to a person for review.",
                    "no_context": "The documents do not answer this question."}[self.status]
        lines = []
        for c in self.claims:
            refs = ", ".join(sorted({x["chunk_id"] for x in c["citations"]}))
            lines.append(f"- {c['text']} [{refs}]")
        if self.dropped:
            lines.append(f"({len(self.dropped)} unverifiable claim(s) removed)")
        return "\n".join(lines)


def _sources_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<source chunk_id="{c.id}">\n{c.text}\n</source>' for c in chunks)


def ask(question: str, index: BM25, llm: LLM, k: int = 4, max_attempts: int = 2) -> Answer:
    hits = [c for c, _ in index.search(question, k)]
    if not hits:
        return Answer("no_context", [], sources=[])
    red = Redactor()
    chunks = [Chunk(c.id, c.doc, red.redact(c.text)) for c in hits]
    user = f"{_sources_block(chunks)}\n\nQuestion: {red.redact(question)}"

    out, checks, attempts, history = {"claims": []}, [], 0, []
    while attempts < max_attempts:
        attempts += 1
        out = llm.generate(SYSTEM, user, SCHEMA)
        if out.get("insufficient_context") and not out.get("claims"):
            return Answer("no_context", [], attempts=attempts, sources=[c.id for c in chunks])
        checks = check_citations(out["claims"], chunks)
        history.append(checks)
        bad = [c for c in checks if not c.ok]
        if not bad:
            break
        # Feedback loop: tell the model exactly which citations failed and why.
        feedback = "\n".join(f"- claim {c.claim}: chunk {c.chunk_id or '?'}: {c.reason}" for c in bad)
        user = (f"{user}\n\nYour previous answer:\n{json.dumps(out, ensure_ascii=False)}\n\n"
                f"These citations failed server verification:\n{feedback}\n"
                "Fix them with verbatim quotes from the listed chunks, or remove the claim.")

    kept, dropped = verified_claims(out["claims"], checks)
    for c in kept:
        c["text"] = red.restore(c["text"])
    status = "answered" if kept and not dropped else "partial" if kept else "needs_review"
    return Answer(status, kept, dropped, checks, attempts, [c.id for c in chunks], history)
