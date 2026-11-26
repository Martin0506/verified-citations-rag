"""Ask the policy documents a question.

    python -m vcrag "How long do I have to submit an expense report?"     # uses Claude
    python -m vcrag --demo                                               # offline, no API key

`--demo` replays a scripted model that first fabricates a quote, so you can see
the server reject it, send feedback, and accept the corrected answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .answer import ask
from .corpus import BM25, load_corpus
from .llm import ClaudeLLM, ScriptedLLM

DATA = Path(__file__).resolve().parents[2] / "data"

DEMO_QUESTION = "How long do I have to submit an expense report, and who handles questions?"
DEMO_RESPONSES = [
    {"insufficient_context": False, "claims": [
        {"text": "Expense reports must be submitted within 30 calendar days.",
         "citations": [{"chunk_id": "expense_policy#1",
                        "quote": "Expense reports must be submitted within 30 calendar days after the expense"}]},
        {"text": "Late reports are rejected automatically.",
         "citations": [{"chunk_id": "expense_policy#1", "quote": "late reports are rejected automatically"}]},
        {"text": "Questions go to the finance desk at [EMAIL_1].",
         "citations": [{"chunk_id": "expense_policy#1",
                        "quote": "Questions about a specific report go to the finance desk at [EMAIL_1]"}]},
    ]},
    {"insufficient_context": False, "claims": [
        {"text": "Expense reports must be submitted within 30 calendar days.",
         "citations": [{"chunk_id": "expense_policy#1",
                        "quote": "Expense reports must be submitted within 30 calendar days after the expense"}]},
        {"text": "Late reports are reimbursed only with written approval from the finance director.",
         "citations": [{"chunk_id": "expense_policy#1",
                        "quote": "reimbursed only with written approval from the finance director"}]},
        {"text": "Questions go to the finance desk at [EMAIL_1].",
         "citations": [{"chunk_id": "expense_policy#1",
                        "quote": "Questions about a specific report go to the finance desk at [EMAIL_1]"}]},
    ]},
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)

    index = BM25(load_corpus(args.data))
    if args.demo:
        llm, question = ScriptedLLM([dict(r) for r in DEMO_RESPONSES]), DEMO_QUESTION
    elif args.question:
        llm, question = ClaudeLLM(), args.question
    else:
        ap.error("give a question or --demo")

    ans = ask(question, index, llm)
    print(f"Q: {question}\n")
    for n, checks in enumerate(ans.history, 1):
        print(f"attempt {n}:")
        for c in checks:
            mark = "ok " if c.ok else "REJ"
            print(f"  [{mark}] claim {c.claim} -> {c.chunk_id}: {c.reason}")
    print(f"\nstatus: {ans.status} | attempts: {ans.attempts} | sources: {', '.join(ans.sources)}\n")
    print(ans.render())
    if args.demo:
        print("\n(The model never saw the email address: it only saw [EMAIL_1].)")
        sent = llm.prompts[0][1]
        assert "finanzas@" not in sent
