from pathlib import Path

import pytest

from vcrag.answer import ask
from vcrag.corpus import BM25, Chunk, load_corpus
from vcrag.llm import ScriptedLLM
from vcrag.redact import Redactor
from vcrag.verify import check_citations

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def index():
    return BM25(load_corpus(DATA))


def claim(text, chunk_id, quote):
    return {"text": text, "citations": [{"chunk_id": chunk_id, "quote": quote}]}


CHUNK = Chunk("doc#1", "doc", "Employees may work remotely up to two days per week.")


def test_verbatim_quote_from_retrieved_chunk_is_accepted():
    [c] = check_citations([claim("x", "doc#1", "work remotely up to two days")], [CHUNK])
    assert c.ok


def test_fabricated_quote_is_rejected():
    [c] = check_citations([claim("x", "doc#1", "work remotely up to five days")], [CHUNK])
    assert not c.ok and "does not appear" in c.reason


def test_citation_to_a_chunk_that_was_not_retrieved_is_rejected():
    [c] = check_citations([claim("x", "other#3", "work remotely up to two days")], [CHUNK])
    assert not c.ok and "not retrieved" in c.reason


def test_too_short_quotes_and_uncited_claims_are_rejected():
    checks = check_citations([claim("x", "doc#1", "two days"), {"text": "y", "citations": []}], [CHUNK])
    assert [c.ok for c in checks] == [False, False]


def test_quote_matching_ignores_case_and_whitespace():
    [c] = check_citations([claim("x", "doc#1", "WORK  remotely\nup to two days.")], [CHUNK])
    assert c.ok


def test_bm25_ranks_the_relevant_policy_first(index):
    top, _ = index.search("business class international flights", 1)[0]
    assert top.doc == "travel_policy"


def test_redactor_hides_and_restores_personal_data():
    r = Redactor()
    text = "Write to dpo@condor-logistica.example or call 310 555 0142, ID 1020304050."
    hidden = r.redact(text)
    assert "dpo@" not in hidden and "310 555" not in hidden and "1020304050" not in hidden
    assert r.restore(hidden) == text


def test_retry_loop_fixes_a_rejected_citation(index):
    bad = {"insufficient_context": False, "claims": [
        claim("Business class is always allowed.", "travel_policy#1", "business class is always allowed")]}
    good = {"insufficient_context": False, "claims": [
        claim("Business class only on long international flights.", "travel_policy#1",
              "Business class is allowed only for international flights longer than six hours")]}
    llm = ScriptedLLM([bad, good])
    ans = ask("Can I fly business class?", index, llm)
    assert ans.status == "answered" and ans.attempts == 2
    assert "failed server verification" in llm.prompts[1][1]


def test_unverifiable_claims_are_dropped_and_flagged(index):
    out = {"insufficient_context": False, "claims": [
        claim("Business class is allowed on long international flights.", "travel_policy#1",
              "Business class is allowed only for international flights longer than six hours"),
        claim("The CEO approves all trips.", "travel_policy#1", "the CEO approves all trips")]}
    ans = ask("Can I fly business class?", index, ScriptedLLM([out, out]))
    assert ans.status == "partial" and len(ans.claims) == 1 and len(ans.dropped) == 1


def test_no_verified_claim_means_human_review(index):
    out = {"insufficient_context": False, "claims": [claim("Made up.", "x#1", "this text is invented here")]}
    ans = ask("Can I fly business class?", index, ScriptedLLM([out, out]))
    assert ans.status == "needs_review" and ans.claims == []


def test_personal_data_never_reaches_the_model_but_answer_restores_it(index):
    def respond(system, user):
        assert "finanzas@condor-logistica.example" not in user
        return {"insufficient_context": False, "claims": [
            claim("Write to [EMAIL_1].", "expense_policy#1",
                  "Questions about a specific report go to the finance desk at [EMAIL_1]")]}
    ans = ask("Who answers questions about an expense report?", index, ScriptedLLM([respond]))
    assert ans.status == "answered"
    assert "finanzas@condor-logistica.example" in ans.claims[0]["text"]


def test_insufficient_context_is_reported_without_claims(index):
    ans = ask("What is the parental leave policy?", index,
              ScriptedLLM([{"insufficient_context": True, "claims": []}]))
    assert ans.status == "no_context"
