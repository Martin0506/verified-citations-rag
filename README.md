# Verified-Citations RAG

Retrieval-augmented generation where **the server, not the model, decides whether a citation is real**, and personal data never reaches the model.

This is a **clean-room reference implementation** of the governed-AI pattern I used in production copilots at MYQ (Management & Quality), on Claude through Azure AI Foundry. It contains **no proprietary code and no client data**. The documents in `data/` are policies of a fictional company.

## Why

A model that "cites its sources" can still invent the quote, cite a document it never saw, or paraphrase into something the source doesn't say. In regulated settings, an answer is only useful if every sentence can be traced to a real fragment. So the rule here is: **no fragment, no citation; no valid citation, no claim.**

## How it works

```
question ─► BM25 retrieve ─► redact PII ─► Claude (JSON schema) ─► verify every citation
                                                ▲                         │
                                                └── feedback: which ──────┤ some failed
                                                    citations failed      │
                                                                          ▼
                                        answered │ partial (bad claims dropped) │ needs human review
```

1. **Retrieve** the top chunks with BM25. Each chunk has a stable id such as `expense_policy#1`.
2. **Redact** emails, phone numbers and ID numbers into placeholders (`[EMAIL_1]`). The mapping stays on the server.
3. **Generate** with structured outputs: Claude must return claims, each with a `chunk_id` and a verbatim `quote`.
4. **Verify** on the server: the chunk must be one that was retrieved for this question, and the quote must appear in it (ignoring case and whitespace, at least 12 characters).
5. **Feedback loop:** if any citation fails, the model gets the exact list of failures and one more attempt.
6. **Decide:** verified claims are shown with their sources and placeholders are restored. Claims that still fail are dropped and reported. If nothing survives, the question goes to a person.

## Offline demo (no API key)

`--demo` replays a scripted model whose first answer invents a rule that is not in the policy:

```
$ python -m vcrag --demo
Q: How long do I have to submit an expense report, and who handles questions?

attempt 1:
  [ok ] claim 0 -> expense_policy#1: verified
  [REJ] claim 1 -> expense_policy#1: quote does not appear in the chunk
  [ok ] claim 2 -> expense_policy#1: verified
attempt 2:
  [ok ] claim 0 -> expense_policy#1: verified
  [ok ] claim 1 -> expense_policy#1: verified
  [ok ] claim 2 -> expense_policy#1: verified

status: answered | attempts: 2 | sources: travel_policy#1, expense_policy#1

- Expense reports must be submitted within 30 calendar days. [expense_policy#1]
- Late reports are reimbursed only with written approval from the finance director. [expense_policy#1]
- Questions go to the finance desk at finanzas@condor-logistica.example. [expense_policy#1]

(The model never saw the email address: it only saw [EMAIL_1].)
```

## With Claude

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...        # or `ant auth login`
python -m vcrag "Can I fly business class to a client in Mexico City?"
```

`ClaudeLLM` uses the official `anthropic` SDK with JSON-schema structured outputs (`output_config.format`), so the response always parses. It enables server-side refusal fallbacks and records token usage per call. The pipeline depends only on a small `LLM` protocol, so swapping providers or using Claude on Azure AI Foundry is a new adapter.

## Tests

```bash
pytest     # 12 tests, no network
```

They cover: verbatim quotes accepted; fabricated quotes, unretrieved chunks, too-short quotes and uncited claims rejected; the retry loop; partial answers; human review; PII never reaching the model while the final answer restores it; and "insufficient context".

## Limits (on purpose)

- Redaction uses patterns for emails, phones and ID numbers. Names would need a named-entity model; this demo leaves them out.
- Retrieval is BM25 to keep the demo dependency-free. Production used embeddings in pgvector; the verification step does not depend on how chunks were retrieved.

## Stack

Python 3.12 · Anthropic SDK (Claude, structured outputs) · pytest
