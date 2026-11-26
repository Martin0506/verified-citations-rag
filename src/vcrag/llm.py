"""LLM port and adapters.

The pipeline only depends on `LLM.generate(system, user, schema) -> dict`.
`ClaudeLLM` calls the Anthropic API with structured outputs, so the response is
always JSON matching the schema. `ScriptedLLM` returns canned responses, which
keeps tests and the offline demo deterministic and free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def generate(self, system: str, user: str, schema: dict) -> dict: ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ClaudeLLM:
    """Claude via the official SDK, with JSON-schema structured outputs.

    Server-side refusal fallbacks are enabled (`fallbacks="default"`): if the
    model declines on a safety category, the API retries on a fallback model in
    the same call. Remove the `betas`/`fallbacks` arguments to opt out.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        import anthropic  # imported lazily so tests run without the SDK configured

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.usage = Usage()

    def generate(self, system: str, user: str, schema: dict) -> dict:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        return json.loads(text)


@dataclass
class ScriptedLLM:
    """Returns responses in order. Each item is a dict or a function of (system, user)."""

    responses: list[dict | Callable[[str, str], dict]]
    prompts: list[tuple[str, str]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def generate(self, system: str, user: str, schema: dict) -> dict:
        self.prompts.append((system, user))
        self.usage.calls += 1
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        r = self.responses.pop(0)
        return r(system, user) if callable(r) else r
