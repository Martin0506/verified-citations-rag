"""Keep personal data away from the model.

Emails, phone numbers and ID numbers are replaced with placeholders such as
[EMAIL_1] before any text is sent. The mapping stays on the server and is used
to restore the values in the final answer shown to an authorized user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"(?:\+57\s?)?3\d{2}[\s-]?\d{3}[\s-]?\d{4}\b")),
    ("ID", re.compile(r"\b\d{7,10}\b")),
]


@dataclass
class Redactor:
    mapping: dict[str, str] = field(default_factory=dict)
    _reverse: dict[str, str] = field(default_factory=dict)

    def redact(self, text: str) -> str:
        for label, rx in PATTERNS:
            text = rx.sub(lambda m, label=label: self._token(label, m.group(0)), text)
        return text

    def restore(self, text: str) -> str:
        for token, value in self.mapping.items():
            text = text.replace(token, value)
        return text

    def _token(self, label: str, value: str) -> str:
        if value in self._reverse:
            return self._reverse[value]
        n = sum(1 for t in self.mapping if t.startswith(f"[{label}_")) + 1
        token = f"[{label}_{n}]"
        self.mapping[token] = value
        self._reverse[value] = token
        return token
