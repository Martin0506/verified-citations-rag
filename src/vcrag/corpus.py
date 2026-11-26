"""Load documents, split them into citable chunks, and retrieve with BM25."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("""a al con de del el en es la las lo los o para por que se su un una y
the a an and are as at be by for from has in is it of on or that the to was were will with""".split())


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc>#<n>", stable and human readable
    doc: str
    text: str


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("áéíóúñ", "aeioun"))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


def load_corpus(folder: str | Path, max_chars: int = 700) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        paras = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        buf, n = "", 0
        for p in paras:
            if buf and len(buf) + len(p) > max_chars:
                n += 1
                chunks.append(Chunk(f"{path.stem}#{n}", path.stem, buf))
                buf = ""
            buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            n += 1
            chunks.append(Chunk(f"{path.stem}#{n}", path.stem, buf))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for c, tf, ln in zip(self.chunks, self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
