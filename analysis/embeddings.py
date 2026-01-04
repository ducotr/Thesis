# analysis/embeddings.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

from data.publication import Publication

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word2VecConfig:
    """ """

    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    workers: int = 4
    sg: int = 1  # 1 = skip-gram, 0 = CBOW
    seed: int | None = None
    epochs: int = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """ """
    if not text:
        return []
    tokens = simple_preprocess(text, deacc=True, min_len=2)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def keyword_token(kw: str) -> str | None:
    """ """
    kw = kw.strip()
    if not kw:
        return None
    return kw.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Sentence construction
# ---------------------------------------------------------------------------


def make_sentences(
    publications: List[Publication],
) -> list[list[str]]:
    """ """
    sentences: list[list[str]] = []

    for pub in publications:
        keyword_tokens = tokenize(" ".join(pub.keywords))
        keyword_tokens = [
            t for t in (keyword_token(kw) for kw in pub.keywords) if t is not None
        ]
        title_tokens = tokenize(pub.title)
        abstract_tokens = tokenize(pub.abstract)
        sentence = keyword_tokens + title_tokens + abstract_tokens
        sentences.append(sentence)

    return sentences


def train_word2vec(
    publications: List[Publication],
    config: Word2VecConfig,
) -> Word2Vec:
    """ """
    sentences = make_sentences(publications=publications)

    if not sentences:
        raise ValueError("No training sentences found. Check your publications.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=config.sg,
        seed=config.seed,
        epochs=config.epochs,
    )

    return model


def keyword_frequencies(
    publications: List[Publication],
) -> dict[str, int]:
    freqs: dict[str, int] = {}
    for pub in publications:
        for kw in pub.keywords:
            token = keyword_token(kw)
            if token is None:
                continue
            freqs[token] = freqs.get(token, 0) + 1
    return freqs
