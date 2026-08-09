from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.tfidf_backend import NotFittedError, TfidfEmbedding

_CORPUS = [
    "The cat sat on the cat mat in the sunny afternoon.",
    "The cat and the mat are favorite things for a cat owner.",
    "The stock market fell sharply amid inflation concerns.",
    "Investors worried about rising interest rates and inflation.",
]


def test_embed_before_fit_raises() -> None:
    model = TfidfEmbedding(svd_dimensions=4)
    with pytest.raises(NotFittedError):
        model.embed(["some text"])


def test_dimensions_before_fit_raises() -> None:
    model = TfidfEmbedding(svd_dimensions=4)
    with pytest.raises(NotFittedError):
        _ = model.dimensions


def test_fit_and_embed_shape() -> None:
    model = TfidfEmbedding(svd_dimensions=2).fit(_CORPUS)
    vectors = model.embed(_CORPUS)
    assert vectors.shape == (len(_CORPUS), model.dimensions)


def test_embeddings_are_normalized_by_default() -> None:
    model = TfidfEmbedding(svd_dimensions=2, normalize=True).fit(_CORPUS)
    vectors = model.embed(_CORPUS)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_similar_texts_more_similar_than_dissimilar() -> None:
    """The real signal this backend needs to provide: two texts about the
    same topic should have higher cosine similarity than two texts about
    different topics — verified directly rather than assumed.
    """
    model = TfidfEmbedding(svd_dimensions=3).fit(_CORPUS)
    vectors = model.embed(_CORPUS)

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    cats_similarity = cos_sim(vectors[0], vectors[1])
    cats_vs_market_similarity = cos_sim(vectors[0], vectors[2])
    assert cats_similarity > cats_vs_market_similarity


def test_empty_texts_returns_empty_array() -> None:
    model = TfidfEmbedding(svd_dimensions=2).fit(_CORPUS)
    result = model.embed([])
    assert result.shape[0] == 0


def test_small_corpus_clamps_svd_dimensions_gracefully() -> None:
    tiny_corpus = ["only one document here"]
    model = TfidfEmbedding(svd_dimensions=256).fit(tiny_corpus)
    vectors = model.embed(tiny_corpus)
    assert vectors.shape[0] == 1
    assert model.dimensions == vectors.shape[1]
