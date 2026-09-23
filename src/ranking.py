"""Rank only contractors that have already passed the existing hard filters."""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.filters import city_category_candidates, get_rejection_reasons
from src.models import SearchQuery


def contractor_text(row: pd.Series) -> str:
    """Use only descriptive catalogue fields, never IDs or operational data."""
    return "\n".join(
        str(row[field]).replace("|", ", ").strip()
        for field in ("description", "categories", "event_formats", "languages")
        if pd.notna(row[field]) and str(row[field]).strip()
    )


def query_text(query: SearchQuery) -> str:
    """Preferences are the semantic signal; use context only when absent."""
    preferences = (query.preferences or "").strip()
    return preferences or f"{query.category}; {query.event_format}"


def tfidf_scores(text: str, documents: list[str]) -> np.ndarray:
    """Deterministic offline cosine similarity, including empty vocabulary."""
    if not documents:
        return np.zeros(0, dtype=float)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    corpus = [text, *documents]
    if not any(vectorizer.build_analyzer()(item) for item in corpus):
        return np.zeros(len(documents), dtype=float)
    vectors = vectorizer.fit_transform(corpus)
    return np.clip(cosine_similarity(vectors[:1], vectors[1:])[0], 0.0, 1.0)


def openai_scores(
    text: str, documents: list[str], api_key: str,
) -> np.ndarray:
    """Fetch one embedding batch; errors are handled at the provider boundary."""
    from openai import OpenAI

    inputs = [text, *documents]
    with OpenAI(api_key=api_key, timeout=10.0, max_retries=0) as client:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=inputs,
            encoding_format="float",
        )
    # Match vectors to inputs explicitly, even if response items are unordered.
    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(len(inputs))):
        raise ValueError("Incomplete embedding response")
    vectors = np.asarray([item.embedding for item in items], dtype=float)
    if (
        vectors.ndim != 2
        or vectors.shape[1] == 0
        or not np.isfinite(vectors).all()
        or np.any(np.linalg.norm(vectors, axis=1) == 0)
    ):
        raise ValueError("Invalid embedding vectors")
    return np.clip(cosine_similarity(vectors[:1], vectors[1:])[0], 0.0, 1.0)


def semantic_scores(
    candidates: pd.DataFrame,
    query: SearchQuery,
    *,
    use_openai: bool = True,
) -> tuple[np.ndarray, str]:
    """Prefer embeddings; fall back as a whole batch on any provider failure."""
    text = query_text(query)
    documents = [contractor_text(row) for _, row in candidates.iterrows()]
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if use_openai and api_key and documents:
        try:
            return openai_scores(text, documents, api_key), "openai"
        except Exception:
            # Provider outages, timeouts and malformed responses must not stop
            # recommendations. Do not log exceptions that might contain secrets.
            pass
    return tfidf_scores(text, documents), "tfidf"


def budget_score(price_from_kzt: float, budget_kzt: float) -> float:
    """A broad comfort band, not a cheapest-first or price-as-quality metric."""
    if not np.isfinite(budget_kzt) or budget_kzt <= 0:
        raise ValueError("budget_kzt must be positive and finite")
    if not np.isfinite(price_from_kzt) or price_from_kzt < 0:
        raise ValueError("price_from_kzt must be non-negative and finite")
    if price_from_kzt > budget_kzt:
        raise ValueError("Over-budget contractor must be filtered out")
    ratio = price_from_kzt / budget_kzt
    # Starting prices at 50-85% leave room for a final quote. Cheap offers still
    # score strongly; price alone is neither a quality signal nor a guarantee.
    if ratio < 0.5:
        return 0.8 + 0.2 * ratio / 0.5
    if ratio <= 0.85:
        return 1.0
    return float(1.0 - 0.2 * (ratio - 0.85) / 0.15)


def duration_score(max_hours: float, duration_hours: float | None) -> float:
    """Reward some spare capacity; NaN means presence duration is irrelevant."""
    if duration_hours is None or pd.isna(max_hours):
        return 1.0
    if max_hours < duration_hours:
        raise ValueError("Insufficient duration must be filtered out")
    # Exact coverage is already a good fit; 25% spare time gives full score.
    spare_ratio = (max_hours - duration_hours) / duration_hours
    return float(0.8 + 0.2 * min(spare_ratio / 0.25, 1.0))


def rank_candidates(
    candidates: pd.DataFrame,
    query: SearchQuery,
    *,
    use_openai: bool = True,
) -> pd.DataFrame:
    """Return up to three eligible contractors with scores and original fields.

    Call city_category_candidates(), then filter_candidates(), before this
    function. Invalid input fails closed using existing eligibility rules;
    this module never loads, expands, or relaxes the catalogue.
    Set use_openai=False for offline tests, regardless of environment keys.
    """
    if not np.isfinite(query.budget_kzt) or query.budget_kzt <= 0:
        raise ValueError("budget_kzt must be positive and finite")
    if query.duration_hours is not None and (
        not np.isfinite(query.duration_hours) or query.duration_hours <= 0
    ):
        raise ValueError("duration_hours must be positive and finite")
    ranked = candidates.copy()
    if ranked.empty:
        for column in ("semantic_score", "budget_score", "duration_score", "final_score"):
            ranked[column] = pd.Series(index=ranked.index, dtype=float)
        ranked["semantic_provider"] = pd.Series(index=ranked.index, dtype=str)
        return ranked

    if len(city_category_candidates(ranked, query)) != len(ranked) or any(
        get_rejection_reasons(row, query) for _, row in ranked.iterrows()
    ):
        raise ValueError("rank_candidates requires only hard-filtered candidates")

    ranked["budget_score"] = [
        budget_score(price, query.budget_kzt) for price in ranked["price_from_kzt"]
    ]
    ranked["duration_score"] = [
        duration_score(hours, query.duration_hours) for hours in ranked["max_hours"]
    ]
    scores, provider = semantic_scores(ranked, query, use_openai=use_openai)
    ranked["semantic_score"] = scores
    ranked["semantic_provider"] = provider
    ranked["final_score"] = (
        0.65 * ranked["semantic_score"]
        + 0.25 * ranked["budget_score"]
        + 0.10 * ranked["duration_score"]
    )
    # Stable single-key passes guarantee lexicographic order with mergesort.
    for column, ascending in (
        ("id", True), ("price_from_kzt", True), ("final_score", False),
    ):
        ranked = ranked.sort_values(column, ascending=ascending, kind="mergesort")
    return ranked.copy()

