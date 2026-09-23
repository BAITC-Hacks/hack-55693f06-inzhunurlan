import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_contractors
from src.filters import city_category_candidates, filter_candidates
from src.models import SearchQuery
from src.ranking import (
    budget_score,
    duration_score,
    rank_candidates,
    semantic_scores,
)


QUERY = SearchQuery(
    city="Алматы",
    event_date="2026-09-23",
    event_format="свадьба",
    category="Ведущий",
    budget_kzt=1_500_000,
)


def eligible_candidates(query=QUERY):
    catalogue = load_contractors()
    city_category = city_category_candidates(catalogue, query)
    return filter_candidates(city_category, query)[0]


def test_returns_all_eligible_candidates_in_deterministic_order():
    candidates = eligible_candidates()
    ranked = rank_candidates(candidates, QUERY, use_openai=False)
    shuffled = rank_candidates(
        candidates.sample(frac=1, random_state=7), QUERY, use_openai=False
    )

    assert len(ranked) == len(candidates)
    assert ranked.id.tolist() == shuffled.id.tolist()
    assert ranked.final_score.is_monotonic_decreasing


def test_preserves_catalogue_fields_and_adds_scores():
    candidates = eligible_candidates()
    ranked = rank_candidates(candidates, QUERY, use_openai=False)

    assert set(candidates.columns).issubset(ranked.columns)
    for column in (
        "semantic_score",
        "budget_score",
        "duration_score",
        "final_score",
    ):
        assert ranked[column].between(0, 1).all()
    assert (ranked.semantic_provider == "tfidf").all()
    assert (ranked.semantic_fallback_reason == "openai_disabled").all()


def test_successful_mocked_openai_embeddings_use_openai(monkeypatch):
    candidates = eligible_candidates().iloc[:2]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Response:
        data = [
            type("Embedding", (), {"index": 0, "embedding": [1.0, 0.0]})(),
            type("Embedding", (), {"index": 1, "embedding": [1.0, 0.0]})(),
            type("Embedding", (), {"index": 2, "embedding": [0.0, 1.0]})(),
        ]

    class Embeddings:
        def create(self, **kwargs):
            return Response()

    class Client:
        embeddings = Embeddings()

        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 10.0
            assert kwargs["max_retries"] == 0

        def close(self):
            pass

    monkeypatch.setattr("openai.OpenAI", Client)
    ranked = rank_candidates(candidates, QUERY)
    assert (ranked.semantic_provider == "openai").all()
    assert (ranked.semantic_fallback_reason == "").all()


def test_missing_key_uses_tfidf_with_reason(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ranked = rank_candidates(eligible_candidates().iloc[:1], QUERY)
    assert ranked.semantic_provider.iloc[0] == "tfidf"
    assert ranked.semantic_fallback_reason.iloc[0] == "missing_api_key"


def test_openai_exception_uses_tfidf_with_safe_reason(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fail(*args, **kwargs):
        raise TimeoutError("secret-looking details must not be exposed")

    monkeypatch.setattr("src.ranking.openai_scores", fail)
    ranked = rank_candidates(eligible_candidates().iloc[:1], QUERY)
    assert ranked.semantic_provider.iloc[0] == "tfidf"
    assert ranked.semantic_fallback_reason.iloc[0] == "openai_error:TimeoutError"


def test_tie_breaks_by_price_then_id():
    candidates = eligible_candidates().iloc[:1].copy()
    candidates = pd.concat([candidates] * 4, ignore_index=True)
    candidates["id"] = ["D", "B", "A", "C"]
    candidates["price_from_kzt"] = [900_000, 800_000, 800_000, 1_000_000]

    ranked = rank_candidates(candidates, QUERY, use_openai=False)
    assert ranked.id.tolist() == ["A", "B", "D", "C"]


def test_budget_score_is_not_cheapest_first():
    assert budget_score(500_000, 1_000_000) == budget_score(850_000, 1_000_000)
    assert budget_score(100_000, 1_000_000) < budget_score(700_000, 1_000_000)


def test_duration_nan_is_neutral():
    assert duration_score(np.nan, 8) == 1.0
    assert duration_score(8, None) == 1.0


def test_empty_input_returns_score_columns():
    ranked = rank_candidates(pd.DataFrame(), QUERY, use_openai=False)
    assert ranked.empty
    assert {
        "semantic_score",
        "budget_score",
        "duration_score",
        "final_score",
        "semantic_provider",
    }.issubset(ranked.columns)


def test_ineligible_candidate_is_rejected_before_ranking():
    candidates = eligible_candidates().iloc[:1].copy()
    candidates.loc[candidates.index[0], "price_from_kzt"] = QUERY.budget_kzt + 1

    with pytest.raises(ValueError, match="hard-filtered"):
        rank_candidates(candidates, QUERY, use_openai=False)

