from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import SearchQuery
from src.recommender import _count_phrase, _reason_summary, recommend_contractors


def catalogue():
    rows = [
        ("A", "Ведущий", "Алматы", 500000, "свадьба|корпоратив", "русский", 10, "2026-12-01", False),
        ("B", "Ведущий", "Алматы", 800000, "свадьба", "русский|казахский", 8, "2026-12-02", False),
        ("C", "Ведущий", "Алматы", 1200000, "свадьба", "русский", 8, "2026-12-01", False),
        ("D", "Ведущий", "Астана", 400000, "свадьба", "русский", 8, "", False),
        ("E", "Флорист", "Алматы", 200000, "свадьба", "русский", np.nan, "", True),
    ]
    result = pd.DataFrame(rows, columns=["id", "categories", "city", "price_from_kzt", "event_formats", "languages", "max_hours", "busy_dates", "synthetic"])
    result["anon_name"] = result["id"]
    result["city_imputed"] = False
    result["price_imputed"] = result["id"].eq("E")
    result["category_list"] = result["categories"].str.split("|")
    result["event_format_list"] = result["event_formats"].str.split("|")
    result["language_list"] = result["languages"].str.split("|")
    result["busy_date_set"] = result["busy_dates"].apply(lambda value: set(value.split("|")) if value else set())
    result["description"] = ""
    return result


def query(**changes):
    values = dict(city="Алматы", event_date="2026-12-01", event_format="свадьба", category="Ведущий", budget_kzt=900000)
    values.update(changes)
    return SearchQuery(**values)


def test_russian_count_phrase_edge_cases():
    forms = ("кандидат", "кандидата", "кандидатов")
    assert [_count_phrase(n, forms) for n in (1, 2, 3, 4, 5, 11, 21, 22)] == [
        "1 кандидат", "2 кандидата", "3 кандидата", "4 кандидата",
        "5 кандидатов", "11 кандидатов", "21 кандидат", "22 кандидата",
    ]


def test_reason_summary_uses_correct_contractor_pluralization():
    assert _reason_summary({"over_budget": 1}) == "1 подрядчик имеет стартовую цену выше бюджета"
    assert _reason_summary({"over_budget": 2}) == "2 подрядчика имеют стартовую цену выше бюджета"
    assert _reason_summary({"over_budget": 5}) == "5 подрядчиков имеют стартовую цену выше бюджета"


def test_matched_returns_maximum_three_cards_and_explanations():
    response = recommend_contractors(query(), catalogue(), use_openai=False)
    assert response["status"] == "MATCHED"
    assert 1 <= len(response["results"]) <= 3
    assert all(card["explanation"] for card in response["results"])
    assert len({card["explanation"] for card in response["results"]}) == len(response["results"])


def test_category_not_found_does_not_rank_or_call_openai():
    with patch("src.recommender.rank_candidates", side_effect=AssertionError), patch("src.recommender.load_contractors", side_effect=AssertionError):
        response = recommend_contractors(query(category="Декоратор"), catalogue(), use_openai=True)
    assert response["status"] == "CATEGORY_NOT_FOUND"
    assert response["results"] == []


def test_no_eligible_candidates_has_rejection_diagnostics():
    response = recommend_contractors(query(event_format="конференция"), catalogue(), use_openai=False)
    assert response["status"] == "NO_ELIGIBLE_CANDIDATES"
    assert response["results"] == []
    assert response["diagnostics"]["rejection_reason_counts"]["wrong_format"] == 3


def test_busy_over_budget_format_language_and_duration_never_appear():
    response = recommend_contractors(query(language="английский", duration_hours=9), catalogue(), use_openai=False)
    assert response["status"] == "NO_ELIGIBLE_CANDIDATES"
    reasons = response["diagnostics"]["rejection_reason_counts"]
    assert reasons["busy"] == 2
    assert reasons["over_budget"] == 1
    assert reasons["wrong_language"] == 3
    assert reasons["duration_too_long"] == 2


def test_nan_max_hours_remains_eligible_for_non_presence_service():
    response = recommend_contractors(query(category="Флорист"), catalogue(), use_openai=False)
    assert response["status"] == "MATCHED"
    assert response["results"][0]["id"] == "E"
    assert pd.isna(response["results"][0]["max_hours"])


def test_fewer_than_three_is_explicit_and_metadata_is_preserved():
    response = recommend_contractors(
        query(event_date="2026-12-03"), catalogue().iloc[[0, 1]], use_openai=False
    )
    assert response["message"].startswith("Под условия подходят только 2")
    assert len(response["results"]) == 2
    assert {"synthetic", "city_imputed", "price_imputed"}.issubset(response["results"][0])


def test_deterministic_order_and_scores_with_fallback():
    first = recommend_contractors(query(), catalogue(), use_openai=False)
    second = recommend_contractors(query(), catalogue().sample(frac=1, random_state=9), use_openai=False)
    assert [card["id"] for card in first["results"]] == [card["id"] for card in second["results"]]
    assert first["diagnostics"]["semantic_provider"] == "tfidf"


def test_openai_provider_and_fallback_diagnostic_are_kept_internal():
    ranked = catalogue().iloc[[0, 1]].copy()
    scores = pd.DataFrame({"semantic_score": [0.9, 0.8], "budget_score": [1, 1], "duration_score": [1, 1], "final_score": [0.9, 0.8], "semantic_provider": ["openai", "openai"], "semantic_fallback_reason": ["", ""]}, index=ranked.index)
    with patch("src.recommender.rank_candidates", return_value=ranked.assign(**scores)):
        response = recommend_contractors(query(), ranked, use_openai=True)
    assert response["diagnostics"]["semantic_provider"] == "openai"
    assert all("semantic_fallback_reason" not in card for card in response["results"])


def test_no_secrets_in_diagnostics():
    response = recommend_contractors(query(event_format="конференция"), catalogue(), use_openai=False)
    assert "OPENAI_API_KEY" not in repr(response)
    assert "secret" not in repr(response).lower()

