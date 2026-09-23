import numpy as np
import pandas as pd

from src.explanations import generate_explanation, generate_explanations
from src.models import SearchQuery


QUERY = SearchQuery(
    city="Алматы",
    event_date="2026-10-10",
    event_format="свадьба",
    category="Ведущий",
    budget_kzt=1_000_000,
    duration_hours=6,
    language="русский",
    preferences="лёгкий юмор",
)


def candidates():
    return pd.DataFrame([
        {"id": "A", "price_from_kzt": 900000, "price_imputed": False, "max_hours": 10,
         "languages": "русский|казахский", "language_list": ["русский", "казахский"],
         "semantic_score": .51, "event_format_list": ["свадьба"]},
        {"id": "B", "price_from_kzt": 700000, "price_imputed": False, "max_hours": 8,
         "languages": "русский", "language_list": ["русский"],
         "semantic_score": .40, "event_format_list": ["свадьба"]},
        {"id": "C", "price_from_kzt": 850000, "price_imputed": True, "max_hours": np.nan,
         "languages": "казахский", "language_list": ["казахский"],
         "semantic_score": .30, "event_format_list": ["свадьба"]},
    ])


def test_price_and_starting_price_language_and_duration():
    text = generate_explanation(candidates().iloc[0], candidates(), QUERY)
    assert "Стартовая цена — 900 000 ₸" in text
    assert "на 100 000 ₸ ниже" in text
    assert "русский" in text
    assert "на 4 часа больше запрошенной длительности" in text
    assert "финаль" not in text.lower()


def test_language_only_appears_when_requested():
    query = QUERY.__class__(**{**QUERY.__dict__, "language": None})
    text = generate_explanation(candidates().iloc[0], candidates(), query)
    assert "русский" not in text


def test_nan_duration_does_not_create_duration_claim():
    text = generate_explanation(candidates().iloc[2], candidates(), QUERY)
    assert "час" not in text
    assert "значение восстановлено" in text


def test_semantic_winner_and_cheapest_are_distinctive():
    ranked = candidates()
    first = generate_explanation(ranked.iloc[0], ranked, QUERY)
    second = generate_explanation(ranked.iloc[1], ranked, QUERY)
    assert "самое сильное смысловое совпадение" in first
    assert "самая низкая стартовая цена" in second
    assert first != second


def test_same_inputs_are_deterministic_and_top_set_is_distinguishable():
    ranked = candidates()
    one = generate_explanations(ranked, QUERY)
    two = generate_explanations(ranked, QUERY)
    assert one == two
    assert len(set(one)) == 3


def test_flags_are_not_positive_reasons():
    text = generate_explanation(candidates().iloc[2], candidates(), QUERY)
    assert "synthetic" not in text.lower()
    assert "city imputed" not in text.lower()
    assert "recommended because" not in text.lower()

