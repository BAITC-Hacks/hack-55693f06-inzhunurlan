"""Deterministic, evidence-based explanations for ranked contractors."""

import pandas as pd

from src.models import SearchQuery


def _values(row: pd.Series, parsed: str, raw: str) -> set[str]:
    value = row.get(parsed)
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    value = row.get(raw, "")
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip()}


def _money(value: float) -> str:
    return f"{int(value):,}".replace(",", " ")


def _hours(value: float) -> str:
    number = int(value) if float(value).is_integer() else value
    last_two = int(value) % 100
    last = int(value) % 10
    word = "час" if last == 1 and last_two != 11 else "часа" if 2 <= last <= 4 and not 12 <= last_two <= 14 else "часов"
    return f"{number} {word}"


def _duration_buffer(row: pd.Series, query: SearchQuery) -> float | None:
    if query.duration_hours is None:
        return None
    maximum = row.get("max_hours")
    if pd.isna(maximum):
        return None
    return float(maximum) - float(query.duration_hours)


def _semantic_winner(row: pd.Series, top_candidates: pd.DataFrame) -> bool:
    if "semantic_score" not in top_candidates.columns:
        return False
    scores = pd.to_numeric(top_candidates["semantic_score"], errors="coerce")
    if scores.notna().sum() == 0:
        return False
    best = scores.max()
    return bool(pd.to_numeric(pd.Series([row.get("semantic_score")]), errors="coerce").iloc[0] == best)


def _cheapest(row: pd.Series, top_candidates: pd.DataFrame) -> bool:
    prices = pd.to_numeric(top_candidates.get("price_from_kzt"), errors="coerce")
    price = pd.to_numeric(pd.Series([row.get("price_from_kzt")]), errors="coerce").iloc[0]
    return bool(pd.notna(price) and price == prices.min())


def _largest_buffer(row: pd.Series, top_candidates: pd.DataFrame, query: SearchQuery) -> bool:
    if query.duration_hours is None:
        return False
    buffers = top_candidates.apply(lambda item: _duration_buffer(item, query), axis=1)
    buffer = _duration_buffer(row, query)
    numeric = pd.to_numeric(buffers, errors="coerce")
    return bool(buffer is not None and numeric.notna().any() and buffer == numeric.max())


def generate_explanation(
    row: pd.Series,
    top_candidates: pd.DataFrame,
    query: SearchQuery,
) -> str:
    """Generate one or two deterministic sentences from explicit catalogue evidence."""
    price = float(row["price_from_kzt"])
    difference = float(query.budget_kzt) - price
    price_note = "Стартовая цена"
    imputed_note = " (значение восстановлено при подготовке каталога)" if bool(row.get("price_imputed", False)) else ""
    if difference > 0:
        budget_text = f"{price_note} — {_money(price)} ₸{imputed_note}, что на {_money(difference)} ₸ ниже указанного бюджета."
    elif difference == 0:
        budget_text = f"{price_note} — {_money(price)} ₸{imputed_note}, что равно указанному бюджету."
    else:
        budget_text = f"{price_note} — {_money(price)} ₸{imputed_note}, что выше указанного бюджета."

    reasons: list[str] = []
    if _semantic_winner(row, top_candidates):
        reasons.append("Среди рекомендованных кандидатов профиль имеет самое сильное смысловое совпадение с указанными пожеланиями")
    if _cheapest(row, top_candidates):
        reasons.append("Среди рекомендованных кандидатов у профиля самая низкая стартовая цена")
    buffer = _duration_buffer(row, query)
    if _largest_buffer(row, top_candidates, query) and buffer is not None:
        reasons.append(f"Может работать до {_hours(float(row['max_hours']))} — это на {_hours(buffer)} больше запрошенной длительности")

    requested_language = (query.language or "").strip()
    if requested_language and requested_language in _values(row, "language_list", "languages"):
        reasons.append(f"Поддерживает запрошенный язык: {requested_language}")

    # These facts come from hard-filter success and do not depend on text inference.
    reasons.append(f"Работает с форматом «{query.event_format}» и свободен на выбранную дату")
    selected_reasons = reasons[:3]
    selected_reasons[0] = selected_reasons[0][:1].upper() + selected_reasons[0][1:]
    selected_reasons[1:] = [item[:1].lower() + item[1:] for item in selected_reasons[1:]]
    reason_text = "; ".join(selected_reasons)
    return f"{budget_text} {reason_text}."


def generate_explanations(
    ranked_candidates: pd.DataFrame,
    query: SearchQuery,
) -> list[str]:
    """Generate explanations in the existing ranked order."""
    return [
        generate_explanation(row, ranked_candidates, query)
        for _, row in ranked_candidates.iterrows()
    ]

