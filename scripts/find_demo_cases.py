"""Find deterministic, dataset-backed scenarios for the hackathon demo.

This utility deliberately stops at hard filtering.  It never calls OpenAI or
the ranking layer, so running it is safe during a live-demo rehearsal.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_contractors
from src.filters import city_category_candidates, filter_candidates
from src.models import SearchQuery


DATA_PATH = ROOT / "data" / "hackathon_dataset_anonymized.csv"
DATE_START = date(2026, 9, 23)
DATE_END = date(2026, 12, 31)
DENSE_CATEGORIES = ("Ведущий", "Фотограф", "Банкетный зал")
RARE_CATEGORIES = (
    "Флорист", "Декоратор", "Подарки и сувениры", "Ведущий церемонии",
    "Фото и видеобудки", "Отель", "Инструменталист",
)


def dates_in_window() -> list[str]:
    """Return every supported ISO date in deterministic order."""
    days = (DATE_END - DATE_START).days + 1
    return [(DATE_START + timedelta(days=offset)).isoformat() for offset in range(days)]


def dates_for_rows(rows: pd.DataFrame, *, weekends: bool = False) -> list[str]:
    """Use dates relevant to these profiles while staying inside the supported window."""
    dates = {DATE_START.isoformat(), DATE_END.isoformat()}
    for busy_dates in rows["busy_date_set"]:
        dates.update(day for day in busy_dates if DATE_START.isoformat() <= day <= DATE_END.isoformat())
    if weekends:
        dates.update(day for day in dates_in_window() if date.fromisoformat(day).weekday() >= 5)
    return sorted(dates)


def _values(df: pd.DataFrame, column: str) -> list[str]:
    values: set[str] = set()
    for value in df[column].dropna():
        values.update(item.strip() for item in str(value).split("|") if item.strip())
    return sorted(values)


def _money_steps(values: list[float]) -> list[int]:
    """Create realistic budget candidates from observed starting prices."""
    if not values:
        return []
    # The discovery search is intentionally bounded; no random or exhaustive
    # budget grid is needed because no-result cases derive their own budget.
    maximum = max(values)
    return [max(50_000, int((maximum // 50_000) * 50_000))]


def _duration_options(rows: pd.DataFrame) -> list[float | None]:
    return [None]


def _language_options(rows: pd.DataFrame) -> list[str | None]:
    return [None]


def evaluate(df: pd.DataFrame, query: SearchQuery) -> dict[str, Any]:
    """Apply the existing discovery and hard-filter functions once."""
    candidates = city_category_candidates(df, query)
    eligible, diagnostics = filter_candidates(candidates, query)
    return {
        "query": query,
        "candidates": candidates,
        "eligible": eligible,
        "diagnostics": diagnostics,
    }


def _query_variants(df: pd.DataFrame, categories: tuple[str, ...]):
    """Yield bounded, deterministic query combinations from observed values."""
    for category in categories:
        category_rows = df[df["category_list"].apply(lambda values: category in values)]
        for city in sorted(category_rows["city"].unique()):
            rows = category_rows[category_rows["city"] == city]
            formats = _values(rows, "event_formats")
            budgets = _money_steps(pd.to_numeric(rows["price_from_kzt"]).tolist())
            for event_format in formats:
                for budget in budgets:
                    for language in _language_options(rows):
                        for duration in _duration_options(rows):
                            yield category, city, event_format, budget, language, duration, rows


def find_dense(df: pd.DataFrame) -> dict[str, Any]:
    """Choose a dense case with at least three eligible and one rejected row."""
    for category, city, event_format, budget, language, duration, rows in _query_variants(df, DENSE_CATEGORIES):
        for event_date in dates_for_rows(rows):
            query = SearchQuery(city, event_date, event_format, category, budget, duration, language)
            result = evaluate(df, query)
            candidates = result["candidates"]
            eligible = result["eligible"]
            rejected = len(candidates) - len(eligible)
            if len(candidates) <= 3 or len(eligible) < 3 or rejected == 0:
                continue
            return result
    raise RuntimeError("Не найден dense-кейс с минимум 3 eligible-кандидатами.")


def find_rare(df: pd.DataFrame) -> dict[str, Any]:
    """Choose a rare category with an honest pool of one to three results."""
    for category, city, event_format, budget, language, duration, rows in _query_variants(df, RARE_CATEGORIES):
        for event_date in dates_for_rows(rows):
            query = SearchQuery(city, event_date, event_format, category, budget, duration, language)
            result = evaluate(df, query)
            candidates = result["candidates"]
            eligible = result["eligible"]
            if candidates.empty or not 1 <= len(eligible) <= 3:
                continue
            return result
    raise RuntimeError("Не найден rare-кейс с 1–3 eligible-кандидатами.")


def find_no_result(df: pd.DataFrame) -> dict[str, Any]:
    """Find a realistic zero-result query, preferring weekend budget pressure."""
    for category, city, event_format, _budget, language, duration, rows in _query_variants(df, DENSE_CATEGORIES + RARE_CATEGORIES):
        prices = sorted(pd.to_numeric(rows["price_from_kzt"]).astype(int).tolist())
        if not prices:
            continue
        minimum = prices[0]
        # Stay below the cheapest observed starting price, but avoid absurd budgets.
        budget = max(100_000, int((minimum * 0.8) // 50_000 * 50_000))
        if budget >= minimum:
            continue
        for event_date in dates_for_rows(rows, weekends=True):
            query = SearchQuery(city, event_date, event_format, category, budget, language=language, duration_hours=duration)
            result = evaluate(df, query)
            if result["candidates"].empty or not result["eligible"].empty:
                continue
            return result
    raise RuntimeError("Не найден realistic no-result-кейс.")


def find_date_sensitivity(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find two dates for one query whose eligible IDs differ due to busy_dates."""
    for category, city, event_format, budget, language, duration, rows in _query_variants(df, ("Ведущий", "Фотограф")):
        seen: dict[frozenset[str], dict[str, Any]] = {}
        for event_date in dates_for_rows(rows):
            query = SearchQuery(city, event_date, event_format, category, budget, duration, language)
            result = evaluate(df, query)
            ids = frozenset(result["eligible"]["id"])
            if len(ids) < 2 or ids in seen:
                continue
            for previous in seen.values():
                previous_ids = set(previous["eligible"]["id"])
                difference = set(ids).symmetric_difference(previous_ids)
                if not difference:
                    continue
                date_a, date_b = previous["query"].event_date, event_date
                busy_difference = difference & set(rows.loc[rows["busy_date_set"].apply(lambda dates_: date_a in dates_ or date_b in dates_), "id"])
                if not busy_difference:
                    continue
                return previous, result
            seen[ids] = result
    raise RuntimeError("Не найден date-sensitive dense-кейс.")


def _names(rows: pd.DataFrame) -> list[str]:
    return [f"{row.id} — {row.anon_name}" for row in rows.itertuples()]


def _print_query(query: SearchQuery) -> None:
    print(f"Город: {query.city}")
    print(f"Дата: {query.event_date}")
    print(f"Формат мероприятия: {query.event_format}")
    print(f"Категория: {query.category}")
    print(f"Бюджет: {query.budget_kzt:,} KZT".replace(",", " "))
    print(f"Язык: {query.language or 'не указан'}")
    print(f"Длительность: {query.duration_hours:g} ч." if query.duration_hours is not None else "Длительность: не указана")


def print_standard_case(title: str, result: dict[str, Any]) -> None:
    query = result["query"]
    candidates, eligible = result["candidates"], result["eligible"]
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)
    _print_query(query)
    print(f"Город/категория кандидатов: {len(candidates)}")
    print(f"Eligible: {len(eligible)}")
    print(f"Отклонено: {len(candidates) - len(eligible)}")
    print(f"Причины: {dict(sorted(result['diagnostics']['reason_counts'].items())) or 'нет'}")
    print("Подходящие ID/имена:")
    for item in _names(eligible):
        print(f"  - {item}")


def print_date_case(first: dict[str, Any], second: dict[str, Any]) -> None:
    query_a, query_b = first["query"], second["query"]
    print("\n" + "=" * 58)
    print("DEMO 4 — DATE SENSITIVITY")
    print("=" * 58)
    print("Общие параметры запроса:")
    print(f"Город: {query_a.city}; формат: {query_a.event_format}; категория: {query_a.category}; бюджет: {query_a.budget_kzt:,} KZT".replace(",", " "))
    print(f"Язык: {query_a.language or 'не указан'}; длительность: {query_a.duration_hours or 'не указана'}")
    for label, result in (("DATE A", first), ("DATE B", second)):
        query = result["query"]
        candidates, eligible = result["candidates"], result["eligible"]
        eligible_ids = set(eligible["id"])
        busy_ids = set(candidates.loc[candidates["busy_date_set"].apply(lambda dates: query.event_date in dates), "id"])
        print(f"\n{label}: {query.event_date}")
        print(f"Eligible IDs: {sorted(eligible_ids)}")
        print(f"Busy relevant IDs: {sorted(busy_ids)}")
    ids_a, ids_b = set(first["eligible"]["id"]), set(second["eligible"]["id"])
    print(f"\nРазница из-за доступности: {sorted(ids_a.symmetric_difference(ids_b))}")


def main() -> None:
    df = load_contractors(DATA_PATH)
    dense = find_dense(df)
    rare = find_rare(df)
    no_result = find_no_result(df)
    date_a, date_b = find_date_sensitivity(df)
    print_standard_case("DEMO 1 — DENSE CATEGORY", dense)
    print_standard_case("DEMO 2 — RARE CATEGORY", rare)
    print_standard_case("DEMO 3 — NO ELIGIBLE CANDIDATES", no_result)
    print_date_case(date_a, date_b)


if __name__ == "__main__":
    main()
