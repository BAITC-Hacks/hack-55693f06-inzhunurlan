"""Application orchestration for the contractor recommendation MVP."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import DEFAULT_DATA_PATH, load_contractors
from src.explanations import generate_explanations
from src.filters import city_category_candidates, filter_candidates
from src.models import SearchQuery
from src.ranking import rank_candidates


RESULT_FIELDS = (
    "id", "anon_name", "categories", "city", "price_from_kzt",
    "event_formats", "languages", "max_hours", "synthetic", "city_imputed",
    "price_imputed", "semantic_score", "budget_score", "duration_score",
    "final_score", "semantic_provider", "explanation",
)


def _count_phrase(count: int, forms: tuple[str, str, str]) -> str:
    """Return a deterministic Russian noun phrase for 1/2-4/5+ counts."""
    number = abs(int(count))
    last_two = number % 100
    last = number % 10
    if last == 1 and last_two != 11:
        word = forms[0]
    elif 2 <= last <= 4 and not 12 <= last_two <= 14:
        word = forms[1]
    else:
        word = forms[2]
    return f"{count} {word}"


def _reason_summary(reason_counts: dict[str, int]) -> str:
    labels = {
        "busy": (("подрядчик", "подрядчика", "подрядчиков"),
                 ("занят на выбранную дату", "заняты на выбранную дату")),
        "over_budget": (("подрядчик", "подрядчика", "подрядчиков"),
                        ("имеет стартовую цену выше бюджета", "имеют стартовую цену выше бюджета")),
        "wrong_format": (("подрядчик", "подрядчика", "подрядчиков"),
                         ("не поддерживает выбранный формат", "не поддерживают выбранный формат")),
        "wrong_language": (("подрядчик", "подрядчика", "подрядчиков"),
                           ("не поддерживает запрошенный язык", "не поддерживают запрошенный язык")),
        "duration_too_long": (("подрядчик", "подрядчика", "подрядчиков"),
                              ("не проходит по длительности", "не проходят по длительности")),
    }
    parts = []
    for reason, count in reason_counts.items():
        if not count:
            continue
        label = labels.get(reason)
        if label is None:
            parts.append(f"{count} {reason}")
            continue
        noun_forms, predicate_forms = label
        predicate = predicate_forms[0] if _count_phrase(count, noun_forms).split(" ", 1)[1] == noun_forms[0] else predicate_forms[1]
        parts.append(f"{_count_phrase(count, noun_forms)} {predicate}")
    return ", ".join(parts)


def _card(row: pd.Series) -> dict[str, Any]:
    """Convert one ranked row to the stable user-facing card schema."""
    return {
        field: row.get(field)
        for field in RESULT_FIELDS
        if field != "explanation"
    }


def recommend_contractors(
    query: SearchQuery,
    contractors: pd.DataFrame | None = None,
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    use_openai: bool = True,
) -> dict[str, Any]:
    """Run discovery, hard filtering, ranking, top-three selection and explanations."""
    catalogue = (
        contractors.copy()
        if contractors is not None
        else load_contractors(data_path)
    )
    candidates = city_category_candidates(catalogue, query)
    diagnostics: dict[str, Any] = {
        "city_category_candidates": int(len(candidates)),
        "eligible_candidates": 0,
        "rejection_reason_counts": {},
    }

    if candidates.empty:
        return {
            "status": "CATEGORY_NOT_FOUND",
            "message": f"В городе «{query.city}» нет подрядчиков категории «{query.category}».",
            "results": [],
            "diagnostics": diagnostics,
        }

    eligible, filter_diagnostics = filter_candidates(candidates, query)
    diagnostics["eligible_candidates"] = int(len(eligible))
    diagnostics["rejection_reason_counts"] = filter_diagnostics["reason_counts"]
    if eligible.empty:
        summary = _reason_summary(filter_diagnostics["reason_counts"])
        suffix = f" Среди причин: {summary}." if summary else ""
        candidate_phrase = _count_phrase(
            len(candidates), ("подрядчик", "подрядчика", "подрядчиков")
        )
        return {
            "status": "NO_ELIGIBLE_CANDIDATES",
            "message": (
                f"В городе найдено {candidate_phrase} этой категории, "
                f"но ни один не проходит заданные условия.{suffix}"
            ),
            "results": [],
            "diagnostics": diagnostics,
        }

    ranked = rank_candidates(eligible, query, use_openai=use_openai)
    top = ranked.head(3).copy()
    explanations = generate_explanations(top, query)
    providers = (
        sorted(set(top["semantic_provider"].dropna().astype(str)))
        if "semantic_provider" in top else []
    )
    diagnostics["semantic_provider"] = providers[0] if len(providers) == 1 else providers
    fallback_reasons = (
        sorted(set(top["semantic_fallback_reason"].dropna().astype(str)))
        if "semantic_fallback_reason" in top
        else []
    )
    fallback_reasons = [reason for reason in fallback_reasons if reason]
    if fallback_reasons:
        diagnostics["semantic_fallback_reason"] = (
            fallback_reasons[0] if len(fallback_reasons) == 1 else fallback_reasons
        )

    results = []
    for (_, row), explanation in zip(top.iterrows(), explanations):
        card = _card(row)
        card["explanation"] = explanation
        results.append(card)

    if len(results) < 3:
        candidate_phrase = _count_phrase(
            len(results), ("кандидат", "кандидата", "кандидатов")
        )
        verb = "подходит" if len(results) == 1 else "подходят"
        message = f"Под условия {verb} только {candidate_phrase} из доступных."
    else:
        message = "Найдено 3 подходящих подрядчика."
    return {
        "status": "MATCHED",
        "message": message,
        "results": results,
        "diagnostics": diagnostics,
    }


recommend = recommend_contractors

