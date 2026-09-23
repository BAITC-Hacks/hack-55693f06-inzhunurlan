from collections import Counter

import pandas as pd

from src.models import SearchQuery


def city_category_candidates(
    df: pd.DataFrame,
    query: SearchQuery,
) -> pd.DataFrame:
    """
    Find contractors belonging to the requested
    category in the requested city.

    This stage is intentionally separate because the
    product must distinguish:

    1. category does not exist in the city;
    2. category exists, but nobody passes the order
       constraints.
    """

    mask = (
        (df["city"] == query.city)
        & df["category_list"].apply(
            lambda categories:
            query.category in categories
        )
    )

    return df.loc[mask].copy()


def get_rejection_reasons(
    row: pd.Series,
    query: SearchQuery,
) -> list[str]:
    """
    Return every hard constraint violated by
    one contractor.
    """

    reasons = []

    # Availability
    if query.event_date in row["busy_date_set"]:
        reasons.append("busy")

    # Budget
    # price_from_kzt means starting price.
    if row["price_from_kzt"] > query.budget_kzt:
        reasons.append("over_budget")

    # Event format
    if query.event_format not in row["event_format_list"]:
        reasons.append("wrong_format")

    # Optional language
    if (
        query.language
        and query.language not in row["language_list"]
    ):
        reasons.append("wrong_language")

    # Optional duration
    #
    # null max_hours means that duration is not tied
    # to physical presence (e.g. florist/decorator),
    # therefore it must NOT cause rejection.
    if query.duration_hours is not None:
        max_hours = row["max_hours"]

        if (
            pd.notna(max_hours)
            and max_hours < query.duration_hours
        ):
            reasons.append("duration_too_long")

    return reasons


def filter_candidates(
    candidates: pd.DataFrame,
    query: SearchQuery,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply hard constraints.

    Returns:
        eligible contractors
        diagnostics
    """

    if candidates.empty:
        return candidates.copy(), {
            "reason_counts": {},
            "rejected_contractors": [],
        }

    eligible_indices = []
    rejected_contractors = []
    reason_counter = Counter()

    for index, row in candidates.iterrows():

        reasons = get_rejection_reasons(
            row,
            query,
        )

        if reasons:
            reason_counter.update(reasons)

            rejected_contractors.append(
                {
                    "id": row["id"],
                    "name": row["anon_name"],
                    "reasons": reasons,
                }
            )

        else:
            eligible_indices.append(index)

    eligible = candidates.loc[
        eligible_indices
    ].copy()

    diagnostics = {
        "reason_counts": dict(reason_counter),
        "rejected_contractors":
            rejected_contractors,
    }

    return eligible, diagnostics