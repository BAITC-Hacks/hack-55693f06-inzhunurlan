from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/hackathon_dataset_anonymized.csv")

REQUIRED_COLUMNS = {
    "id",
    "anon_name",
    "categories",
    "city",
    "city_imputed",
    "synthetic",
    "price_from_kzt",
    "price_imputed",
    "event_formats",
    "languages",
    "max_hours",
    "busy_dates",
    "description",
}


def split_pipe(value) -> list[str]:
    """
    Convert pipe-separated values like:
    'Ведущий|Ведущий церемонии'

    into:
    ['Ведущий', 'Ведущий церемонии']
    """
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # -------------------------------------------------
    # 1. Schema
    # -------------------------------------------------

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    extra_columns = set(df.columns) - REQUIRED_COLUMNS

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("\n[OK] All required columns are present.")

    if extra_columns:
        print(
            "[INFO] Extra columns:",
            sorted(extra_columns),
        )

    # -------------------------------------------------
    # 2. IDs
    # -------------------------------------------------

    if df["id"].isna().any():
        raise ValueError("Found missing IDs.")

    duplicated_ids = df[df["id"].duplicated()]

    if not duplicated_ids.empty:
        raise ValueError(
            "Duplicate IDs found:\n"
            + duplicated_ids[["id", "anon_name"]].to_string(
                index=False
            )
        )

    print("[OK] IDs are non-null and unique.")

    # -------------------------------------------------
    # 3. Required values
    # -------------------------------------------------

    required_non_null = [
        "anon_name",
        "categories",
        "city",
        "price_from_kzt",
        "event_formats",
        "languages",
    ]

    for column in required_non_null:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            raise ValueError(
                f"Column '{column}' contains "
                f"{missing_count} null values."
            )

    print("[OK] Required business fields are populated.")

    # -------------------------------------------------
    # 4. Prices
    # -------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df["price_from_kzt"]
    ):
        raise ValueError(
            "price_from_kzt must be numeric."
        )

    invalid_prices = df[
        df["price_from_kzt"] < 0
    ]

    if not invalid_prices.empty:
        raise ValueError(
            "Negative prices found."
        )

    print(
        "[OK] Prices are valid."
    )

    print(
        f"     Min price: "
        f"{int(df['price_from_kzt'].min()):,} KZT"
    )

    print(
        f"     Max price: "
        f"{int(df['price_from_kzt'].max()):,} KZT"
    )

    # -------------------------------------------------
    # 5. max_hours
    # -------------------------------------------------

    invalid_max_hours = df[
        df["max_hours"].notna()
        & (df["max_hours"] <= 0)
    ]

    if not invalid_max_hours.empty:
        raise ValueError(
            "Found invalid max_hours <= 0."
        )

    null_max_hours = int(
        df["max_hours"].isna().sum()
    )

    print(
        f"[OK] max_hours valid. "
        f"Null/not-applicable values: "
        f"{null_max_hours}"
    )

    # -------------------------------------------------
    # 6. Boolean flags
    # -------------------------------------------------

    flag_columns = [
        "synthetic",
        "city_imputed",
        "price_imputed",
    ]

    for column in flag_columns:
        values = set(
            df[column]
            .dropna()
            .unique()
            .tolist()
        )

        if not values.issubset({True, False}):
            raise ValueError(
                f"Unexpected values in '{column}': "
                f"{values}"
            )

    print("[OK] Boolean flags contain valid values.")

    # -------------------------------------------------
    # 7. Categories
    # -------------------------------------------------

    all_categories = sorted(
        {
            category
            for raw in df["categories"]
            for category in split_pipe(raw)
        }
    )

    print(
        f"\nUnique categories: "
        f"{len(all_categories)}"
    )

    print("\nCATEGORY COUNTS")

    category_counts = []

    for category in all_categories:
        count = int(
            df["categories"].apply(
                lambda raw:
                category in split_pipe(raw)
            ).sum()
        )

        category_counts.append(
            (category, count)
        )

    for category, count in sorted(
        category_counts,
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"{category:<35} {count:>3}"
        )

    # -------------------------------------------------
    # 8. Cities
    # -------------------------------------------------

    print("\nCITY COUNTS")

    print(
        df["city"]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------
    # 9. Event formats
    # -------------------------------------------------

    all_formats = sorted(
        {
            event_format
            for raw in df["event_formats"]
            for event_format in split_pipe(raw)
        }
    )

    print("\nEVENT FORMATS")

    for event_format in all_formats:
        print(f"- {event_format}")

    # -------------------------------------------------
    # 10. Languages
    # -------------------------------------------------

    all_languages = sorted(
        {
            language
            for raw in df["languages"]
            for language in split_pipe(raw)
        }
    )

    print("\nLANGUAGES")

    for language in all_languages:
        print(f"- {language}")

    # -------------------------------------------------
    # 11. Synthetic / imputed statistics
    # -------------------------------------------------

    print("\nDATA QUALITY FLAGS")

    print(
        "Synthetic profiles:",
        int(df["synthetic"].sum()),
    )

    print(
        "City imputed:",
        int(df["city_imputed"].sum()),
    )

    print(
        "Price imputed:",
        int(df["price_imputed"].sum()),
    )

    # -------------------------------------------------
    # 12. Busy dates basic validation
    # -------------------------------------------------

    invalid_busy_dates = []

    for _, row in df.iterrows():
        for date_value in split_pipe(
            row["busy_dates"]
        ):
            try:
                parsed = pd.to_datetime(
                    date_value,
                    format="%Y-%m-%d",
                    errors="raise",
                )

                if (
                    parsed < pd.Timestamp("2026-09-23")
                    or parsed
                    > pd.Timestamp("2026-12-31")
                ):
                    invalid_busy_dates.append(
                        (
                            row["id"],
                            date_value,
                            "outside expected range",
                        )
                    )

            except Exception:
                invalid_busy_dates.append(
                    (
                        row["id"],
                        date_value,
                        "invalid date",
                    )
                )

    if invalid_busy_dates:
        print(
            "\n[WARNING] Suspicious busy_dates:"
        )

        for item in invalid_busy_dates[:20]:
            print(item)

    else:
        print(
            "\n[OK] busy_dates are valid "
            "and inside expected range."
        )

    # -------------------------------------------------
    # Final
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()