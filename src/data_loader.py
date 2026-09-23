from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(
    "data/hackathon_dataset_anonymized.csv"
)


def split_pipe(value) -> list[str]:
    """
    Convert pipe-separated catalogue fields into
    normalized Python lists.

    Example:
        "русский|казахский"
    becomes:
        ["русский", "казахский"]
    """
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def parse_busy_dates(value) -> set[str]:
    """
    Busy dates are stored as pipe-separated ISO dates.

    A set is used because availability checks are
    membership operations:
        event_date in busy_dates
    """
    return set(split_pipe(value))


def load_contractors(
    path: Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the original hackathon catalogue and add
    parsed helper columns.

    The original source columns are preserved.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path).copy()

    df["category_list"] = (
        df["categories"].apply(split_pipe)
    )

    df["event_format_list"] = (
        df["event_formats"].apply(split_pipe)
    )

    df["language_list"] = (
        df["languages"].apply(split_pipe)
    )

    df["busy_date_set"] = (
        df["busy_dates"].apply(parse_busy_dates)
    )

    df["description"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df