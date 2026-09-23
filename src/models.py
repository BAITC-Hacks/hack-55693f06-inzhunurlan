from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SearchQuery:
    city: str
    event_date: str
    event_format: str
    category: str
    budget_kzt: int
    duration_hours: Optional[float] = None
    language: Optional[str] = None
    preferences: Optional[str] = None