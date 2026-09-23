"""Streamlit demo UI for the contractor recommendation MVP."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DEFAULT_DATA_PATH, load_contractors
from src.models import SearchQuery
from src.recommender import recommend_contractors


DATA_PATH = Path(__file__).resolve().parent / DEFAULT_DATA_PATH
SUPPORTED_START = date(2026, 9, 23)
SUPPORTED_END = date(2026, 12, 31)


def _catalog_values(catalogue: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique values from a pipe-separated catalogue column."""
    values: set[str] = set()
    for value in catalogue[column].dropna():
        values.update(item.strip() for item in str(value).split("|") if item.strip())
    return sorted(values, key=str.casefold)


def _format_price(value: object) -> str:
    """Format a catalogue price for a compact card."""
    try:
        return f"{float(value):,.0f}".replace(",", " ") + " ₸"
    except (TypeError, ValueError):
        return "Цена не указана"


def _display_pipe(value: object) -> str:
    """Render pipe-separated catalogue values as readable UI text."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " · ".join(item.strip() for item in str(value).split("|") if item.strip())


def _inject_styles() -> None:
    """Apply the restrained editorial visual system without changing app behaviour."""
    st.markdown(
        """
        <style>
        :root { --ink:#111111; --muted:#6b6b66; --paper:#f5f4f0; --surface:#ffffff; --soft:#ecebe6; --line:#ddddd7; }
        .stApp { background:var(--paper); color:var(--ink); }
        .block-container { max-width:1160px; padding:28px 5vw 56px; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stToolbar"] { visibility:hidden; }
        [data-testid="stDecoration"], #MainMenu, footer { display:none; }
        h1,h2,h3 { font-family:Inter,Arial,sans-serif; color:var(--ink); letter-spacing:-.045em; }
        p, label, .stMarkdown, .stCaption { font-family:Inter,Arial,sans-serif; }
        .brand-bar { display:flex; justify-content:space-between; align-items:center; padding:2px 0 72px; border-bottom:1px solid var(--line); }
        .wordmark { font:800 18px Inter,Arial,sans-serif; letter-spacing:.08em; }
        .descriptor { margin-left:12px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.13em; }
        .nav { color:var(--muted); font-size:12px; letter-spacing:.06em; text-transform:uppercase; }
        .nav span { margin-left:26px; }
        .eyebrow, .section-label { color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
        .hero { padding:76px 0 52px; max-width:780px; }
        .hero h1 { margin:16px 0 20px; font-size:clamp(42px,6vw,78px); line-height:.98; }
        .hero-copy { max-width:610px; color:var(--muted); font-size:17px; line-height:1.55; }
        .metrics { display:flex; gap:42px; padding-top:38px; }
        .metric { border-left:1px solid var(--line); padding-left:14px; }
        .metric strong { display:block; font:700 22px Inter,Arial,sans-serif; }
        .metric span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.1em; }
        .section-heading { margin:12px 0 24px; font:700 clamp(28px,4vw,44px) Inter,Arial,sans-serif; letter-spacing:-.04em; }
        div[data-testid="stForm"] { background:var(--surface); border:1px solid var(--line); border-radius:22px; padding:26px 30px 30px; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="textarea"], input, textarea { background:#fcfcfa !important; border:1px solid var(--line) !important; border-radius:13px !important; color:var(--ink) !important; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"], input { min-height:48px; }
        div[data-baseweb="input"] input, textarea { padding-top:12px !important; padding-bottom:12px !important; }
        div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within, textarea:focus { border-color:var(--ink) !important; box-shadow:0 0 0 1px var(--ink) !important; }
        div[data-testid="stFormSubmitButton"] button { min-height:50px; background:var(--ink); color:white; border:1px solid var(--ink); border-radius:999px; padding:11px 28px; font-size:14px; font-weight:600; transition:background .15s ease, transform .15s ease; }
        div[data-testid="stFormSubmitButton"] button:hover { background:#363633; color:#fff; border-color:#363633; }
        .stSelectbox label, .stNumberInput label, .stDateInput label, .stTextArea label { color:var(--ink); font-weight:600; }
        .hint { color:var(--muted); font-size:12px; line-height:1.45; margin-top:-8px; }
        .results-wrap { margin-top:78px; padding-top:30px; border-top:1px solid var(--line); }
        .result-note { color:var(--muted); margin:-10px 0 26px; }
        .result-card { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:28px 30px; margin:14px 0; }
        .card-top { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }
        .rank { color:var(--muted); font:700 12px Inter,Arial,sans-serif; letter-spacing:.12em; }
        .card-name { margin:7px 0 5px; font:700 28px Inter,Arial,sans-serif; letter-spacing:-.035em; }
        .card-meta { color:var(--muted); font-size:13px; }
        .price { font:700 20px Inter,Arial,sans-serif; white-space:nowrap; }
        .why-label { margin-top:28px; color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.14em; }
        .explanation { max-width:820px; margin-top:8px; font-size:16px; line-height:1.55; }
        .tags { display:flex; flex-wrap:wrap; gap:7px; margin-top:22px; }
        .tag { background:var(--soft); border-radius:999px; padding:6px 10px; color:#44443f; font-size:11px; text-transform:uppercase; letter-spacing:.07em; }
        .catalogue-note { margin:15px 0 20px; color:#85857e; font-size:11px; }
        .empty-state { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:34px 30px; margin-top:28px; }
        .empty-label { color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.15em; }
        .empty-title { margin:10px 0; font:700 29px Inter,Arial,sans-serif; letter-spacing:-.035em; }
        .empty-copy { color:var(--muted); line-height:1.55; }
        .pipeline { color:var(--muted); font-size:12px; padding-bottom:14px; border-bottom:1px solid var(--line); }
        .method-strip { margin-top:70px; padding:24px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
        .method-title { font:700 20px Inter,Arial,sans-serif; }
        .method-copy { color:var(--muted); font-size:13px; line-height:1.5; max-width:740px; }
        @media (max-width:700px) { .brand-bar { padding-bottom:38px; } .nav { display:none; } .hero { padding-top:48px; } .metrics { gap:18px; flex-wrap:wrap; } div[data-testid="stForm"] { padding:20px 18px; } .card-top { display:block; } .price { margin-top:18px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_badges(card: dict) -> None:
    """Show transparency metadata without framing it as a recommendation reason."""
    badges = []
    if bool(card.get("synthetic")):
        badges.append("synthetic")
    if bool(card.get("city_imputed")):
        badges.append("город восстановлен")
    if bool(card.get("price_imputed")):
        badges.append("цена восстановлена")
    if badges:
        st.markdown(f'<div class="catalogue-note"><strong>ДАННЫЕ КАТАЛОГА</strong><br>{escape(" · ".join(badges))}</div>', unsafe_allow_html=True)


def _show_diagnostics(diagnostics: dict) -> None:
    """Render the pipeline and concise diagnostics in a secondary section."""
    with st.expander("Как система приняла решение"):
        st.markdown('<div class="pipeline">Каталог &nbsp;→&nbsp; город + категория &nbsp;→&nbsp; обязательные условия &nbsp;→&nbsp; смысловое ранжирование &nbsp;→&nbsp; до 3 рекомендаций</div>', unsafe_allow_html=True)
        st.write(f"Кандидатов в городе и категории: {diagnostics.get('city_category_candidates', 0)}")
        st.write(f"Прошли обязательные условия: {diagnostics.get('eligible_candidates', 0)}")
        st.write(f"Семантический метод: {diagnostics.get('semantic_provider', '—')}")
        reasons = diagnostics.get("rejection_reason_counts") or {}
        if reasons:
            st.write("Причины отклонения:")
            st.json(reasons)


def _show_empty_state(label: str, title: str, message: str, diagnostics: dict) -> None:
    """Render an editorial empty state while preserving the exact backend message."""
    st.markdown(f'<div class="empty-state"><div class="empty-label">{escape(label)}</div><div class="empty-title">{escape(title)}</div><div class="empty-copy">{escape(message)}</div></div>', unsafe_allow_html=True)
    _show_diagnostics(diagnostics)


def _show_results(response: dict) -> None:
    """Render all backend states and recommendation cards."""
    status = response.get("status")
    if status == "CATEGORY_NOT_FOUND":
        _show_empty_state("НЕТ КАТЕГОРИИ", "В этом городе пока нет таких подрядчиков", response.get("message", "Подходящая категория не найдена."), response.get("diagnostics", {}))
        return
    if status == "NO_ELIGIBLE_CANDIDATES":
        _show_empty_state("НЕТ СОВПАДЕНИЙ", "Кандидаты есть, но условия слишком узкие", response.get("message", "Подходящих подрядчиков не найдено."), response.get("diagnostics", {}))
        return
    if status != "MATCHED":
        st.markdown('<div class="empty-state"><div class="empty-title">Не удалось определить результат поиска</div></div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="results-wrap"><div class="section-label">02 / РЕЗУЛЬТАТ</div><div class="section-heading">Подходящие варианты</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-note">{escape(response.get("message", ""))}</div>', unsafe_allow_html=True)
    for rank, card in enumerate(response.get("results", []), start=1):
        name = escape(str(card.get("anon_name", "Подрядчик")))
        city = escape(str(card.get("city", "Город не указан")))
        price = escape(_format_price(card.get("price_from_kzt")))
        explanation = escape(str(card.get("explanation", "")))
        tags = []
        if card.get("languages"):
            tags.append(_display_pipe(card["languages"]))
        max_hours = card.get("max_hours")
        if pd.notna(max_hours):
            tags.append(f"до {max_hours:g} ч.")
        tags_html = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)
        category = escape(_display_pipe(card.get("categories")) or "Категория не указана")
        st.markdown(f'<article class="result-card"><div class="card-top"><div><div class="rank">{rank:02d} / 03</div><div class="card-name">{name}</div><div class="card-meta">{category} &nbsp;·&nbsp; {city}</div></div><div class="price">от {price}</div></div><div class="why-label">ПОЧЕМУ В РЕКОМЕНДАЦИИ</div><div class="explanation">{explanation}</div><div class="tags">{tags_html}</div></article>', unsafe_allow_html=True)
        _show_badges(card)
    _show_diagnostics(response.get("diagnostics", {}))


@st.cache_data
def _load_catalogue() -> pd.DataFrame:
    """Load and parse the immutable hackathon catalogue once per session."""
    return load_contractors(DATA_PATH)


def main() -> None:
    """Build the single-page search form and invoke the existing backend."""
    st.set_page_config(page_title="MATCH — умный подбор подрядчиков", page_icon="·", layout="centered")
    _inject_styles()
    st.markdown('<div class="brand-bar"><div><span class="wordmark">MATCH</span><span class="descriptor">умный подбор подрядчиков</span></div><div class="nav"><span>Как это работает</span><span>О системе</span></div></div>', unsafe_allow_html=True)
    st.markdown('<section class="hero"><div class="eyebrow">SMART EVENT MATCHING / КАЗАХСТАН</div><h1>Найдите тех,<br>кто действительно подходит.</h1><div class="hero-copy">Не ещё один каталог подрядчиков. Система проверяет условия заказа, доступность и смысловое соответствие — и показывает до трёх обоснованных вариантов.</div><div class="metrics"><div class="metric"><strong>66</strong><span>профилей</span></div><div class="metric"><strong>17</strong><span>категорий</span></div><div class="metric"><strong>3</strong><span>рекомендации</span></div></div></section>', unsafe_allow_html=True)

    try:
        catalogue = _load_catalogue()
    except Exception:
        st.error("Не удалось загрузить каталог подрядчиков. Проверьте файл данных и попробуйте снова.")
        return

    cities = _catalog_values(catalogue, "city")
    categories = _catalog_values(catalogue, "categories")
    formats = _catalog_values(catalogue, "event_formats")
    languages = _catalog_values(catalogue, "languages")
    st.markdown('<div class="section-label">01 / ПАРАМЕТРЫ СОБЫТИЯ</div><div class="section-heading">Расскажите, что вам нужно</div>', unsafe_allow_html=True)
    with st.form("recommendation_form"):
        left, right = st.columns(2)
        with left:
            city = st.selectbox("Город", cities)
            event_format = st.selectbox("Формат мероприятия", formats)
            event_date = st.date_input("Дата мероприятия", value=SUPPORTED_START, min_value=SUPPORTED_START, max_value=SUPPORTED_END, help="Каталог содержит доступность с 23 сентября по 31 декабря 2026 года.")
        with right:
            category = st.selectbox("Категория подрядчика", categories)
            budget = st.number_input("Бюджет, ₸", min_value=1, value=500_000, step=50_000)
            duration_enabled = st.checkbox("Указать длительность")
            duration = st.number_input("Длительность, часов", min_value=0.5, value=4.0, step=0.5) if duration_enabled else None
        language = st.selectbox("Язык (необязательно)", ["Не указывать", *languages])
        preferences = st.text_area("Пожелания (необязательно)", placeholder="Например: современный ведущий, лёгкий юмор, без банальных конкурсов")
        st.markdown('<div class="hint">Пожелания используются для смыслового сравнения уже подходящих по обязательным условиям кандидатов.</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Подобрать подрядчиков →", type="primary", use_container_width=True)

    if not submitted:
        st.markdown('<div class="method-strip"><div class="method-title">Сначала ограничения. Затем смысл.</div><div class="method-copy">Занятые, слишком дорогие и неподходящие по формату кандидаты исключаются до AI-ранжирования. При недоступности embedding API система использует детерминированный TF-IDF fallback.</div></div>', unsafe_allow_html=True)
        return

    query = SearchQuery(city=city, event_date=event_date.isoformat(), event_format=event_format, category=category, budget_kzt=int(budget), duration_hours=float(duration) if duration is not None else None, language=None if language == "Не указывать" else language, preferences=preferences.strip() or None)
    try:
        with st.spinner("Проверяем доступность и подбираем варианты…"):
            response = recommend_contractors(query, contractors=catalogue, use_openai=True)
        _show_results(response)
        st.markdown('<div class="method-strip"><div class="method-title">Сначала ограничения. Затем смысл.</div><div class="method-copy">Занятые, слишком дорогие и неподходящие по формату кандидаты исключаются до AI-ранжирования. При недоступности embedding API система использует детерминированный TF-IDF fallback.</div></div>', unsafe_allow_html=True)
    except Exception:
        st.error("Не удалось выполнить подбор. Проверьте введённые данные и попробуйте ещё раз.")


if __name__ == "__main__":
    main()
