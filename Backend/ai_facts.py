"""Conservative lexical rules with exact source excerpts; no second filter."""

import re

from Backend.schemas import CALENDAR_END, CALENDAR_START, Contractor, SearchRequest

RULES_VERSION = "lexical-1"
# One point per concept explicitly present in both wishes and description.
CONCEPTS = {
    "reportage": r"\bрепортаж\w*",
    "documentary": r"\b(?:документаль\w*|фотожурнализм\w*)",
    "individual_script": r"\bиндивидуаль\w*\s+сценари\w*",
    "live_music": r"\bжив\w*\s+музык\w*",
    "intelligent_humor": r"\bинтеллигентн\w*\s+юмор\w*",
    "calm_style": r"\bспокойн\w*\s+(?:стил\w*|ведени\w*)",
    "floral_design": r"\bцветочн\w*\s+оформлен\w*",
    "natural_emotions": r"\b(?:искренн\w*\s+эмоци\w*|жив\w*\s+кадр\w*)",
}
UNSAFE = re.compile(
    r"ignore|instruction|system|prompt|api.?key|secret|sk-\w|"
    r"игнорир|инструкц|промпт|раскро|секрет|ключ|поставь|верни\s+id|"
    r"лучши|\bтоп\b|рейтинг|гарантир", re.I,
)
NEGATION = re.compile(r"\b(?:не|нет|без|никак\w*|not|without)\b", re.I)


def fragments(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?\n;•]+", text)
            if 12 <= len(part.strip()) <= 260 and not UNSAFE.search(part)]


def matched_evidence(order: SearchRequest, person: Contractor) -> dict[str, str]:
    evidence = {}
    wish_parts = [p for p in re.split(r"[.!?\n;•]+", order.wishes)
                  if not NEGATION.search(p) and not UNSAFE.search(p)]
    description_parts = [p for p in fragments(person.description) if not NEGATION.search(p)]
    for name, pattern in CONCEPTS.items():
        if any(re.search(pattern, part, re.I) for part in wish_parts):
            quote = next((part for part in description_parts if re.search(pattern, part, re.I)), None)
            if quote:
                evidence[f"wish:{name}"] = quote
    return evidence


def unique_candidates(candidates: list[Contractor]) -> list[Contractor]:
    by_id: dict[str, Contractor] = {}
    for person in candidates:
        previous = by_id.get(person.id)
        if previous is not None and previous != person:
            raise ValueError("conflicting_candidate_id")
        by_id[person.id] = person
    return list(by_id.values())


def select_candidates(order: SearchRequest, candidates: list[Contractor]) -> list[Contractor]:
    return sorted(unique_candidates(candidates), key=lambda p: (
        -len(matched_evidence(order, p)), p.price_from_kzt, p.id,
    ))[:3]


def build_evidence(order: SearchRequest, person: Contractor) -> dict[str, str]:
    facts = {
        "city": f"Город в каталоге: {person.city}",
        "categories": "Категории: " + ", ".join(person.categories),
        "price": f"Начальная цена за мероприятие от {person.price_from_kzt} тенге, не окончательная смета",
        "languages": "Языки в каталоге: " + ", ".join(person.languages),
        "formats": "Форматы в каталоге: " + ", ".join(person.event_formats),
    }
    if person.max_hours is not None:
        facts["duration"] = f"Указанная предельная длительность: {person.max_hours:g} ч"
    if (order.event_date and CALENDAR_START <= order.event_date <= CALENDAR_END
            and order.event_date not in person.busy_dates):
        facts["date"] = f"По календарю каталога свободен {order.event_date}, это не бронь"
    matches = matched_evidence(order, person)
    facts.update(matches)
    quotes = [part for part in fragments(person.description) if re.search(r"[а-яё]{5,}", part, re.I)]
    if not matches and quotes:
        # Prefer a concrete service/style excerpt over an introductory sentence.
        facts["profile"] = next((part for part in quotes if any(
            re.search(pattern, part, re.I) for pattern in CONCEPTS.values()
        )), quotes[0])
    return facts


def with_caveats(reason: str, person: Contractor) -> str:
    parts = []
    if person.synthetic:
        parts.append("синтетический профиль")
    if person.city_imputed:
        parts.append("город восстановлен в данных")
    if person.price_imputed:
        parts.append("цена оценочная")
    return reason.rstrip(". !") + ("; " + "; ".join(parts) if parts else "") + "."


def fallback_reason(order: SearchRequest, person: Contractor) -> str:
    facts = build_evidence(order, person)
    quote = next((value for ref, value in facts.items() if ref.startswith("wish:")),
                 facts.get("profile"))
    context = f"{', '.join(person.categories)}; город в каталоге — {person.city}"
    if order.event_format and order.event_format.casefold() in {f.casefold() for f in person.event_formats}:
        context += f"; указан запрошенный формат «{order.event_format}»"
    if "date" in facts:
        context += f"; {facts['date']}"
    reason = (f"Базовое объяснение без ИИ: {context}, "
              f"начальная цена от {person.price_from_kzt} тенге за мероприятие, не итоговая смета. ")
    if quote:
        connection = " — это совпадает с пожеланием" if any(r.startswith("wish:") for r in facts) else ""
        reason += f"В описании профиля: «{quote}»{connection}"
    else:
        reason += "Конкретики в описании недостаточно; выбор основан на цене и ID среди допустимых кандидатов"
    return with_caveats(reason, person)
