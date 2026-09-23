"""Existing synchronous AI entry point: select in Python, explain once with OpenAI."""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from time import monotonic
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from Backend import ai_cache, ai_facts, prompts
from Backend.ai_cache import clear_cache
from Backend.schemas import Contractor, Recommendation, SearchRequest

logger = logging.getLogger(__name__)
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=900)]


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    contractor_id: str
    reason: Reason
    evidence_refs: list[str] = Field(min_length=1, max_length=12)


class ModelExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    recommendations: list[ModelChoice] = Field(min_length=1, max_length=3)


class AIServiceError(Exception):
    """Safe public error; never include provider bodies, keys or user input."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class TemporaryAIError(Exception):
    """Only a fixed, non-sensitive error category may be stored here."""


def numeric_setting(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError
        return value
    except ValueError:
        raise AIServiceError(f"Некорректная настройка {name}.", 503) from None


def validate_reason(choice: ModelChoice, person: Contractor, facts: dict[str, str]) -> None:
    refs = choice.evidence_refs
    if len(set(refs)) != len(refs) or not set(refs).issubset(facts):
        raise ValueError("evidence")
    reason = choice.reason
    lower = reason.casefold().replace("ё", "е")
    cited = " ".join(facts[ref] for ref in refs).casefold().replace("ё", "е")
    if not re.search(r"[а-я]", lower) or ai_facts.UNSAFE.search(reason):
        raise ValueError("unsafe_or_non_russian")
    if any(word in lower for word in ("mock", "базовое объяснение", "безлимит", "неограниченн", "забронирован")):
        raise ValueError("unsupported_claim")
    sentences = re.split(r"[.!?]+(?:\s|$)", re.sub(r"«[^»]*»", "цитата", reason))
    if len([part for part in sentences if part.strip()]) > 2:
        raise ValueError("too_many_sentences")
    # Check numbers against cited facts, not arbitrary numbers in the user's wishes.
    def numbers(text: str) -> set[str]:
        text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))", "", text)
        return set(re.findall(r"\d+(?:[.,]\d+)?", text))
    if not numbers(lower).issubset(numbers(cited)):
        raise ValueError("unsupported_number")
    if re.search(r"цен|стоим|тенге|₸", lower):
        if "price" not in refs or not re.search(r"\bот\s+\d", lower):
            raise ValueError("price_not_starting")
        if re.search(r"за\s+час|почас|окончательная цена|итоговая цена", lower):
            raise ValueError("price_not_event")
        if str(person.price_from_kzt) not in re.sub(r"\s", "", lower):
            raise ValueError("wrong_price")
    for language in ("английск", "казахск", "русск", "немецк", "французск", "китайск"):
        if (re.search(r"\b" + language + r"(?:ий|ом|и)\b", lower)
                and not any(language in item.casefold() for item in person.languages)):
            raise ValueError("unsupported_language")
    if re.search(r"свобод|доступен|доступна", lower) and "date" not in refs:
        raise ValueError("unsupported_availability")
    if re.search(r"\bчас(?:а|ов)?\b|\bдлительн", lower) and "duration" not in refs:
        raise ValueError("unsupported_duration")
    stated_city = re.search(r"\b[Гг]ород(?:е)?\b\s*[:—-]?\s*([А-ЯЁ][а-яё-]+)", reason)
    if stated_city:
        if stated_city.group(1).casefold() != person.city.casefold():
            raise ValueError("wrong_city")
    for claim in ("опыт", "отзыв", "оборудован", "вместим", "наград"):
        if claim in lower and claim not in cited:
            raise ValueError("unsupported_attribute")
    profile_refs = [ref for ref in facts if ref.startswith("wish:") or ref == "profile"]
    if profile_refs:
        used = [ref for ref in refs if ref in profile_refs]
        if not used or not re.search(r"в профиле|по описанию|описании профиля", lower):
            raise ValueError("missing_profile_attribution")
        words = re.findall(r"[а-я]{5,}", " ".join(facts[ref] for ref in used).casefold())
        if not any(word in lower for word in words):
            raise ValueError("generic_reason")


def parse_selection(payload: dict, candidates: list[Contractor],
                    order: SearchRequest | None = None) -> list[Recommendation]:
    """Candidates here are the already selected group, never the full input pool."""
    try:
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            raise ValueError("incomplete")
        parts = [part for item in payload["output"] if item.get("type") == "message"
                 for part in item.get("content", [])]
        if any(part.get("type") == "refusal" for part in parts):
            raise ValueError("refusal")
        text = "".join(part["text"] for part in parts if part.get("type") == "output_text")
        choices = ModelExplanation.model_validate_json(text).recommendations
        by_id = {choice.contractor_id: choice for choice in choices}
        expected = {person.id for person in candidates}
        if len(choices) != len(expected) or len(by_id) != len(choices) or set(by_id) != expected:
            raise ValueError("invalid_ids")
        result = []
        for person in candidates:
            choice = by_id[person.id]
            validate_reason(choice, person, ai_facts.build_evidence(order or SearchRequest(), person))
            result.append(Recommendation(contractor=person,
                                         reason=ai_facts.with_caveats(choice.reason, person)))
        return result
    except (ValueError, KeyError, TypeError, AttributeError, ValidationError):
        raise AIServiceError("ИИ вернул непригодное объяснение.") from None


def build_payload(order: SearchRequest, selected: list[Contractor], model: str) -> dict:
    return {
        "model": model, "store": False, "max_output_tokens": 1800,
        "instructions": prompts.SYSTEM_PROMPT,
        "input": json.dumps({
            "order": order.model_dump(mode="json"),
            "candidates": [{"contractor_id": person.id,
                            "evidence": ai_facts.build_evidence(order, person),
                            "synthetic": person.synthetic, "city_imputed": person.city_imputed,
                            "price_imputed": person.price_imputed} for person in selected],
        }, ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": "contractor_explanations",
                            "strict": True, "schema": ModelExplanation.model_json_schema()}},
    }


def check_http(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    try:
        error = response.json().get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        kind = error.get("type", "") if isinstance(error, dict) else ""
    except (ValueError, AttributeError):
        code, kind = "", ""
    code = code if isinstance(code, str) else ""
    kind = kind if isinstance(kind, str) else ""
    if kind == "insufficient_quota" or code in {
        "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
        "usage_limit_reached", "organization_usage_limit_exceeded",
    }:
        raise AIServiceError("Исчерпана квота OpenAI; проверьте оплату и лимиты.", 503)
    if status in (401, 403):
        raise AIServiceError("OpenAI отклонил доступ; проверьте ключ и права проекта.", 503)
    if status in (400, 404, 422):
        raise AIServiceError("Проверьте модель OpenAI и поддержку параметров запроса.", 503)
    if status in (408, 409, 429) or status >= 500:
        raise TemporaryAIError("temporary_http")
    raise AIServiceError("OpenAI отклонил запрос.", 502)


async def request_model(payload: dict, api_key: str, budget: float) -> dict:
    # Single attempt: no SDK/transport retries, sleeps, or JSON repair requests.
    # Cancellation covers the complete request, including a slow response body.
    async def send() -> dict:
        async with httpx.AsyncClient(timeout=budget) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"}, json=payload,
            )
            check_http(response)
            try:
                result = response.json()
            except ValueError:
                raise TemporaryAIError("invalid_json") from None
            if not isinstance(result, dict):
                raise TemporaryAIError("invalid_json")
            return result
    try:
        return await asyncio.wait_for(send(), timeout=budget)
    except (TimeoutError, httpx.TimeoutException):
        raise TemporaryAIError("timeout") from None
    except httpx.RequestError:
        raise TemporaryAIError("network") from None


def cache_key(order: SearchRequest, selected: list[Contractor], model: str, api_key: str) -> str:
    data = {
        "order": order.model_dump(mode="json"),
        "selected": [person.model_dump(mode="json") for person in selected],
        "model": model, "mode": "openai", "rules": ai_facts.RULES_VERSION,
        "prompt_version": prompts.PROMPT_VERSION, "prompt": prompts.SYSTEM_PROMPT,
        # Partition accounts without storing/logging the credential itself.
        "account": hashlib.sha256(api_key.encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def recommend_contractors(order: SearchRequest, candidates: list[Contractor]) -> list[Recommendation]:
    started = monotonic()
    if not candidates:
        return []
    mode = os.getenv("AI_MODE", "openai").strip().lower()
    if mode not in {"mock", "openai"}:
        raise AIServiceError("AI_MODE должен быть openai или mock.", 503)
    try:
        unique = ai_facts.unique_candidates(candidates)
    except ValueError:
        raise AIServiceError("Конфликтующие записи подрядчика с одинаковым ID.", 422) from None
    if mode == "mock":
        return [Recommendation(contractor=person, reason=(
            "[MOCK — тестовый ответ] Подрядчик взят из переданного списка "
            "в порядке входных данных. Это тестовая заглушка без оценки ИИ."
        )) for person in unique[:3]]
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        raise AIServiceError("Настройте OPENAI_API_KEY и OPENAI_MODEL в .env на сервере.", 503)
    budget = numeric_setting("AI_TIMEOUT_SECONDS", 8, 0.01, 60)
    ttl = numeric_setting("AI_CACHE_TTL_SECONDS", 300, 0, 3600)
    selected = ai_facts.select_candidates(order, unique)
    key = cache_key(order, selected, model, api_key)
    epoch = ai_cache.generation()
    cached = ai_cache.get(key) if ttl else None
    if cached is not None:
        logger.info("ai mode=model cache_hit=true elapsed_ms=%.1f", (monotonic() - started) * 1000)
        return [Recommendation(contractor=p, reason=r) for p, r in zip(selected, cached)]
    payload = build_payload(order, selected, model)
    try:
        remaining = budget - (monotonic() - started)
        if remaining <= 0:
            raise TemporaryAIError("timeout")
        result = asyncio.run(request_model(payload, api_key, remaining))
        try:
            recommendations = parse_selection(result, selected, order)
        except AIServiceError:
            raise TemporaryAIError("invalid_explanation") from None
        if monotonic() - started >= budget:
            raise TemporaryAIError("timeout")
    except TemporaryAIError as error:
        logger.warning("ai mode=fallback category=%s cache_hit=false elapsed_ms=%.1f",
                       str(error), (monotonic() - started) * 1000)
        return [Recommendation(contractor=p, reason=ai_facts.fallback_reason(order, p)) for p in selected]
    except AIServiceError:
        logger.warning("ai mode=error category=configuration_or_access elapsed_ms=%.1f",
                       (monotonic() - started) * 1000)
        raise
    ai_cache.put(key, tuple(r.reason for r in recommendations), ttl, epoch)
    usage = result.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    logger.info("ai mode=model cache_hit=false elapsed_ms=%.1f total_tokens=%s",
                (monotonic() - started) * 1000, tokens if type(tokens) is int else "unknown")
    return recommendations
