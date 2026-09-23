import json
import os

import httpx
from pydantic import ValidationError

from backend.prompts import SYSTEM_PROMPT
from backend.schemas import AISelection, Contractor, Recommendation, SearchRequest


class AIServiceError(Exception):
    """Безопасное сообщение для клиента без ключей и ответа провайдера."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_selection(payload: dict, candidates: list[Contractor]) -> list[Recommendation]:
    try:
        if payload.get("status") != "completed":
            raise ValueError("Incomplete response")
        parts = [part for item in payload["output"] if item.get("type") == "message"
                 for part in item.get("content", [])]
        if any(part.get("type") == "refusal" for part in parts):
            raise ValueError("Refusal")
        text = "".join(part["text"] for part in parts if part.get("type") == "output_text")
        selection = AISelection.model_validate_json(text)
        choices = selection.recommendations
        by_id = {person.id: person for person in candidates}
        ids = [choice.contractor_id for choice in choices]
        if len(ids) != min(3, len(candidates)) or len(set(ids)) != len(ids):
            raise ValueError("Invalid selection size or duplicate IDs")
        if any(candidate_id not in by_id for candidate_id in ids):
            raise ValueError("Unknown contractor ID")
        return [Recommendation(contractor=by_id[choice.contractor_id], reason=choice.reason)
                for choice in choices]
    except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as error:
        raise AIServiceError("ИИ вернул некорректный ответ. Повторите поиск.") from error


def recommend_contractors(order: SearchRequest, candidates: list[Contractor]) -> list[Recommendation]:
    if not candidates:
        return []
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        raise AIServiceError("Настройте OPENAI_API_KEY и OPENAI_MODEL в .env на сервере.", 503)
    # Передаем весь отфильтрованный набор, а не исходный CSV или первые три строки.
    payload = {
        "model": model,
        "store": False,
        "instructions": SYSTEM_PROMPT,
        "input": json.dumps({
            "order": order.model_dump(mode="json"),
            "candidates": [person.model_dump(mode="json") for person in candidates],
        }, ensure_ascii=False),
        "text": {"format": {
            "type": "json_schema", "name": "contractor_selection", "strict": True,
            "schema": AISelection.model_json_schema(),
        }},
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"}, json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.TimeoutException as error:
        raise AIServiceError("ИИ не ответил вовремя. Повторите поиск.", 504) from error
    except httpx.HTTPError as error:
        raise AIServiceError("Сервис ИИ недоступен. Проверьте настройки или повторите поиск.") from error
    except ValueError as error:
        raise AIServiceError("Сервис ИИ вернул некорректный JSON.") from error
    return parse_selection(result, candidates)
