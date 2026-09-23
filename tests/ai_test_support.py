"""Offline fixtures and HTTP transport only; never call the public Internet."""
import json
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from Backend.ai_facts import build_evidence
from Backend.schemas import Contractor, SearchRequest


def make_candidate(index: int, **changes) -> Contractor:
    data = dict(id=f"test-{index}", anon_name=f"Test contractor {index}",
                categories=["Фотограф"], city="Алматы", city_imputed=False,
                synthetic=True, price_from_kzt=1000, price_imputed=False,
                event_formats=["свадьба"], languages=["русский"], max_hours=None,
                busy_dates=[], description="Снимаю репортажные фотографии мероприятий.")
    data.update(changes)
    return Contractor(**data)


def choice(person: Contractor, order: SearchRequest | None = None) -> dict:
    evidence = build_evidence(order or SearchRequest(), person)
    ref = next((r for r in evidence if r.startswith("wish:")), "profile")
    if ref in evidence:
        reason = f"По описанию, «{evidence[ref]}». Начальная цена от {person.price_from_kzt} тенге за мероприятие."
        refs = [ref, "price"]
    else:
        reason = f"Начальная цена от {person.price_from_kzt} тенге за мероприятие; конкретики в описании недостаточно."
        refs = ["price"]
    return dict(contractor_id=person.id, reason=reason, evidence_refs=refs)


def response_payload(choices: list[dict]) -> dict:
    return {"status": "completed", "usage": {"total_tokens": 120}, "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps(
            {"recommendations": choices}, ensure_ascii=False)}]},
    ]}


@contextmanager
def transport(handler):
    original_client = httpx.AsyncClient
    with patch("Backend.ai_service.httpx.AsyncClient", side_effect=lambda **kwargs:
               original_client(transport=httpx.MockTransport(handler), **kwargs)):
        yield
