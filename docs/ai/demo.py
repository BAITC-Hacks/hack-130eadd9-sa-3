"""Offline demonstration on the real catalog; outbound HTTP is always replaced."""
import json
import os
from time import perf_counter
from unittest.mock import patch

import httpx

from Backend.ai_service import clear_cache, recommend_contractors
from Backend.dataset import load_contractors
from Backend.filtering import filter_contractors
from Backend.schemas import SearchRequest


def main() -> None:
    order = SearchRequest(categories=["Фотограф"], wishes="Документальная фотография")
    candidates = filter_contractors(load_contractors(), order)
    original_client = httpx.AsyncClient
    calls = []

    def fake_response(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        inputs = json.loads(json.loads(request.content)["input"])
        choices = []
        for item in inputs["candidates"]:
            evidence = item["evidence"]
            ref = next((key for key in evidence if key.startswith("wish:")), "profile")
            reason = f"По описанию, «{evidence[ref]}» — особенность этого профиля."
            choices.append({"contractor_id": item["contractor_id"], "reason": reason, "evidence_refs": [ref]})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"recommendations": choices})}]}]})

    outputs = []
    with patch.dict(os.environ, {"AI_MODE": "mock", "OPENAI_API_KEY": "offline-test-key",
                                 "OPENAI_MODEL": "gpt-4o-mini", "AI_CACHE_TTL_SECONDS": "300"}):
        clear_cache()
        with patch("httpx.AsyncHTTPTransport.handle_async_request", side_effect=AssertionError("Network forbidden")):
            for mode in ("mock", "fallback", "fixture", "cache"):
                os.environ["AI_MODE"] = "mock" if mode == "mock" else "openai"
                handler = (lambda request: httpx.Response(503)) if mode == "fallback" else fake_response
                with patch("Backend.ai_service.httpx.AsyncClient", side_effect=lambda **kwargs:
                           original_client(transport=httpx.MockTransport(handler), **kwargs)):
                    started = perf_counter()
                    result = recommend_contractors(order, candidates)
                    elapsed = (perf_counter() - started) * 1000
                outputs.append({"origin": mode, "elapsed_ms": round(elapsed, 3),
                                "recommendations": [{"contractor_id": r.contractor.id, "reason": r.reason} for r in result]})
    print(json.dumps({"real_api_requests": 0, "successful_fake_http_requests": len(calls), "results": outputs},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
