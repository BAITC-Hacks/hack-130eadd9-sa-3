import json
import os
import unittest
from datetime import date
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from Backend.ai_service import AIServiceError, parse_selection, recommend_contractors
from Backend.dataset import load_contractors
from Backend.filtering import filter_contractors
from Backend.main import app
from Backend.schemas import Recommendation, SearchRequest
from ai_test_support import choice


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(os.environ, {"AI_MODE": "openai", "AI_CACHE_TTL_SECONDS": "0"})
        environment.start()
        self.addCleanup(environment.stop)
        guard = patch("httpx.AsyncHTTPTransport.handle_async_request", side_effect=AssertionError("Unexpected network"))
        guard.start()
        self.addCleanup(guard.stop)

    @classmethod
    def setUpClass(cls) -> None:
        cls.people = load_contractors()
        cls.client = TestClient(app)

    def test_dataset(self) -> None:
        self.assertEqual(len(self.people), 66)
        self.assertEqual(len({person.id for person in self.people}), 66)
        self.assertEqual(len(filter_contractors(self.people, SearchRequest())), 66)

    def test_combined_filters_and_boundaries(self) -> None:
        person = self.people[1].model_copy(update={
            "categories": ["Ведущий"], "city": "Алматы", "price_from_kzt": 500000,
            "languages": ["русский", "английский"], "event_formats": ["корпоратив"],
            "max_hours": 6, "busy_dates": [date(2026, 10, 1)],
        })
        order = SearchRequest(categories=["фотограф", " ВЕДУЩИЙ "], city="алматы",
                              min_price_kzt=500000, max_price_kzt=500000,
                              languages=["Русский", "английский"], event_format="корпоратив",
                              duration_hours=6, event_date=date(2026, 10, 2))
        self.assertEqual(filter_contractors([person], order), [person])
        exclusions = [
            {"categories": ["Флорист"]}, {"city": "Астана"}, {"price_from_kzt": 500001},
            {"price_from_kzt": 499999}, {"languages": ["русский"]},
            {"event_formats": ["свадьба"]}, {"max_hours": 5}, {"max_hours": None},
            {"busy_dates": [date(2026, 10, 2)]},
        ]
        for change in exclusions:
            with self.subTest(change=change):
                self.assertEqual(filter_contractors([person.model_copy(update=change)], order), [])

    def test_routes_without_ai(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertIn("Ведущий", self.client.get("/api/filters").json()["categories"])
        result = self.client.post("/api/contractors/filter", json={"max_price_kzt": 500000})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(all(p["price_from_kzt"] <= 500000 for p in result.json()["candidates"]))
        with patch("Backend.main.recommend_contractors") as ai:
            result = self.client.post("/api/contractors/search", json={"max_price_kzt": 0})
            self.assertEqual(result.json()["recommendations"], [])
            ai.assert_not_called()

    def test_validation(self) -> None:
        for body in [{"max_price_kzt": -1}, {"min_price_kzt": 2, "max_price_kzt": 1},
                     {"event_date": "2027-01-01"}, {"event_date": "2026-09-22"},
                     {"duration_hours": 0}, {"unknown": True}, {"languages": [" "]}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/contractors/search", json=body).status_code, 422)

    def test_search_passes_only_filtered_candidates(self) -> None:
        def fake_ai(order, candidates):
            self.assertTrue(candidates)
            self.assertTrue(all(p.city == "Астана" for p in candidates))
            return [Recommendation(contractor=p, reason="Подходит по параметрам") for p in candidates[:3]]

        with patch("Backend.main.recommend_contractors", side_effect=fake_ai):
            response = self.client.post("/api/contractors/search", json={"city": "Астана"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["recommendations"]), 3)
        self.assertEqual(response.json()["matched_candidates"], 15)

    def test_ai_selection_validation(self) -> None:
        candidates = self.people[:3]
        valid = [choice(p) for p in candidates]

        def payload(choices):
            return {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps({"recommendations": choices})}]}]}

        self.assertEqual(len(parse_selection(payload(valid), candidates)), 3)
        self.assertEqual(len(parse_selection(payload(valid[:1]), candidates[:1])), 1)
        self.assertEqual(len(parse_selection(payload(valid[:2]), candidates[:2])), 2)
        for choices in [[], valid[:2], [valid[0]] * 3, valid + [valid[0]],
                        valid[:2] + [{"contractor_id": "unknown", "reason": "Причина"}],
                        valid[:2] + [{"contractor_id": candidates[2].id, "reason": " "}]]:
            with self.subTest(choices=choices), self.assertRaises(AIServiceError):
                parse_selection(payload(choices), candidates)
        for bad in [{}, {"status": "incomplete"}, {"status": "completed", "output": None},
                    {"status": "completed", "output": [{"type": "message", "content": [
                        {"type": "refusal", "refusal": "No"}]}]}]:
            with self.assertRaises(AIServiceError):
                parse_selection(bad, candidates)

    def test_ai_http_payload_and_errors(self) -> None:
        candidates = self.people[:1]
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        result = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"recommendations": [
                choice(candidates[0])]})}]}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}):
            with patch("Backend.ai_service.httpx.AsyncClient") as client_class:
                post = client_class.return_value.__aenter__.return_value.post
                post.return_value = httpx.Response(200, json=result, request=request)
                self.assertEqual(len(recommend_contractors(SearchRequest(), candidates)), 1)
                sent = post.call_args.kwargs["json"]
                self.assertEqual(len(json.loads(sent["input"])["candidates"]), 1)
                self.assertNotIn("test-key", json.dumps(sent))
                post.side_effect = httpx.ReadTimeout("timeout")
                fallback = recommend_contractors(SearchRequest(), candidates)
                self.assertIn("Базовое объяснение без ИИ", fallback[0].reason)
                self.assertEqual(fallback[0].contractor.id, candidates[0].id)
                post.side_effect = None
                post.return_value = httpx.Response(429, json={"error": "secret"}, request=request)
                fallback = recommend_contractors(SearchRequest(), candidates)
                self.assertIn("Базовое объяснение без ИИ", fallback[0].reason)
                self.assertNotIn("secret", fallback[0].reason)

    def test_api_failure_responses(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": ""}):
            self.assertEqual(self.client.post("/api/contractors/search", json={}).status_code, 503)
        for code in [502, 504]:
            with patch("Backend.main.recommend_contractors", side_effect=AIServiceError("Ошибка ИИ", code)):
                self.assertEqual(self.client.post("/api/contractors/search", json={}).status_code, code)
        with patch("Backend.main.load_contractors", side_effect=ValueError("Bad CSV")):
            self.assertEqual(self.client.post("/api/contractors/filter", json={}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
