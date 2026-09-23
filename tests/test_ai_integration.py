"""Real CSV -> real backend filters -> real AI service; only OpenAI transport is fake."""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
from datetime import date
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request, urlopen

import httpx
from fastapi.testclient import TestClient

from Backend.ai_facts import build_evidence
from Backend.ai_service import clear_cache, recommend_contractors
from Backend.dataset import load_contractors
from Backend.filtering import filter_contractors
from Backend.main import app
from Backend.schemas import SearchRequest
from ai_test_support import choice, response_payload, transport


class AIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.people = load_contractors()
        self.by_id = {p.id: p for p in self.people}
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        env = patch.dict(os.environ, {"AI_MODE": "openai", "OPENAI_API_KEY": "offline-test-key",
                                     "OPENAI_MODEL": "gpt-4o-mini", "AI_CACHE_TTL_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)
        guard = patch("httpx.AsyncHTTPTransport.handle_async_request", side_effect=AssertionError("Unexpected network"))
        guard.start()
        self.addCleanup(guard.stop)
        clear_cache()
        self.sent = []

    def success(self, request):
        data = json.loads(json.loads(request.content)["input"])
        self.sent.append(data)
        order = SearchRequest.model_validate(data["order"])
        choices = [choice(self.by_id[p["contractor_id"]], order) for p in reversed(data["candidates"])]
        return httpx.Response(200, json=response_payload(choices))

    def test_dense_category_all_candidates_reach_ai(self):
        expected = [p for p in self.people if "Ведущий" in p.categories]
        self.assertEqual(len(expected), 15)
        with transport(self.success), patch("Backend.main.recommend_contractors", wraps=recommend_contractors) as spy:
            response = self.client.post("/api/contractors/search", json={"categories": ["Ведущий"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual({p.id for p in spy.call_args.args[1]}, {p.id for p in expected})
        result = response.json()["recommendations"]
        self.assertEqual([r["contractor"]["id"] for r in result],
                         [p.id for p in sorted(expected, key=lambda p: (p.price_from_kzt, p.id))[:3]])
        self.assertTrue(all(set(r) == {"contractor", "reason"} for r in result))
        self.assertTrue(all("Базовое" not in r["reason"] for r in result))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.sent[0]["candidates"]), 3)

    def test_rare_one_and_two_without_padding(self):
        for city, count in (("Астана", 1), ("Алматы", 2)):
            expected = {p.id for p in self.people if "Флорист" in p.categories and p.city == city}
            self.assertEqual(len(expected), count)
            with self.subTest(city=city), transport(self.success):
                response = self.client.post("/api/contractors/search", json={"categories": ["Флорист"], "city": city})
                self.assertEqual(response.status_code, 200)
                result = response.json()["recommendations"]
                self.assertEqual({r["contractor"]["id"] for r in result}, expected)
                self.assertTrue(all("Базовое" not in r["reason"] for r in result))

    def test_two_empty_business_outcomes_do_not_call_ai(self):
        self.assertTrue(any(p.city == "Зарубежье" for p in self.people))
        self.assertFalse(any(p.city == "Зарубежье" and "Флорист" in p.categories for p in self.people))
        self.assertTrue(all(p.price_from_kzt > 0 for p in self.people))
        with patch("Backend.ai_service.httpx.AsyncClient") as client:
            for body in ({"city": "Зарубежье", "categories": ["Флорист"]}, {"max_price_kzt": 0}):
                response = self.client.post("/api/contractors/search", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["matched_candidates"], 0)
                self.assertEqual(response.json()["recommendations"], [])
            client.assert_not_called()

    def test_two_real_dates_busy_person_never_selected(self):
        person = self.by_id["HK-44733"]
        self.assertIn(date(2026, 9, 23), person.busy_dates)
        self.assertNotIn(date(2026, 9, 24), person.busy_dates)
        groups = []
        for day in (23, 24):
            order = SearchRequest(categories=["Ведущий"], city="Алматы", event_date=date(2026, 9, day))
            candidates = filter_contractors(self.people, order)
            groups.append({p.id for p in candidates})
            with transport(self.success), patch("Backend.main.recommend_contractors", wraps=recommend_contractors) as spy:
                response = self.client.post("/api/contractors/search", json=order.model_dump(mode="json"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual({p.id for p in spy.call_args.args[1]}, groups[-1])
            for card in response.json()["recommendations"]:
                self.assertNotIn(order.event_date, self.by_id[card["contractor"]["id"]].busy_dates)
            for sent in self.sent[-1]["candidates"]:
                self.assertIn(str(order.event_date), sent["evidence"]["date"])
        self.assertNotEqual(*groups)
        self.assertNotIn(person.id, groups[0])
        self.assertIn(person.id, groups[1])

    def test_real_catalog_semantic_and_fallback_quotes(self):
        order = SearchRequest(categories=["Фотограф"], wishes="Документальная фотография")
        candidates = filter_contractors(self.people, order)
        # This phrase and profile exist in the source catalog, not in a generated response.
        documentary = self.by_id["HK-61323"]
        self.assertIn("фотожурнализм", documentary.description)
        with transport(lambda request: httpx.Response(503)):
            result = recommend_contractors(order, candidates)
        self.assertEqual(result[0].contractor.id, documentary.id)
        for card in result:
            evidence = build_evidence(order, card.contractor)
            quote = next((v for k, v in evidence.items() if k.startswith("wish:")), evidence.get("profile"))
            self.assertIn("Базовое объяснение без ИИ", card.reason)
            self.assertIn(f"от {card.contractor.price_from_kzt}", card.reason)
            if quote:
                self.assertIn(quote, card.contractor.description)
                self.assertIn(quote, card.reason)

    def test_mock_backend_integration_no_api(self):
        with patch.dict(os.environ, {"AI_MODE": "mock", "OPENAI_API_KEY": "", "OPENAI_MODEL": ""}):
            with patch("Backend.ai_service.httpx.AsyncClient") as client:
                response = self.client.post("/api/contractors/search", json={"categories": ["Флорист"], "city": "Астана"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["recommendations"]), 1)
                self.assertIn("[MOCK — тестовый ответ]", response.json()["recommendations"][0]["reason"])
                client.assert_not_called()

    def test_catalog_excerpt_validation_regression(self):
        from Backend.ai_service import parse_selection
        for person in self.people:
            with self.subTest(contractor_id=person.id):
                result = parse_selection(response_payload([choice(person)]), [person])
                self.assertEqual(result[0].contractor, person)

    def test_uvicorn_mock_startup_without_case_workaround(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        environment = dict(os.environ, AI_MODE="mock", OPENAI_API_KEY="", OPENAI_MODEL="")
        environment.pop("PYTHONCASEOK", None)
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "Backend.main:app", "--host", "127.0.0.1",
             "--port", str(port), "--no-access-log"], env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 8
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                        self.assertEqual(json.load(response), {"status": "ok"})
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("Local uvicorn did not start")
                    time.sleep(0.05)
            request = Request(f"http://127.0.0.1:{port}/api/contractors/search",
                              data=b"{}", headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=2) as response:
                result = json.load(response)
            self.assertEqual(len(result["recommendations"]), 3)
            self.assertTrue(all("MOCK" in item["reason"] for item in result["recommendations"]))
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
