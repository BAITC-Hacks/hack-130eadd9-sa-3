import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx

from backend.ai_service import AIServiceError, recommend_contractors
from backend.schemas import Contractor, Recommendation, SearchRequest


def make_candidate(index: int) -> Contractor:
    """Synthetic test fixture; never added to the catalog."""
    return Contractor(
        id=f"test-{index}", anon_name=f"Test contractor {index}",
        categories=["test-category"], city="test-city", city_imputed=False,
        synthetic=True, price_from_kzt=1000, price_imputed=False,
        event_formats=[], languages=[], max_hours=None, busy_dates=[],
        description="Synthetic profile for offline tests only.",
    )


class AIServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"AI_MODE": "mock"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch("backend.ai_service.httpx.Client",
                             side_effect=AssertionError("Network is forbidden"))
        self.client = self.network.start()
        self.addCleanup(self.network.stop)
        self.order = SearchRequest()

    def test_mock_candidate_counts_and_marker(self) -> None:
        for count in (0, 1, 2, 3, 5):
            with self.subTest(count=count):
                candidates = [make_candidate(i) for i in range(count)]
                result = recommend_contractors(self.order, candidates)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), min(3, count))
                self.assertEqual([r.contractor.id for r in result],
                                 [c.id for c in candidates[:3]])
                for recommendation in result:
                    self.assertIsInstance(recommendation, Recommendation)
                    self.assertIn("[MOCK — тестовый ответ]", recommendation.reason)
        self.client.assert_not_called()

    def test_mock_unique_ids_from_input(self) -> None:
        first, second, third, fourth = [make_candidate(i) for i in range(4)]
        for candidates in ([first, first.model_copy(), second, third, fourth],
                           [first, first, second, second]):
            with self.subTest(ids=[c.id for c in candidates]):
                ids = [r.contractor.id for r in recommend_contractors(self.order, candidates)]
                expected = list(dict.fromkeys(c.id for c in candidates))[:3]
                self.assertEqual(ids, expected)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(set(ids).issubset(c.id for c in candidates))

    def test_mock_does_not_mutate_input(self) -> None:
        candidates = [make_candidate(i) for i in (3, 1, 2, 0)]
        before = copy.deepcopy((self.order, candidates))
        recommend_contractors(self.order, candidates)
        self.assertEqual((self.order, candidates), before)

    def test_mock_without_credentials_or_network(self) -> None:
        self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertNotIn("OPENAI_MODEL", os.environ)
        self.assertEqual(len(recommend_contractors(self.order, [make_candidate(0)])), 1)
        self.client.assert_not_called()

    def test_missing_credentials_do_not_enable_mock(self) -> None:
        for mode in (None, "openai"):
            with self.subTest(mode=mode):
                settings = {} if mode is None else {"AI_MODE": mode}
                with patch.dict(os.environ, settings, clear=True):
                    with self.assertRaises(AIServiceError) as error:
                        recommend_contractors(self.order, [make_candidate(0)])
                    self.assertEqual(error.exception.status_code, 503)
        self.client.assert_not_called()

    def test_unknown_mode_fails_without_network(self) -> None:
        with patch.dict(os.environ, {"AI_MODE": "typo"}):
            with self.assertRaises(AIServiceError):
                recommend_contractors(self.order, [make_candidate(0)])
        self.client.assert_not_called()

    def test_empty_openai_input_needs_no_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(recommend_contractors(self.order, []), [])
        self.client.assert_not_called()

    def test_default_and_explicit_openai_with_fake_http(self) -> None:
        candidate = make_candidate(0)
        payload = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"recommendations": [
                {"contractor_id": candidate.id, "reason": "Offline stub response"}]})}]}]}
        for mode in (None, "openai"):
            with self.subTest(mode=mode):
                settings = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}
                if mode is not None:
                    settings["AI_MODE"] = mode
                with patch.dict(os.environ, settings, clear=True):
                    with patch("backend.ai_service.httpx.Client") as fake_client:
                        post = fake_client.return_value.__enter__.return_value.post
                        post.return_value = httpx.Response(
                            200, json=payload,
                            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
                        )
                        result = recommend_contractors(self.order, [candidate])
                        self.assertEqual(result, [Recommendation(
                            contractor=candidate, reason="Offline stub response")])
                        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
