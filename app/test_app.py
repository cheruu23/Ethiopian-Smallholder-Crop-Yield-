"""Smoke tests for the Flask routes; run with python app/test_app.py."""
import unittest

from app.app import app
from app.predictor import DEFAULTS


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_homepage_renders_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Crop Yield", response.data)
        self.assertIn(b"Planting month", response.data)

    def test_api_predicts_yield(self):
        response = self.client.post("/api/predict", json=DEFAULTS)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["yield_t_ha"], 0)
        self.assertGreater(result["revenue_birr"], 0)

    def test_api_rejects_invalid_payload(self):
        response = self.client.post("/api/predict", json=[])
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])


if __name__ == "__main__":
    unittest.main()
