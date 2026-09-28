import unittest

from fastapi.testclient import TestClient

from app.main import app


class APIWorkflowTest(unittest.TestCase):
    def test_generate_through_langgraph(self):
        with TestClient(app) as client:
            response = client.post(
                "/generate",
                json={
                    "instruction": "Explain overfitting.",
                    "context": "",
                },
            )

            self.assertEqual(response.status_code, 200)

            body = response.json()

            self.assertEqual(body["model_mode"], "mock")
            self.assertIn("Explain overfitting.", body["text"])
            self.assertGreaterEqual(body["generated_tokens"], 1)

    def test_metrics_endpoint(self):
        with TestClient(app) as client:
            client.post(
                "/generate",
                json={
                    "instruction": "Explain overfitting.",
                    "context": "",
                },
            )

            response = client.get("/metrics")

            self.assertEqual(response.status_code, 200)

            body = response.text

            self.assertIn("llm_requests_total", body)
            self.assertIn("llm_inference_latency_seconds", body)
            self.assertIn("llm_generated_tokens_total", body)

    def test_ready_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/ready")

            self.assertEqual(response.status_code, 200)

            body = response.json()

            self.assertEqual(body["status"], "ready")
            self.assertEqual(body["model_mode"], "mock")


if __name__ == "__main__":
    unittest.main()