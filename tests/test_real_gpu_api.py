import os
import unittest

import httpx


RUN_GPU_INTEGRATION = os.getenv("RUN_GPU_INTEGRATION") == "1"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8040")


@unittest.skipUnless(
    RUN_GPU_INTEGRATION,
    "Set RUN_GPU_INTEGRATION=1 to run the real GPU API test.",
)
class RealGPUAPITest(unittest.TestCase):
    def test_real_qlora_generation(self):
        response = httpx.post(
            f"{API_URL}/generate",
            json={
                "instruction": (
                    "Explain the difference between overfitting "
                    "and underfitting in machine learning."
                ),
                "context": "",
            },
            timeout=60.0,
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["model_mode"],
            "qlora_4bit_peft",
        )
        self.assertGreater(
            body["generated_tokens"],
            0,
        )
        self.assertTrue(
            body["text"].strip(),
        )


if __name__ == "__main__":
    unittest.main()