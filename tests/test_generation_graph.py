import unittest

from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.mock_model_service import MockModelService
from app.services.model_service import ModelService
from app.workflows.generation_graph import build_generation_graph


class EmptyResponseModelService(ModelService):
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return GenerationResponse(
            text="",
            model_mode="test_empty",
            latency_seconds=0.0,
            generated_tokens=0,
        )


class GenerationGraphTest(unittest.TestCase):
    def test_valid_mock_generation(self):
        model_service = MockModelService()
        graph = build_generation_graph(model_service)

        request = GenerationRequest(
            instruction="Explain overfitting.",
            context=None,
        )

        result = graph.invoke(
            {
                "request": request,
            }
        )

        self.assertTrue(result["output_valid"])
        self.assertEqual(
            result["response"].model_mode,
            "mock",
        )
        self.assertIn(
            "Explain overfitting.",
            result["response"].text,
        )

    def test_empty_output_raises_error(self):
        model_service = EmptyResponseModelService()
        graph = build_generation_graph(model_service)

        request = GenerationRequest(
            instruction="Explain overfitting.",
            context=None,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Model generated an empty response.",
        ):
            graph.invoke(
                {
                    "request": request,
                }
            )


if __name__ == "__main__":
    unittest.main()