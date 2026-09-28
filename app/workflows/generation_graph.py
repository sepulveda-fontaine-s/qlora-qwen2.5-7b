from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.model_service import ModelService


class GenerationState(TypedDict, total=False):
    request: GenerationRequest
    response: GenerationResponse
    output_valid: bool


def build_generation_graph(model_service: ModelService):
    def generate(state: GenerationState) -> GenerationState:
        response = model_service.generate(state["request"])

        return {
            **state,
            "response": response,
        }

    def validate_output(state: GenerationState) -> GenerationState:
        response = state["response"]

        return {
            **state,
            "output_valid": bool(response.text.strip()),
        }

    def route_after_validation(state: GenerationState) -> str:
        if state["output_valid"]:
            return "complete"

        return "invalid_output"

    def invalid_output(state: GenerationState) -> GenerationState:
        raise RuntimeError("Model generated an empty response.")

    graph = StateGraph(GenerationState)

    graph.add_node("generate", generate)
    graph.add_node("validate_output", validate_output)
    graph.add_node("invalid_output", invalid_output)

    graph.set_entry_point("generate")

    graph.add_edge("generate", "validate_output")

    graph.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {
            "complete": END,
            "invalid_output": "invalid_output",
        },
    )

    graph.add_edge("invalid_output", END)

    return graph.compile()