"""
AgentGuard-Clinical — agent chain, built incrementally.

Phase 3, step 1: just two nodes wired together —
intake -> classifier

This proves the LangGraph plumbing works before we add
the reasoning and explainability agents on top.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from model.classifier_agent import classify_image


# ---- Shared state passed between all agents in the graph ----
class PipelineState(TypedDict):
    image_path: str
    diagnosis: Optional[str]
    confidence: Optional[float]
    logits: Optional[list]
    class_names: Optional[list]
    error: Optional[str]


# ---- Node 1: intake ----
def intake_node(state: PipelineState) -> PipelineState:
    """
    Just validates the image path exists and passes state through.
    Kept deliberately simple for now — this is where you'd later add
    metadata validation, file-type checks, etc.
    """
    import os
    if not os.path.exists(state["image_path"]):
        return {**state, "error": f"Image not found: {state['image_path']}"}
    print(f"[intake] Received image: {state['image_path']}")
    return state


# ---- Node 2: classifier ----
def classifier_node(state: PipelineState) -> PipelineState:
    """
    Calls the existing classify_image() function and merges its
    output into the pipeline state.
    """
    if state.get("error"):
        # Something already failed upstream — skip and pass the error through
        return state

    result = classify_image(state["image_path"])
    print(f"[classifier] Diagnosis: {result['diagnosis']} "
          f"(confidence: {result['confidence']:.4f})")

    return {
        **state,
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "logits": result["logits"],
        "class_names": result["class_names"],
    }


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("intake", intake_node)
    graph.add_node("classifier", classifier_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", END)

    return graph.compile()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graph.py <path_to_image>")
        sys.exit(1)

    app = build_graph()
    initial_state: PipelineState = {
        "image_path": sys.argv[1],
        "diagnosis": None,
        "confidence": None,
        "logits": None,
        "class_names": None,
        "error": None,
    }

    final_state = app.invoke(initial_state)
    print("\n=== FINAL STATE ===")
    print(final_state)