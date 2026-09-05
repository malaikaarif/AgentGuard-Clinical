"""
AgentGuard-Clinical — agent chain, built incrementally.

Phase 3, step 4 (complete chain):
intake -> classifier -> reasoning -> explainability

All four agents now run end-to-end on one image. The next phase
(not yet built here) is the audit layer: comparing reasoning_text's
claimed region against where the Grad-CAM heatmap actually activated.
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from model.classifier_agent import classify_image
from model.explainability_agent import generate_explainability

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Check that .env exists in the project "
        "root and contains GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-3.6-flash"


# ---- Shared state passed between all agents in the graph ----
class PipelineState(TypedDict):
    image_path: str
    diagnosis: Optional[str]
    confidence: Optional[float]
    logits: Optional[list]
    class_names: Optional[list]
    reasoning_text: Optional[str]
    heatmap_path: Optional[str]
    error: Optional[str]


# ---- Node 1: intake ----
def intake_node(state: PipelineState) -> PipelineState:
    if not os.path.exists(state["image_path"]):
        return {**state, "error": f"Image not found: {state['image_path']}"}
    print(f"[intake] Received image: {state['image_path']}")
    return state


# ---- Node 2: classifier ----
def classifier_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
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


# ---- Node 3: reasoning ----
def reasoning_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    class_probs = dict(zip(state["class_names"], state["logits"]))
    probs_str = ", ".join(f"{k}: {v:.4f}" for k, v in class_probs.items())

    prompt = f"""You are assisting a radiologist reviewing a brain MRI scan.
A classifier model produced the following prediction:

Predicted diagnosis: {state['diagnosis']}
Confidence: {state['confidence']:.4f}
Full class probabilities: {probs_str}

Write a brief (3-4 sentence) differential diagnosis justification.
Explicitly mention what visual features on a brain MRI would typically
support this diagnosis (e.g. location, shape, signal characteristics),
so this can later be checked against an explainability heatmap.
Be direct and clinical in tone. If confidence is not high, note that
uncertainty explicitly rather than overstating certainty."""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        reasoning_text = response.text.strip()
    except Exception as e:
        return {**state, "error": f"Reasoning agent failed: {str(e)}"}

    print(f"[reasoning] {reasoning_text}\n")

    return {**state, "reasoning_text": reasoning_text}


# ---- Node 4: explainability ----
def explainability_node(state: PipelineState) -> PipelineState:
    """
    Runs Grad-CAM on the same image and saves the heatmap overlay.
    Doesn't yet compare it against reasoning_text's claims — that's
    the audit layer, built as a separate next step.
    """
    if state.get("error"):
        return state

    try:
        result = generate_explainability(state["image_path"])
    except Exception as e:
        return {**state, "error": f"Explainability agent failed: {str(e)}"}

    print(f"[explainability] Saved heatmap to: {result['heatmap_path']}")

    return {**state, "heatmap_path": result["heatmap_path"]}


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("intake", intake_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("explainability", explainability_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", "reasoning")
    graph.add_edge("reasoning", "explainability")
    graph.add_edge("explainability", END)

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
        "reasoning_text": None,
        "heatmap_path": None,
        "error": None,
    }

    final_state = app.invoke(initial_state)
    print("\n=== FINAL STATE ===")
    print(final_state)