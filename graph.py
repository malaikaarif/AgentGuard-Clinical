"""
AgentGuard-Clinical — full agent chain.

intake -> classifier -> reasoning -> explainability -> audit

Phase 4: reasoning agent's Gemini call is now wrapped with ARW's
retry-with-backoff, so a transient failure (e.g. a 429 rate limit,
which happened during real testing) gets retried instead of crashing
the whole pipeline.
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from model.classifier_agent import classify_image
from model.explainability_agent import generate_explainability
from model.audit_agent import describe_heatmap_location, check_consistency
from model.arw import retry_with_backoff

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
    heatmap_array: Optional[object]
    region_label: Optional[str]
    audit_verdict: Optional[str]
    audit_explanation: Optional[str]
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


# ---- Node 3: reasoning (now ARW-wrapped) ----
@retry_with_backoff(max_retries=3, base_delay=12)
def _call_gemini_reasoning(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()


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
        reasoning_text = _call_gemini_reasoning(prompt)
    except Exception as e:
        return {**state, "error": f"Reasoning agent failed after retries: {str(e)}"}

    print(f"[reasoning] {reasoning_text}\n")

    return {**state, "reasoning_text": reasoning_text}


# ---- Node 4: explainability ----
def explainability_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    try:
        result = generate_explainability(state["image_path"])
    except Exception as e:
        return {**state, "error": f"Explainability agent failed: {str(e)}"}

    print(f"[explainability] Saved heatmap to: {result['heatmap_path']}")

    return {
        **state,
        "heatmap_path": result["heatmap_path"],
        "heatmap_array": result["heatmap_array"],
    }


# ---- Node 5: audit (uses ARW-wrapped check_consistency internally) ----
def audit_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    try:
        location_info = describe_heatmap_location(state["heatmap_array"])
        result = check_consistency(state["reasoning_text"], location_info)
    except Exception as e:
        return {**state, "error": f"Audit agent failed after retries: {str(e)}"}

    print(f"[audit] Region: {location_info['region_label']}")
    print(f"[audit] Verdict: {result['verdict'].upper()} — {result['explanation']}\n")

    return {
        **state,
        "region_label": location_info["region_label"],
        "audit_verdict": result["verdict"],
        "audit_explanation": result["explanation"],
    }


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("intake", intake_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("explainability", explainability_node)
    graph.add_node("audit", audit_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", "reasoning")
    graph.add_edge("reasoning", "explainability")
    graph.add_edge("explainability", "audit")
    graph.add_edge("audit", END)

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
        "heatmap_array": None,
        "region_label": None,
        "audit_verdict": None,
        "audit_explanation": None,
        "error": None,
    }

    final_state = app.invoke(initial_state)

    printable_state = {k: v for k, v in final_state.items() if k != "heatmap_array"}
    print("\n=== FINAL STATE ===")
    print(printable_state)