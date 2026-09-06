"""
Audit agent — the core novel piece of AgentGuard-Clinical.

Checks whether the reasoning agent's stated justification (e.g. "the
sella turcica and suprasellar region support this diagnosis") is
actually consistent with where the Grad-CAM heatmap activated.

Honest scope note: this is a first-attempt heuristic, not a solved
problem. Matching free-text anatomical claims to a coarse 7x7 activation
grid is inherently approximate. Treat this as a flag for further
review, not a certified verdict — same spirit as the data-leakage
disclosure in the IEEE paper: report the limitation, don't hide it.
"""

import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from model.arw import retry_with_backoff

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-3.6-flash"


@retry_with_backoff(max_retries=3, base_delay=12)
def _call_gemini_audit(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()


def describe_heatmap_location(heatmap: np.ndarray) -> dict:
    """
    Takes the raw (low-resolution, e.g. 7x7) Grad-CAM heatmap and
    computes where its activation is concentrated, in simple spatial
    terms a radiologist-style description can be checked against.

    Returns a dict with the centroid position (as fractions of image
    height/width) and a coarse text label describing the region.
    """
    h, w = heatmap.shape
    total = heatmap.sum()

    if total <= 0:
        return {
            "row_frac": 0.5,
            "col_frac": 0.5,
            "region_label": "no clear activation (heatmap is flat or empty)",
        }

    # Weighted centroid of activation
    row_indices, col_indices = np.indices((h, w))
    row_frac = float((row_indices * heatmap).sum() / total) / (h - 1) if h > 1 else 0.5
    col_frac = float((col_indices * heatmap).sum() / total) / (w - 1) if w > 1 else 0.5

    # Classify centrality: is activation concentrated in the middle
    # third of the image, or biased toward an edge?
    def band(frac):
        if frac < 0.33:
            return "upper" if frac == row_frac else "left"
        elif frac > 0.66:
            return "lower" if frac == row_frac else "right"
        return "central"

    row_band = "upper" if row_frac < 0.33 else "lower" if row_frac > 0.66 else "central"
    col_band = "left" if col_frac < 0.33 else "right" if col_frac > 0.66 else "central"

    if row_band == "central" and col_band == "central":
        region_label = "central region of the image (midline-ish)"
    else:
        parts = [p for p in (row_band, col_band) if p != "central"]
        region_label = f"{'-'.join(parts)} region, away from center" if parts else "central region"

    return {
        "row_frac": row_frac,
        "col_frac": col_frac,
        "region_label": region_label,
    }


def check_consistency(reasoning_text: str, location_info: dict) -> dict:
    """
    Asks Gemini to judge whether the reasoning agent's claimed region
    is plausibly consistent with the measured heatmap location.

    Returns:
        {
            "verdict": str,       # "consistent" | "inconsistent" | "uncertain"
            "explanation": str,   # one or two sentence justification
        }
    """
    prompt = f"""You are auditing an AI diagnostic pipeline for consistency.

The reasoning agent wrote this justification for a brain MRI diagnosis:
"{reasoning_text}"

Independently, a Grad-CAM heatmap analysis found that the model's
attention was concentrated in this location: {location_info['region_label']}
(as a fraction of image height/width: row={location_info['row_frac']:.2f}, col={location_info['col_frac']:.2f},
where 0.5, 0.5 is the exact center of the image).

Judge whether the reasoning agent's claimed anatomical region is
CONSISTENT, INCONSISTENT, or UNCERTAIN relative to the measured
heatmap location. Note: this is a coarse, approximate check — if you
genuinely cannot tell, say UNCERTAIN rather than guessing.

Respond in exactly this format:
VERDICT: <consistent|inconsistent|uncertain>
EXPLANATION: <one or two sentences>"""

    try:
        text = _call_gemini_audit(prompt)

        verdict = "uncertain"
        explanation = text
        for line in text.splitlines():
            if line.upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        return {"verdict": verdict, "explanation": explanation}
    except Exception as e:
        return {"verdict": "uncertain", "explanation": f"Audit check failed after retries: {str(e)}"}


if __name__ == "__main__":
    # Quick standalone test with fake data
    fake_heatmap = np.zeros((7, 7))
    fake_heatmap[3, 3] = 1.0  # dead center activation
    loc = describe_heatmap_location(fake_heatmap)
    print("Location info:", loc)

    result = check_consistency(
        "The lesion appears centered within the sella turcica, a midline structure.",
        loc,
    )
    print("Consistency check:", result)