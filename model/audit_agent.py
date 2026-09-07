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


def compute_activation_concentration(heatmap: np.ndarray) -> dict:
    """
    Measures whether the heatmap's activation is concentrated in a
    focal hotspot, or diffuse/spread across the whole image.

    Uses coefficient of variation (std/mean) as a simple concentration
    score: a few very active cells among mostly-quiet ones gives a high
    score (focal/concentrated); near-uniform activation everywhere gives
    a low score (diffuse). Threshold of 1.0 is a heuristic, not a
    calibrated statistical cutoff — chosen from inspecting the notumor
    cases that motivated this function, not derived from a larger study.
    """
    mean = float(heatmap.mean())
    std = float(heatmap.std())
    concentration_score = std / (mean + 1e-8)
    is_diffuse = concentration_score < 1.0

    return {
        "concentration_score": concentration_score,
        "is_diffuse": is_diffuse,
    }


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


def check_consistency(reasoning_text: str, location_info: dict, diagnosis: str = None, heatmap: np.ndarray = None) -> dict:
    """
    Asks Gemini to judge whether the reasoning agent's claimed region
    is plausibly consistent with the measured heatmap location.

    For 'notumor' diagnoses, uses a different check: since there's no
    lesion to localize, the expectation flips — we check whether the
    heatmap shows diffuse/low-concentration activation (expected for a
    negative finding) rather than comparing to a specific claimed
    region. This addresses a real pattern found during testing: both
    notumor cases in an early batch were flagged inconsistent for a
    structural reason (no lesion = no valid comparison target), not
    because anything was actually wrong.

    Returns:
        {
            "verdict": str,       # "consistent" | "inconsistent" | "uncertain"
            "explanation": str,   # one or two sentence justification
        }
    """
    if diagnosis == "notumor" and heatmap is not None:
        concentration = compute_activation_concentration(heatmap)
        prompt = f"""You are auditing an AI diagnostic pipeline for consistency.

The classifier predicted NO TUMOR (notumor) for this brain MRI. The
reasoning agent wrote this justification:
"{reasoning_text}"

Since there is no lesion to localize in a true negative case, the
EXPECTED pattern is diffuse or low-concentration heatmap activation,
NOT a focal hotspot on a specific anatomical region. Independently,
the heatmap's activation concentration score is {concentration['concentration_score']:.2f}
({'diffuse — spread across the image' if concentration['is_diffuse'] else 'concentrated — focused on a specific hotspot'}).

Judge: is this CONSISTENT (diffuse activation, matching the expectation
for a no-lesion case), INCONSISTENT (surprisingly concentrated despite
no lesion being predicted, which may indicate the model is attending
to something it shouldn't be), or UNCERTAIN?

Respond in exactly this format:
VERDICT: <consistent|inconsistent|uncertain>
EXPLANATION: <one or two sentences>"""
    else:
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
    # Test 1: original lesion-present case
    fake_heatmap = np.zeros((7, 7))
    fake_heatmap[3, 3] = 1.0  # dead center activation
    loc = describe_heatmap_location(fake_heatmap)
    print("Location info:", loc)

    result = check_consistency(
        "The lesion appears centered within the sella turcica, a midline structure.",
        loc,
        diagnosis="pituitary",
    )
    print("Consistency check (pituitary, focal case):", result)

    # Test 2: notumor case with diffuse activation (expected pattern)
    diffuse_heatmap = np.ones((7, 7)) * 0.5  # near-uniform, low concentration
    loc2 = describe_heatmap_location(diffuse_heatmap)
    concentration2 = compute_activation_concentration(diffuse_heatmap)
    print("\nConcentration info (diffuse):", concentration2)

    result2 = check_consistency(
        "The classifier predicts no evidence of intracranial tumor with high confidence.",
        loc2,
        diagnosis="notumor",
        heatmap=diffuse_heatmap,
    )
    print("Consistency check (notumor, diffuse case):", result2)