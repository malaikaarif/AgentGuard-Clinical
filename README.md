# AgentGuard-Clinical

A multi-agent clinical diagnosis pipeline that audits its own reasoning.

It classifies a brain MRI, has an LLM write a diagnostic justification,
generates a Grad-CAM explainability heatmap, and then checks whether the
justification's claimed anatomical region actually matches where the
model's attention was concentrated — flagging cases for human review
when it doesn't.

**This is a proof-of-concept, not a validated clinical tool.**

See [CASE_STUDY.md](CASE_STUDY.md) for findings, known limitations, and
scope notes.

## Architecture

```
intake → classifier → reasoning → explainability → audit → escalation
```

| Agent | What it does |
|---|---|
| **intake** | Validates the input image exists |
| **classifier** | MobileNetV2, reused from prior IEEE-submitted research on brain tumor MRI trust frameworks — glioma / meningioma / notumor / pituitary |
| **reasoning** | Gemini-generated differential diagnosis justification, prompted to reference specific visual/anatomical features |
| **explainability** | Grad-CAM heatmap over the classifier's last conv layer |
| **audit** | Checks whether the reasoning agent's claimed region is consistent with where the heatmap actually activated — the project's core novel mechanism |
| **escalation** | Flags a case for human review when the audit verdict is inconsistent/uncertain, or classifier confidence drops below 90% |

A reliability layer (adapted from a separate agent-framework-reliability
benchmark study, "ARW") wraps the Gemini calls with retry-with-backoff
and a fallback termination guard — if the reasoning agent fails even
after retries, the pipeline degrades gracefully (placeholder reasoning,
forced human review) instead of crashing.

## Setup

```bash
git clone https://github.com/malaikaarif/AgentGuard-Clinical.git
cd AgentGuard-Clinical

python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

You'll also need:
- The trained model file at `model/mobilenetv2_paper_exact.keras` (not
  included in this repo — too large; sourced from the original research
  project)
- A `.env` file in the project root containing:
  ```
  GEMINI_API_KEY=your_key_here
  ```

## Running it

**Single image, command line:**
```bash
python graph.py path/to/image.jpg
```

**Multiple images, logged to CSV** (paced against Gemini's free-tier
daily quota — safe to run incrementally across multiple days, it skips
images already in the CSV):
```bash
python run_batch.py test_images/
```

**Web dashboard:**
```bash
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000 and upload an image.

**Docker:**
```bash
docker compose build
docker compose up
```
Then open http://127.0.0.1:8000 (same dashboard, containerized).

## Tests

```bash
pytest tests/ -v
```

11 tests covering the audit consistency logic, the ARW retry wrapper,
and the classifier (the last two need a fixture image and the model
file respectively — they skip cleanly on CI where those aren't present).
CI runs this suite automatically on every push via GitHub Actions.

## Known limitations

See [CASE_STUDY.md](CASE_STUDY.md) for the full writeup, including a
documented finding about how the audit mechanism handles `notumor`
(no-lesion) cases differently from lesion-present classes, and why.