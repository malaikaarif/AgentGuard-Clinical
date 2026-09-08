# AgentGuard-Clinical — Case Study

## What this is

A multi-agent clinical diagnosis pipeline that audits its own reasoning.
It classifies a brain MRI, generates an LLM-written diagnostic
justification, produces a Grad-CAM explainability heatmap, and then
checks whether the justification's claimed anatomical region actually
matches where the model's attention was concentrated — flagging cases
for human review when it doesn't.

This is a proof-of-concept, not a validated clinical tool. Findings
below are from a modest test batch (n=20) and should be read as
illustrative, not statistically conclusive.

## Architecture

```
intake → classifier → reasoning → explainability → audit → escalation
```

- **classifier**: MobileNetV2, reused from prior IEEE-submitted research
  ("Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain
  Tumor MRI Classification"), 94.19% accuracy on the paper's held-out
  1,600-image test set.
- **reasoning**: Gemini-generated differential diagnosis justification,
  prompted to reference specific visual/anatomical features.
- **explainability**: Grad-CAM heatmap, code adapted from the same
  research's explainability audit.
- **audit**: computes the heatmap's activation centroid and asks an LLM
  to judge whether the reasoning agent's claimed region is consistent
  with it. This is the project's core novel mechanism.
- **escalation**: flags a case for human review when the audit verdict
  is inconsistent/uncertain, or classifier confidence drops below 90%.
- **reliability layer (ARW)**: Gemini calls (reasoning + audit) are
  wrapped with retry-with-backoff, adapted from a separate agent-
  framework-reliability benchmark study. The reasoning agent also has a
  fallback termination guard: if it fails even after retries, the
  pipeline degrades gracefully (placeholder reasoning text, forced
  human review) instead of crashing — verified against a real Gemini
  free-tier quota exhaustion during testing.

## Finding: audit verdicts across a 20-image test batch

| Verdict | Count | % |
|---|---|---|
| Consistent | 10 | 50% |
| Inconsistent | 7 | 35% |
| Uncertain | 3 | 15% |

Half of cases were flagged as inconsistent or uncertain — the audit
layer is not a rubber stamp; it produces real, substantial variability.
This is still a modest sample (n=20) and should be read as illustrative
of the mechanism working, not as a statistically definitive consistency
rate.

## Finding: a class-specific limitation (notumor cases)

Both `notumor` test cases in this batch were flagged inconsistent.
Visual inspection of their Grad-CAM heatmaps showed diffuse or minimal
activation — consistent with there being no localized lesion for the
model to focus on. The reasoning agent, however, still produced
anatomically specific language (e.g. referencing the sella turcica)
regardless of diagnosis, because its prompt elicits location-specific
justification by default.

**This means the audit mechanism, as originally built, was most
meaningful for lesion-present classes.** For negative (notumor)
predictions, comparing to a specific claimed region was the wrong
check entirely.

**Update: this has since been fixed, with a refined result.** The audit
agent now branches on diagnosis — for `notumor` predictions, it checks
whether the heatmap shows diffuse/low-concentration activation (the
expected pattern when there's no lesion to localize) rather than
comparing to a claimed anatomical region. Re-testing the same `notumor`
case that originally motivated this fix (`Te-no_8.jpg`) with the new
logic still returned **INCONSISTENT** — but for a materially different
and more accurate reason: the heatmap's activation concentration score
(1.51, on a scale where >1.0 indicates a focal hotspot) showed the
model's attention was genuinely concentrated despite predicting no
tumor, not flagged due to a category-mismatch in the check itself. This
is a more honest and more interesting finding than the original: it
suggests the classifier may be attending to something specific even on
negative predictions, which is worth further investigation, not an
artifact of a broken audit check. The concentration-score threshold
(1.0) is a heuristic chosen by inspecting this case, not a calibrated
statistical cutoff — a real limitation to be upfront about.

## What's done since this was first written

- ✅ Class-aware audit logic for notumor cases (see refined finding above)
- ✅ Docker containerization (Dockerfile + docker-compose, verified working end-to-end via the dashboard)
- ✅ Full ARW integration (retry-with-backoff + fallback termination guard, both verified against real failures — a live 429 quota error and a live quota exhaustion mid-batch)
- ✅ Pytest suite (11 tests) + GitHub Actions CI, running on every push
- ✅ FastAPI dashboard (main.py) for browser-based use, not just CLI

## What's still open

- Larger batch run for a more statistically robust consistency rate (currently n=20; could extend to 30-50 if time allows)
- Self-consistency verification (a third ARW mechanism from the original agent-reliability paper — running the reasoning agent multiple times and checking agreement — not yet ported; only retry-with-backoff and the fallback guard are)
- The notumor concentration-score threshold (1.0) is a heuristic, not statistically calibrated — would benefit from validation against a larger set of true-negative cases

## Repository

github.com/malaikaarif/AgentGuard-Clinical