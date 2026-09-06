# AgentGuard-Clinical — Case Study

## What this is

A multi-agent clinical diagnosis pipeline that audits its own reasoning.
It classifies a brain MRI, generates an LLM-written diagnostic
justification, produces a Grad-CAM explainability heatmap, and then
checks whether the justification's claimed anatomical region actually
matches where the model's attention was concentrated — flagging cases
for human review when it doesn't.

This is a proof-of-concept, not a validated clinical tool. Findings
below are from a small test batch (n=8) and should be read as
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
- **reliability layer**: Gemini calls (reasoning + audit) are wrapped
  with retry-with-backoff, adapted from the ARW reliability layer built
  for a separate agent-framework-reliability benchmark study.

## Finding: audit verdicts across an 8-image test batch

| Verdict | Count |
|---|---|
| Consistent | 4 |
| Inconsistent | 3 |
| Uncertain | 1 |

3 of 8 cases were flagged as inconsistent or uncertain — the audit
layer is not a rubber stamp; it produces real variability.

## Finding: a class-specific limitation (notumor cases)

Both `notumor` test cases in this batch were flagged inconsistent.
Visual inspection of their Grad-CAM heatmaps showed diffuse or minimal
activation — consistent with there being no localized lesion for the
model to focus on. The reasoning agent, however, still produced
anatomically specific language (e.g. referencing the sella turcica)
regardless of diagnosis, because its prompt elicits location-specific
justification by default.

**This means the audit mechanism, as currently built, is most
meaningful for lesion-present classes.** For negative (notumor)
predictions, a more appropriate check would verify the *absence* of
concentrated activation, rather than compare against a specific claimed
region. This is a known limitation and a concrete next step, not a
hidden one — the same disclosure standard used for the data-leakage
issue documented in the earlier IEEE paper.

## What's not yet done

- Class-aware audit logic for negative (notumor) predictions
- Larger batch run for a statistically meaningful consistency rate
- Docker containerization
- Full ARW integration (only retry-with-backoff is ported; the fallback
  termination guard and self-consistency verification from the original
  agent-reliability paper are not yet adapted into this pipeline)

## Repository

github.com/malaikaarif/AgentGuard-Clinical