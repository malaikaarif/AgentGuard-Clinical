"""
Test for classifier_agent.py's classify_image function.

Unlike the audit_agent and arw tests, this needs a real image file and
the trained model — it's an integration test, not a pure unit test.
Skips cleanly (instead of failing confusingly) if the fixture image
isn't present, so CI doesn't break just because a large image wasn't
committed.
"""
import os
import pytest
from model.classifier_agent import classify_image, CLASS_NAMES

FIXTURE_IMAGE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_pituitary.jpg")


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_IMAGE),
    reason="Fixture image not found — add tests/fixtures/sample_pituitary.jpg to run this test"
)
def test_classify_image_returns_expected_keys():
    result = classify_image(FIXTURE_IMAGE)

    assert set(result.keys()) == {"diagnosis", "confidence", "logits", "class_names"}
    assert result["diagnosis"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["logits"]) == len(CLASS_NAMES)
    assert result["class_names"] == CLASS_NAMES


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_IMAGE),
    reason="Fixture image not found — add tests/fixtures/sample_pituitary.jpg to run this test"
)
def test_logits_are_valid_probabilities():
    result = classify_image(FIXTURE_IMAGE)

    # Softmax output should sum to ~1.0 and each value should be non-negative
    assert abs(sum(result["logits"]) - 1.0) < 0.01
    assert all(0.0 <= p <= 1.0 for p in result["logits"])