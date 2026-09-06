"""
Tests for audit_agent.py's describe_heatmap_location function.
These are pure unit tests on synthetic heatmaps — no model, no API
calls, no real images needed. Fast and deterministic.
"""
import numpy as np
from model.audit_agent import describe_heatmap_location


def test_centered_heatmap_returns_central_label():
    heatmap = np.zeros((7, 7))
    heatmap[3, 3] = 1.0  # dead center of a 7x7 grid
    result = describe_heatmap_location(heatmap)

    assert 0.4 <= result["row_frac"] <= 0.6
    assert 0.4 <= result["col_frac"] <= 0.6
    assert "central" in result["region_label"]


def test_corner_heatmap_returns_offcenter_label():
    heatmap = np.zeros((7, 7))
    heatmap[0, 0] = 1.0  # top-left corner
    result = describe_heatmap_location(heatmap)

    assert result["row_frac"] < 0.33
    assert result["col_frac"] < 0.33
    assert "away from center" in result["region_label"]


def test_empty_heatmap_does_not_crash():
    heatmap = np.zeros((7, 7))  # all zeros, no activation at all
    result = describe_heatmap_location(heatmap)

    assert result["row_frac"] == 0.5
    assert result["col_frac"] == 0.5
    assert "no clear activation" in result["region_label"]


def test_diffuse_heatmap_lands_near_center():
    # Uniform activation everywhere should average out to roughly central
    heatmap = np.ones((7, 7))
    result = describe_heatmap_location(heatmap)

    assert 0.4 <= result["row_frac"] <= 0.6
    assert 0.4 <= result["col_frac"] <= 0.6