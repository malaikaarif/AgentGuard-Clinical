"""
Tests for arw.py's retry_with_backoff decorator.
time.sleep is mocked so tests run instantly instead of actually
waiting through the backoff delays.
"""
import pytest
from unittest.mock import patch
from model.arw import retry_with_backoff


def test_succeeds_on_first_try_no_retry_needed():
    call_count = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=1)
    def always_works():
        call_count["n"] += 1
        return "success"

    result = always_works()
    assert result == "success"
    assert call_count["n"] == 1  # never had to retry


@patch("model.arw.time.sleep", return_value=None)  # skip real waiting
def test_retries_then_succeeds(mock_sleep):
    call_count = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=1)
    def fails_twice_then_works():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ValueError("simulated transient failure")
        return "success"

    result = fails_twice_then_works()
    assert result == "success"
    assert call_count["n"] == 3  # failed twice, succeeded on 3rd attempt
    assert mock_sleep.call_count == 2  # slept between the two failures


@patch("model.arw.time.sleep", return_value=None)
def test_raises_after_exhausting_all_retries(mock_sleep):
    call_count = {"n": 0}

    @retry_with_backoff(max_retries=2, base_delay=1)
    def always_fails():
        call_count["n"] += 1
        raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="permanent failure"):
        always_fails()

    assert call_count["n"] == 3  # initial attempt + 2 retries = 3 total