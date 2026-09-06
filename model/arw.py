"""
ARW (Adaptive Reliability Wrapper) — lite version, adapted for
AgentGuard-Clinical from the framework-agnostic reliability layer
built for the agent-reliability paper (retry-with-backoff piece only,
for now — fallback termination guard and self-consistency verification
are future work, not yet ported).

This directly addresses a real failure hit during testing: Gemini's
free-tier 429 rate limit errors, which used to just crash the pipeline.
"""

import time
import functools


def retry_with_backoff(max_retries=3, base_delay=12, max_delay=60):
    """
    Retries a function on any exception, with exponential backoff.
    base_delay=12 is chosen based on the actual retry_delay Gemini's
    API returned during a real 429 quota error seen in testing.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"[ARW] Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                    print(f"[ARW] Retrying in {delay}s...")
                    time.sleep(delay)
            print(f"[ARW] All {max_retries + 1} attempts failed. Giving up.")
            raise last_exception
        return wrapper
    return decorator