"""Checks on pretend_tags.py: seeded, throw-ins never have a team, corruption really applied.

    python external/test_pretend_tags.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretend_tags as pt  # noqa: E402


def main():
    period = None
    import metrica_tracking as mt
    period = mt.load_game(1)["period"]
    a = pt.pretend_tags("metrica", 1, 0, period=period)
    b = pt.pretend_tags("metrica", 1, 0, period=period)
    c = pt.pretend_tags("metrica", 1, 1, period=period)
    assert a == b, "same seed, different tags"
    assert a != c, "seed has no effect"
    assert set(pt.MISS_RATE) == set(pt.TYPES)
    assert all(0 <= r <= 1 for r in pt.MISS_RATE.values())
    for tag in a + c:
        assert tag["type"] in pt.TYPES
        assert tag["team"] in ("H", "A", None)
        assert 0 <= tag["start_frame"] <= tag["end_frame"] < len(period)
        if tag["type"] in pt.NA_TEAM_TYPES:
            assert tag["team"] is None
    # Leak guard: with the default corruption, the tags must not be the logged restarts.
    clean = pt.clean_restarts("metrica", 1)
    kicks = {r for r, _, _ in clean}
    exact = sum(t["start_frame"] in kicks for t in a)
    assert exact < 0.2 * len(a), "too many tags start exactly on a logged kick"
    assert len([t for t in a if t["type_raw"] == "pretend"]) < 0.8 * len(clean), "too few misses"
    jitter = [min(abs(t["start_frame"] - k) for k in kicks) for t in a
              if t["type_raw"] == "pretend"]
    assert np.max(jitter) <= (pt.START_LEAD_S + pt.START_JITTER_S) * pt.FPS + 1
    print("pretend_tags checks passed ({} tags, {} start exactly on a kick)".format(len(a), exact))


if __name__ == "__main__":
    main()
