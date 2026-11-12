# Stress test: touch_dead_v2 against wrong noise (2026-11-12)

**Question:** our noisy copies imitate the provided tracking's summary statistics. If we got one
of them wrong, does v2 hold up or collapse?

**How:**
- v2 is fitted per round exactly as `evaluation/score_external.py` does: standard live20/35
  seed-0 copies, reception key.
- Each test game is then predicted on a variant copy (live35, seed 0) with **one** noise setting
  changed at a time.
- Nothing was tuned on these results and nothing was written to `data/`. External data only.
- The "as calibrated" variant reproduces the stored copies exactly (checked array by array), so
  the differences below come from the changed setting alone.

| Variant (live35) | spellF1 (game 1 / 2 / 3) | Change | macroF1 |
|---|---|---|---|
| As calibrated (the copies we score on) | **0.471** (0.51 / 0.45 / 0.46) | — | 0.83 |
| Track breaks ×2 | 0.461 (0.52 / 0.44 / 0.43) | −0.010 | 0.82 |
| Track breaks ×0.5 | 0.469 | −0.002 | 0.83 |
| Ball blur shorter (0.84 s → 0.44 s) | 0.515 (0.52 / 0.52 / 0.50) | +0.044 | 0.84 |
| **Ball blur longer (0.84 s → 1.24 s)** | **0.387** (0.41 / 0.37 / 0.38) | **−0.084** | 0.80 |
| Random ball gaps twice as long | 0.437 (0.45 / 0.43 / 0.43) | −0.034 | 0.81 |
| Ball missing on 65% of live frames | 0.345 (0.37 / 0.33 / 0.34) | −0.126 | 0.78 |
| Still ball never shown in stoppages | 0.474 | +0.003 | 0.82 |
| Still ball always shown in stoppages | 0.481 | +0.010 | 0.84 |
| **Player position wobble 0.1 m per frame** | **0.400** (0.42 / 0.39 / 0.39) | **−0.071** | **0.70** |
| Every high ball lost | 0.465 | −0.006 | 0.83 |

## What it says

**Robust:**
- broken player tracks;
- losing every high ball;
- whether the ball is shown during stoppages: the dead-ball detector doesn't lean on the ball,
  as intended.

**Expected:** more missing ball and longer gaps cost score smoothly, with no collapse. At 65%
missing it drops to 0.35, still about double the nearest-player floor.

**Two sensitivities:**

1. **Ball blur longer than measured: −0.084.** The touch thresholds are fitted to a 0.84 s blur.
   If the real tracking is blurrier, the touches get weaker than the fitted threshold expects. The
   measured blur matches the real touch signal closely (p50–p99 within about 0.3 m/s), so this is
   a risk only if that measurement is off. Shorter blur *helps* (+0.044).
   - *Possible hardening (touch model):* make the touch threshold relative to the match's own
     touch-strength distribution rather than an absolute m/s value, so it adapts to the blur.
2. **Player wobble: −0.071, macroF1 0.83 → 0.70.** The dead-ball detector measures player speed
   frame to frame. A 0.1 m random wobble per frame inflates that by several m/s, so players
   never look still. The real tracking's measured wobble is tiny (about 0.001 m), so this is
   unlikely to bite.
   - *Possible hardening (dead-ball detector):* measure speed over a longer step (for example
     0.4 s), which averages wobble out at little cost.
