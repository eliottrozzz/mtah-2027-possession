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

## Hardening tried (2026-11-12)

**Dead-ball detector: player movement over 0.4 s** (commit 42c04d4; measured with
`evaluation/stress_test.py`, same variants as above):

| Variant (live35) | v2 | v2 + 0.4 s speeds |
|---|---|---|
| As calibrated | 0.471 | 0.465 |
| Player wobble 0.1 m (score) | 0.400 | **0.459** |
| Player wobble 0.1 m (macroF1) | 0.70 | **0.82** |
| Player wobble 0.1 m (change vs as calibrated) | −0.071 | −0.006 |
| Ball blur x1.5 | 0.387 | 0.385 |
| Other variants | — | within about 0.01 of v2's |

**Seed-0 score:**
- headline 0.472 (0.50 / 0.46 / 0.45) against the 0.479 bar (−0.007);
- largest drop on a noisy result −0.018, on a clean result −0.028.

Logged as `results/runs/20261112-131725_touch_dead_v2_seed0.json`.

**Touch threshold relative to the match** (main session): no change at the normal blur or at
1.24 s, so it was reverted. Even v2 fitted on 1.24 s copies reaches only 0.404 there, so most of
that loss is information the blurrier tracking doesn't contain.

**Same fix with a 0.2 s step** (2026-11-13, for comparison):

| Variant (live35) | v2 | 0.4 s | 0.2 s |
|---|---|---|---|
| As calibrated | 0.471 | 0.465 | 0.461 |
| Player wobble 0.1 m (change vs as calibrated) | 0.400 (−0.071) | 0.459 (−0.006) | 0.441 (−0.021) |
| Ball blur x1.5 | 0.387 | 0.385 | 0.381 |
| Seed-0 headline (vs 0.479 bar) | 0.479 | 0.472 (−0.007) | 0.470 (−0.009) |
| Largest noisy / clean drop | — | −0.018 / −0.028 | −0.022 / −0.006 |

Logged as `results/runs/20261113-101504_touch_dead_v2_seed0.json`.

## Ball shown moving during stoppages (2026-11-13)

**Why:** in the real match, measured as summary numbers only, the ball in stoppages is either
missing or moving, never still. Our copies show a still ball in part of some stoppages. "Never
shown" was already tested (+0.003); "shown and moving" wasn't.

**Variant** (`moving_ball_stoppages` / `moving_ball_all_stoppages` in `evaluation/stress_test.py`):
- Each stoppage's ball is carried at 1–3 m/s (drawn per stoppage) from the last ball seen before
  it toward the first seen after it, where play restarts.
- It steers there with some random wandering, then keeps circling near the spot, so it never
  sits still.
- It gets the copies' 0.84 s blur and the usual jitter.
- Either the frames that showed a ball keep showing one (same share, 43% of stoppage frames), or
  the ball is shown on every stoppage frame.
- On stoppage frames, the share where the ball moves under 1 m/s falls from 28% to 3–7%.

Current v2 (0.4 s player speeds), live35, seed 0:

| Variant | spellF1 (game 1 / 2 / 3) | macroF1 | Detector D share (true 0.428) | Dead precision / recall |
|---|---|---|---|---|
| As calibrated | 0.465 (0.50 / 0.44 / 0.45) | 0.820 | 0.418 | 0.90 / 0.88 |
| Moving ball, same share | 0.445 (0.48 / 0.42 / 0.44), −0.020 | 0.792 | 0.390 | 0.89 / 0.82 |
| Moving ball, every stoppage frame | 0.434 (0.47 / 0.41 / 0.42), −0.031 | 0.776 | 0.374 | 0.89 / 0.78 |

**What breaks: `ball_still`.**
- The detector learned "still ball ⇒ dead": on the copies, `ball_still` is 0.75 on dead frames
  against 0.04 on live ones, with large positive weights.
- With a moving ball it drops to 0.06 on dead frames, so the detector misses stoppages: recall
  falls from 0.88 to 0.78–0.82 while precision stays put.
- Blanking `ball_still` at predict time (the fitted model treats it as unmeasured):

  | Variant | Full | Without `ball_still` | Without `ball_still` and `ball_out` |
  |---|---|---|---|
  | As calibrated | 0.465 | 0.448 | 0.441 |
  | Moving, same share | 0.445 | 0.443 | 0.435 |
  | Moving, every frame | 0.434 | 0.445 | 0.437 |

  Without it, the variants score about the same as the calibrated copies (0.443–0.448), so
  `ball_still` accounts for the loss. `ball_out` matters little.

## On the copies with a carried stoppage ball (2026-11-13)

The copies now carry every stoppage ball shown to the restart at 4–6 m/s, a speed fitted to the
real touch signal. The stoppage-ball variants were redone to match:
- **never:** the ball is never shown in stoppages;
- **more:** a ball is shown in every stoppage, for part of it;
- **every frame:** a carried ball is shown on every stoppage frame;
- **still:** the ball lies still, as in the old copies;
- **slow:** the ball is carried at 1–3 m/s.

Stress table, live35, the model before restart snapping (named v2 then, v3 at 0.460 now):

| Variant | spellF1 (game 1 / 2 / 3) | Change | macroF1 |
|---|---|---|---|
| As calibrated | 0.450 (0.50 / 0.41 / 0.44) | — | 0.816 |
| Track breaks ×2 / ×0.5 | 0.441 / 0.451 | −0.010 / +0.001 | 0.81 |
| Ball blur ×0.5 / ×1.5 | 0.486 / 0.389 | +0.036 / −0.061 | 0.83 / 0.79 |
| Random ball gaps ×2 | 0.433 | −0.017 | 0.81 |
| Ball missing on 65% of live frames | 0.340 | −0.110 | 0.77 |
| Stoppage ball never / more / every frame | 0.446 / 0.459 / 0.450 | −0.004 / +0.009 / 0.000 | 0.81–0.82 |
| **Stoppage ball lying still (the old copies)** | **0.414** (0.46 / 0.38 / 0.41) | **−0.037** | **0.657** |
| Stoppage ball carried slowly (1–3 m/s) | 0.433 | −0.017 | 0.793 |
| Player wobble 0.1 m | 0.443 | −0.007 | 0.809 |
| Every high ball lost | 0.470 | +0.019 | 0.819 |

**Dead-ball detector and `ball_still`** (v3 with restart snapping, live35):
- On the new copies the stoppage ball is never still: `ball_still` is 0.000 on dead frames and
  0.036 on live ones.
- So the detector now learns "still ball ⇒ live": the `ball_still` weights are negative
  (round 1: −0.65, −0.33, 0.03, −0.37, −0.56). Physically that is backwards.

| Copies | spellF1 | D share (true 0.428) | Dead precision / recall | Without `ball_still` |
|---|---|---|---|---|
| As calibrated | 0.456 | 0.411 | 0.90 / 0.87 | 0.443 (recall 0.85) |
| Stoppage ball still | 0.417 | 0.224 | 0.88 / **0.46** | 0.457 (recall 0.85) |
| Every stoppage frame | 0.450 | 0.422 | 0.90 / 0.89 | 0.434 |
| Carried slowly | 0.439 | 0.371 | 0.92 / 0.79 | 0.442 |

- Recall on the new copies is back to 0.87, against 0.88 on the old copies.
- `ball_still` is worth +0.013 on the copies we score on.
- If the real stoppage ball is ever still, though, it halves recall and the macroF1 drops to
  0.66.
- Blanking it at predict time keeps every variant at 0.434–0.457. That is the fitted model with
  the feature treated as unmeasured, not a refit.
