# Model versions

One name per model that became the default. Small kept improvements bump the minor number (v3.1, v3.2, ...); a new approach or a big jump bumps the major number (v4). Ideas that were tried and not kept keep their descriptive run names and never get a version number.

| Version | Class (`models/...`) | Run name in results/runs | Seed-0 headline | What it adds |
|---|---|---|---|---|
| v1 | `touch:LastTouch` | `touch_v1` | — | last-touch possession |
| v2 | `touch_dead:TouchDead` (older) | `touch_dead_v2` | 0.468 (old copies) | + dead-ball detector, restart team from the next touch |
| v3 | `touch_dead:TouchDead` | `touch_dead_v3` | 0.479 (131831) | + reappearance touches, 0.4 s player speed, restart snap, long-gap advance, still-ball inputs |
| v3.1 (safe reference) | `touch_dead:TouchDeadShapeUnseen` | `touch_dead_v3_shape_unseen`, then `v3.1` | 0.491 (195851); seed 1 0.488 | + team-shape prior decides where the ball is unseen |
| **v4** (current) | `v4_proto:V4ProtoCap` | `v4_proto_cap`, then `v4` | 0.557 (150253); seed 1 0.550 (152536) | learned per-frame H/A/D scorer on v3.1's out-of-fold signals + Viterbi decoding, switch penalty capped at 0.5–1 |

v4 notes: the switch-penalty cap was chosen after seeing on the seed-0 test folds that higher penalties fail the clean guard, so it isn't independent of the test folds; sealed DFL look 2 is the independent check. On seed 1, game 3 reception clean fell 0.0507 (limit 0.05); Jordan counted it as holding (2026-11-14).

Candidates not merged: two-tier shape override (`TouchDeadShapeTwoTier`, jordan/ideas, 0.496). Tried and not kept: shape override everywhere (0.527, failed the clean guard), min_live_s 1 s (0.486), seen-ball shape overrides, long deflection windows, confirmed switches, touch classifier (in v3.1 and in v4), uncapped v4 (0.577, clean −0.112). See results/retired.txt.
