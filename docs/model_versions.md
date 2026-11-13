# Model versions

One name per model that became the default. Small kept improvements bump the minor number (v3.1, v3.2, ...); a new approach or a big jump bumps the major number (v4). Ideas that were tried and not kept keep their descriptive run names and never get a version number.

| Version | Class (`models/...`) | Run name in results/runs | Seed-0 headline | What it adds |
|---|---|---|---|---|
| v1 | `touch:LastTouch` | `touch_v1` | — | last-touch possession |
| v2 | `touch_dead:TouchDead` (older) | `touch_dead_v2` | 0.468 (old copies) | + dead-ball detector, restart team from the next touch |
| v3 | `touch_dead:TouchDead` | `touch_dead_v3` | 0.479 (131831) | + reappearance touches, 0.4 s player speed, restart snap, long-gap advance, still-ball inputs |
| **v3.1** (current) | `touch_dead:TouchDeadShapeUnseen` | `touch_dead_v3_shape_unseen`, then `v3.1` | 0.491 (195851); seed 1 0.488 | + team-shape prior decides where the ball is unseen |

Candidates not merged: two-tier shape override (`TouchDeadShapeTwoTier`, jordan/ideas, 0.496). Tried and not kept: shape override everywhere (0.527, failed the clean guard), min_live_s 1 s (0.486).
