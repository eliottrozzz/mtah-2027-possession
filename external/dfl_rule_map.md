# Keeping the DFL key in step with the Metrica key

The sealed DFL key must follow the same rules as the Metrica key (`external/metrica_labels.py`,
`POSSESSION_RULES.md`). This maps each Metrica rule to where the DFL key applies it, so a rule
change on the Metrica side can be copied across without missing anything.

Last aligned with: **`answer-key-v1`** (commit 7abc54d on `main`).

## Rule by rule

| Rule (POSSESSION_RULES.md) | Metrica key (`metrica_labels.py`) | DFL key (`dfl_labels.py`) |
|---|---|---|
| A deliberate touch gives that team the ball | `POSSESSION_TYPES` (PASS, SHOT, RECOVERY, CARRY, SET PIECE) | `ds.TOUCH_TAGS` (Play, ShotAtGoal, OtherBallAction, BallClaiming) and the restart kick, in the touch timeline of `label_match` |
| Duels and challenges don't switch on their own | CHALLENGE, BALL LOST, CARD not in `POSSESSION_TYPES` | TacklingGame, BallDeflection not in `TOUCH_TAGS` |
| A won duel that claims the ball switches | CHALLENGE WON followed by a logged RECOVERY | TacklingGame `WinnerResult=ballClaimed`: a touch by the winner (`duels` in `label_match`) |
| Collecting an opponent's loose pass or clearance switches at the collection | Logged as RECOVERY | Not logged by DFL: placed from the tracking between the two teams' logged touches (`received` in `label_match`); the log decides who, the tracking when |
| Deflections and blocks don't switch | `demote_blocks_and_saves`: RECOVERY BLOCKED becomes a deflection | BallClaiming `Type=Block` skipped; the defending touch within `BLOCK_S` after a BlockedShot skipped |
| A save switches only if the defenders play next | `demote_blocks_and_saves`: SAVED kept only if the same team touches next | Implicit: a keeper's catch is a BallClaiming touch; a parry is a BallDeflection (no switch) |
| Ball out of play: D from the crossing | BALL OUT, shot OUT: `end_frame` | `stop_of`, `OUT_RESTARTS`: first tracked crossing (`AT_LINE_M`); fallback decision time − typical gap |
| Goal: D from the goal to the kick-off | shot GOAL: `end_frame` | `stop_of`, KickOff after a SuccessfulShot: crossing within `GOAL_CROSS_S`; fallback shot + 1 s |
| Foul, offside: D from the whistle; advantage is play on | FAULT RECEIVED, CHALLENGE+FAULT, OFFSIDE, HAND BALL, FORCED | `stop_of`, FreeKick/Penalty: Foul/Offside event (lag-corrected); after the last touch if touches follow (advantage) |
| Drop ball | (no event) | `stop_of`, RefereeBall: last touch + `DROP_BALL_S` (a guess) |
| Stoppage start, when DFL's own ball status disagrees (DFL only) | (no equivalent: Metrica has no per-frame status) | End of `label_match`: start where DFL's BallStatus says, unless a touch is logged after it (then `TOUCH_SLACK_S` after the last touch); DFL stopping later counts only with a touch logged in between |
| End of half | END HALF | FinalWhistle loop at the end of `label_match` |
| Start of half: D until the kick-off | `label_period` starts D | The first-touch loop (`for half in (1, 2)`) plus the KickOff restart |
| Restart: D until the restart pass is received (`--restart reception`) or from the kick (`kick`) | SET PIECE then PASS: `end_frame` of the pass | `alive_of`: first other player within `RECEIVE_M`, from `RECEIVE_MIN_FRAMES` after the kick, capped at the next logged touch |
| Restart taken as a shot is alive at the kick | SHOT after SET PIECE: `change(event["start"])` | `alive_of`: ShotAtGoal restart returns the kick row |
| Goal kick waits for its reception like any restart (v1) | goal-kick PASS or BALL LOST GOAL KICK treated as the restart kick | GoalKick is in `RESTARTS` with no exception in `alive_of` |
| Handling the ball in a stoppage doesn't end D (v1: keeper pickup) | RECOVERY in a stoppage: `pass` | Implicit: the D interval `[stop, alive)` overwrites any logged touch inside it |
| Keeper roll-out before a goal kick (12:15) | No rule: the model's job | No rule |
| A touch logged after the stoppage but made before it can switch, never restart | `event["start"] < stopped_at` branch | Implicit: touches sit at their snapped rows, D starts at the crossing |

## When the Metrica rules change

1. Find the changed rule in the table and make the same change in `dfl_labels.py`. If no DFL
   event matches it, write that down in the module docstring rather than guessing.
2. Update "Last aligned with" above.
3. Rebuild and check:

   ```bash
   .venv/bin/python external/dfl_labels.py && .venv/bin/python external/dfl_labels.py --restart kick
   .venv/bin/python external/dfl_noise.py --calibration data/ball_gap_stats.json
   .venv/bin/python evaluation/score_external.py --sealed --oracle    # must print PASS
   ```

   The noisy copies depend on the reception key (the ball is deleted on dead frames), so they
   are rebuilt too.
4. Never look at a real model's DFL score while doing this. The sealed run is the milestone
   only (`--confirm-milestone`).

The sync (`dfl_sync.py`, frozen at b79114c) doesn't depend on the rules and never needs
rebuilding.

## Where our key and DFL's own flags still differ (2026-11-12, all 7 matches)

* **Restarts:** DFL's flags mark the ball alive at the kick; we wait for the reception, a median
  1.2 s later, as the public ground truth does (8:38 free kick, 11:53 corner). So DFL's flags are
  not the system behind the ground truth, and stay a cross-check, not a key.
* **Stoppage starts:** taken from DFL's flags since 2026-11-12, except where logged touches
  contradict them (6 stretches over 10 s remain, each with a touch logged inside DFL's stop).
* **Possession switches:** DFL switched 2–10 s before us 338 times. Most were opponents
  collecting a misplaced pass or clearance, which DFL doesn't log; since 2026-11-13 the key
  places those from the tracking, and won "ballClaimed" duels count. 199 remain, 163 of them
  with the old team still playing the ball after DFL's switch: DFL's possession column seems to
  switch when a player gets close, not at a touch, which POSSESSION_RULES.md rejects.
  Agreement with DFL's flags: 89.4% before, 92.0% after.
