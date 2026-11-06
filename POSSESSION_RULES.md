# Possession rules (working draft)

What we think the ground truth means by "possession", worked out from CHALLENGE.md and from inspecting the public 5 minutes against the tactical video. These rules drive both our answer key for the external games and the design of the model.

**How we're allowed to use the public 5 minutes:** to understand the definition, written up in words, as below. Never to take numbers from (durations, thresholds, offsets), never to tune or train on. Times below are in the tactical video, where kickoff is at 7:27.40.

**Freeze:** these rules get frozen before anyone scores a model against the answer key. Changes after that need a football reason, not a better score.

---

## The labels

| label | meaning |
|---|---|
| `H` | Barcelona in possession, ball alive |
| `A` | Athletic in possession, ball alive |
| `D` | ball dead |
| `X` | no answer: we only write it on ungraded frames |

* **Graded frames:** 11,185 – 90,651 (1st half) and 110,808 – 190,197 (2nd half). Everything else is ignored, so we write `X` there.
* **2nd half trap:** frames 110,808 – 113,741 are graded and all `D` (before the second-half kickoff).
* **Never `X` on a graded frame:** the ranking metric replaces it with our previous label anyway, and macroF1 counts it as a miss. If unsure, write our best guess.

---

## Rule 1: who has the ball

**Possession belongs to the team whose player last touched the ball on purpose.**

| Counts (a deliberate touch) | Doesn't count |
|---|---|
| Pass, including a one-touch pass | Ball bouncing or deflecting off a player by accident |
| Header, including a won aerial duel | Being closest to the ball without touching it |
| Clearance, or deliberately playing the ball after a tackle | A tackle that only blocks or deflects the ball |
| | Shielding the ball out without touching it |
| Shot | |
| Receiving or controlling the ball | |
| Keeper catch | |

* **No control needed.** One deliberate touch is enough, even if the team loses the ball straight away. Possessions under a second long are real.
* **Loose balls** stay with the team that last touched the ball on purpose, until the other team's first deliberate touch.
* **No "contested" label.** The model may use a contested state internally, but every contested frame is rewritten to the last team with a deliberate touch.
* **The switch** happens at the new team's first deliberate touch, possibly a few frames before it, as the player moves onto the ball (unconfirmed).

This is stricter than CHALLENGE.md's wording, which says a won aerial duel without control doesn't count. The public ground truth counts it, and CHALLENGE.md says the ground truth wins when they disagree.

---

## Rule 2: dead ball

* **Starts** when the ball crosses a line, or at the whistle (offside, foul).
* **Advantage** is play on: no `D` until the referee actually stops play.
* **Ends** at the restarting team's first deliberate touch, **even before the restart is legally taken**. At the 12:15 goal kick, `D` ends when the keeper rolls the ball to a defender, about 1.5 s before the defender takes the kick. The restarting team has possession from then. Only one example so far, so check other restarts.
* **Before it goes out:** the last team with a deliberate touch keeps possession until the ball crosses the line, even if an opponent is closer.
* **Goals:** `D` from the goal until the kick-off. Not seen yet: no goal in the public 5 minutes.

---

## Evidence from the public 5 minutes

| Video time | What happens | Label | Shows |
|---|---|---|---|
| 7:42 – 7:45 | Barcelona pass goes straight to Athletic, Athletic shoot, keeper catches | `A`, then `H` at 7:44.96 | Keeper catch counts |
| 8:33 – 8:40 | Offside, then Barcelona free kick | `D` from 8:34.04, `H` from 8:39.56 | Dead at the whistle, alive at the restart |
| 9:05 – 9:09 | Athletic tackle deflects a Barcelona pass (end of 9:06), Athletic defender then passes it (9:08) | `A` from 9:08.16 | Deflection, even from a tackle, doesn't switch; the pass does |
| 10:13 – 10:15 | Barcelona tackle, then kick away without control, Athletic regain | `H` for 0.8 s | One deliberate touch counts |
| 10:30 – 10:32 | Barcelona win a header into an Athletic player, ball bounces loose, Athletic player moves in to take it | `H` for 1.3 s, `A` from 10:31.96 | Won header counts; accidental bounce off a player doesn't |
| 10:33 – 10:56 | Foul, advantage played, whistle, Athletic restart | stays `A`, `D` from 10:35.56, `A` from 10:56.00 | Advantage is play on |
| 11:30 – 11:35 | Athletic pass deflects off a Barcelona player's legs, Barcelona player reaches the ball, his back-pass goes out | `H` from 11:31.72, `D` from 11:35.20 | Deflection doesn't count; switch at the deliberate touch |
| 11:53 | Athletic corner | `A` from 11:53.52 | Restart starts possession |
| 12:00 – 12:01 | Athletic ball into the box, nobody touches it, goes out | stays `A`, `D` from 12:01.08 | No touch, no switch |
| 12:15 – 12:17 | Keeper rolls the ball to a defender (12:15), defender takes the goal kick (12:17) | `H` from 12:15.56 | Dead ball ends at the first touch, before the legal restart |

---

## Open questions

Check these frame by frame (VLC, press E to step one frame):

1. **10:13.84:** is the first `H` frame the tackle or the kick away? After 9:06, we expect the kick.
2. **Switch timing:** do switches generally happen at the touch, or a few frames before, as the player moves onto the ball?
3. **Other restarts:** does `D` always end at the first touch (8:39 free kick, 10:56 restart, 11:53 corner), or was the goal kick special?
4. **Goals, kick-offs, keeper parries, handballs:** not in the public 5 minutes, so we follow the rules above until we know better.

---

## Ideas we dropped

* **"Must keep the ball for 2 seconds" (control time):** erases the real 0.8 s and 1.3 s possessions.
* **"Switch on control, as CHALLENGE.md describes":** the ground truth switches on a single deliberate touch.
* **"Whoever is closest has it" (the baseline):** switches too early when an opponent is near the ball without touching it.
* **"Dead ball ends when the restart is legally taken":** the ground truth ends it at the first touch, even a keeper rolling the ball to a teammate.

---

## What this means for us

* **Answer key (external games):** label by last deliberate touch, ignoring accidental deflections. The mapping from Metrica event types to these rules is still to be written.
* **Model:** it must allow possessions of about 1 second after a tackle or header, while still ignoring flicker caused by tracking noise. Telling a deliberate touch from noise and from an accidental deflection is the core difficulty.
