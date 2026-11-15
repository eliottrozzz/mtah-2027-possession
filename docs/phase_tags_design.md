# Pretend phase tags: design and feasibility (Track B, 2026-11-15)

This is groundwork only. Nothing goes into v5 until the set pieces are done. The sources are
DATA.md's public Smart Tagging section and the external games. Nothing here reads the provided
match.

## 1. What does Team mean for each code?

**The public counts point to "the team doing the phase".** For a DEFENDING code, that means the
defending team. Four codes pair up with exactly equal counts:

| Attacking code | n | Defending code | n |
|---|---|---|---|
| BUILD UP | 22 | DEFENDING IN ATTACKING THIRD | 22 |
| PROGRESSION | 66 | DEFENDING IN MIDDLE THIRD | 66 |
| CREATING CHANCES | 30 | DEFENDING IN DEFENSIVE THIRD | 30 |
| ATTACKING TRANSITION | 24 | DEFENSIVE TRANSITION | 24 |

The pairs make football sense. Building up in your own third is what the other team defends in
*its* attacking third, and the same holds for the middle and final thirds. Equal counts suggest
the tagger writes each attacking phase together with a mirror tag for the other team. A mirror
tag that repeated the same team would add nothing, so the likely reading is:

- **Attacking codes** (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION): Team is
  the team in possession.
- **Defending codes** (DEFENDING IN ..., DEFENSIVE TRANSITION): Team is the team *without* the
  ball.
- **Unpaired codes** (BALL IN FINAL THIRD 93, BALL IN THE BOX 72, PLAYERS IN THE BOX 97, LONG
  BALL 29, GOALS 4): presumably the attacking team. The public text can't confirm it.

All 13 phase codes add up to exactly 579, the phase total Jordan's run printed. This is an
inference from counts, not a fact.

**The Monday summary should test it (the main session writes it, Jordan runs it, counts only):**

1. **Pairing**, from the tag file alone. For each pair: the share of attacking instances with a
   partner of the other code that overlaps at least 50% in time, and the share of those partners
   with the opposite Team.
2. **Agreement with the model**, per code. The share of tagged, predicted-live frames where v4's
   (or v3.1's) predicted team equals the tag's Team. Expect a high share for attacking codes and
   a low one for defending codes.
3. **Edges**, per code: the median and spread of (tag start − nearest predicted switch to that
   team), in seconds. This is the boundary noise the pretend tags must copy.

## 2. Generating pretend tags (Metrica, DFL)

The same recipe works for both: possession spells and ball x in the attacking team's direction.

| Code | Rule (one threshold per rule, fitted so the count per 90 min matches DATA.md) |
|---|---|
| BUILD UP | spell starts in its own third and stays there for at least T s |
| PROGRESSION | spell moves from its own or the middle third into the middle or final third |
| BALL IN FINAL THIRD | spell reaches the final third, duration ≥ T |
| CREATING CHANCES | final-third spell that reaches the box, or ends in a shot |
| BALL IN THE BOX / PLAYERS IN THE BOX | ball in the box, or ≥ k attackers in the box, for at least T s |
| ATTACKING / DEFENSIVE TRANSITION | a sample of direct turnovers, those followed by a fast advance |
| LONG BALL | pass event longer than L m (Metrica events, DFL events) |
| Mirror codes | a copy of each paired attacking tag with the other team |

**Feasibility** (`external/phase_tag_feasibility.py`, Metrica reception labels and ball x, per
game). The counts are in the right range with a single threshold:

| Per game (games 1 / 2 / 3) | Spells ≥ 8 s | Reach final third | Reach box | Start in own third | Direct turnovers |
|---|---|---|---|---|---|
| Metrica | 121 / 125 / 149 | 93 / 97 / 88 (real 93) | 61 / 50 / 41 at ≥ 8 s, 79 / 61 / 61 at ≥ 3 s (real 72) | 45 / 39 / 56 (real BUILD UP 22: needs a stricter rule) | 222 / 194 / 265 (real transitions 24: about 1 in 9) |

## 3. Corruption (the leak)

Pretend tags built from the answer key's spells would carry exact team and boundaries, so the
model would trust them far more than the real tags deserve. Four kinds of noise, each with a rate:

- **Boundary jitter:** start and end move by a random amount, matched to the edge spread in the
  summary's item 3.
- **Team errors:** the team is swapped on a fraction of tags.
- **Missed tags:** a fraction of qualifying spells get no tag.
- **False tags:** tags placed on the wrong spell, or on another stretch of play.

The rates are calibrated so that the pretend tags' agreement with v4's predicted team on the noisy
copies equals the real tags' agreement on the match (summary item 2). That is aggregate
calibration, as for the ball gaps. Until Monday's numbers exist, a conservative default: jitter
±2 s, 10% team errors, 15% missed, 10% false.

One option: build the tags from the *kick* key rather than the reception key. Its stoppage edges
differ, so the pretend tags don't share the graded edges exactly.

## 4. How v5 would use them

Signed per-frame columns in v4's frame_features: "tagged team attacking". Each active tag gives
+1 for home and −1 for away, multiplied by −1 for defending codes. There are three families, each
its own column because their reliability differs: structure (build-up, progression, final third,
chances), box, and transitions. A fourth column, "a transition tag starts within ±1 s", gives
evidence for a switch. Paired codes are counted once, not twice. The mirrored home/away rows make
the columns symmetric with no extra work, and the scorer learns how much to trust each one.

Phases say nothing directly about dead ball, so the D side stays with v4 and the set pieces.

## 5. Effort

| Step | Who | Time |
|---|---|---|
| Monday summary script (pairing, agreement, edges) | main session | 1–2 h, plus Jordan's run |
| Generator for Metrica and DFL, thresholds fitted to the counts | Track B | ~½ day |
| Corruption calibrated to the summary | Track B | 2–3 h, after the summary |
| v5 columns plus a scored run on seed 0, then the keep rule | Track B / D | ~½ day |

That's about 1½ days end to end. The generator can't be calibrated before the summary exists, so
this realistically comes after the set-piece v5, not before Monday's final pick.
