# External data: Metrica's three open games

We may only train and tune on other games, so this folder turns Metrica's open sample games (github.com/metrica-sports/sample-data) into what we need: an answer key and tracking, both indexed by the same frame numbers. Everything is downloaded into `data/external/metrica/`, which is gitignored.

```bash
.venv/bin/python external/metrica_labels.py      # answer key: game{1,2,3}_labels_reception.csv
.venv/bin/python external/metrica_tracking.py    # tracking: game{1,2,3}_tracking.npz, plus checks
```

## For Person 2: loading the tracking

```python
import sys; sys.path.insert(0, "external")
from metrica_tracking import load_game

game = load_game(1)    # about 15 s the first time (download and convert), then instant
```

`game` is a dict of numpy arrays. Row *i* is Metrica frame *i*, the same row as the answer key CSV:

| key | shape | contents |
|---|---|---|
| `ball` | (frames, 2) | ball x, y in 0–1 pitch units; NaN where it isn't tracked |
| `players` | (frames, players, 2) | each player's x, y in 0–1; NaN when they're not on the pitch |
| `team` | (players,) | `'H'` or `'A'` |
| `player` | (players,) | Metrica's player id within that game |
| `period` | (frames,) | 1 or 2; 0 where there's no tracking |

* Multiply x by 105 and y by 68 to get metres (`metrica_tracking.PITCH`).
* 25 frames per second. Row 0 is always empty, because Metrica frames start at 1.
* Unlike our match, players keep the same column all game; in our match, tracks drop out and swap players.

## Read before building features: the ball is missing for different reasons

In the Metrica games the ball is tracked on 98–99% of live frames and is mostly missing when the ball is dead. In our match the ball is missing on 41% of graded frames, including a lot of live play.

So **"ball missing" means "dead ball" in Metrica, but not in our match.** A model trained on clean Metrica tracking would learn that shortcut and then call live play dead on our match, splitting real possessions.

We avoid it in three ways, together:

1. **Noise (Person 1).** In the Metrica copy, delete the ball during live play too, in bursts, so it's missing about as often as in our match. All training, tuning and scoring uses the noisy copy, which also shows whether a model still falls for the shortcut.
2. **Dead-ball detection from the players (Person 2).** Decide "dead or alive" mainly from the players, not the ball: everyone slowing down, players walking, a wall forming, someone waiting at the touchline. That behaves the same whether or not the ball is tracked, so it carries over to our match. The ball is a supporting clue only.
3. **Ball-gap filling (Person 2).** Estimate where the ball is during short gaps, from its last position and who's nearby, so fewer frames have no ball at all. Long gaps stay missing rather than guessed.

## Noisy copies: what everything trains and scores on

```bash
.venv/bin/python external/measure_ball_gaps.py                                   # on the provided match
.venv/bin/python external/metrica_noise.py --calibration data/ball_gap_stats.json
```

`measure_ball_gaps.py` reads the provided tracking (not the ground truth, not the video) and prints only summary numbers: how often and for how long the ball goes missing, and how much more often on fast balls and in crowds. Run it locally; its printout is safe to share. Without it, `metrica_noise.py` uses defaults.

`metrica_noise.py` writes `data/external/metrica/noisy/game{g}_live{level}_seed{s}.npz`, the same format as above. It deletes the ball for most of long passes, crosses, keeper kicks and throws and shots, plus random gaps that are more likely on fast balls and in crowds. It also shows a still ball during some stoppages, breaks player tracks into anonymous team slots, and jitters positions.

* `live20/35/50`: the share of live frames with no ball. We can't measure the real split between live and dead play without the ground truth, but the provided tracking points to between 20% and 35%: its 41% missing comes in long gaps, and more of them end as a Smart Tagging set piece starts than chance would give (47% against 35%). So **tune on `live20` and `live35`, and use `live50` as a stress test.** Don't trust a model that only works at one level.
* What the provided tracking showed (`measure_ball_gaps.py`): 150 gaps in 106 minutes, median 10.7 s; gaps start most often on a slow or still ball (×1.7), in crowds (×2.6 with 3+ players near), near a goal (×1.8) or a touchline (×1.5), and almost never on a fast ball. The noise copies that pattern.
* `seed0` is for tuning, `seed1` for testing.
* `ball_reason` says why the ball is missing on each frame (0 seen, 1 missing in the clean data, 2 in the air, 3 random gap, 4 still ball shown in a stoppage). Use it for analysis only, never as a model input: our match has no such column.

## Also in the tracking

* **2D ball:** a high ball passes straight through the positions of the players it flies over. Being near the ball is not a touch. A touch shows up as a sudden change in the ball's speed or direction.
* **Clean tracking:** these games are much cleaner than ours, so scores on them are too optimistic until the noise is added.
