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

So **"ball missing" means "dead ball" in Metrica, but not in our match.** A model trained on clean Metrica tracking would learn that shortcut and then call live play dead on our match, splitting real possessions. Before training or tuning, we delete the ball during live play in the Metrica copy (the noise step), so the model has to use other clues for a dead ball: players slowing down, the ball still, a set piece forming.

## Also in the tracking

* **2D ball:** a high ball passes straight through the positions of the players it flies over. Being near the ball is not a touch. A touch shows up as a sudden change in the ball's speed or direction.
* **Clean tracking:** these games are much cleaner than ours, so scores on them are too optimistic until the noise is added.
