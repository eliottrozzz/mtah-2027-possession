# Team setup: More than a Hack 2027

Our private working copy of the official challenge repo. We make it public just before the Nov 17 deadline (the rules require a public repo for the submission).

## Key dates
* **Nov 10**: everyone registers with the exact same team name
* **Nov 14**: submission instructions arrive
* **Nov 17, 23:59 CET**: submission deadline

## 1. Get the code (5 min)

Accept the GitHub invite first. Using GitHub Desktop? File > Clone Repository, pick `eliottrozzz/mtah-2027-possession`, then open it in Terminal for the rest. Otherwise:

```bash
git clone https://github.com/eliottrozzz/mtah-2027-possession.git
cd mtah-2027-possession
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-starter.txt
python evaluation/run_evaluation.py
```

If you see the scoreboard with 4 baselines, you're good. This works with no data at all.

## 2. Get the data (no 20 GB download)

The data lives in the shared Drive folder "More Than A Hack 2027 : Initial Challenge".

1. In Drive, right click that folder, then **Organise > Add shortcut**, and put it in My Drive.
2. Install **Google Drive for desktop** and sign in with the account the folder was shared with.
3. Point the repo at it (Mac path shown, adjust the email and shortcut name):

```bash
echo 'export MTAH_DATA_DIR="$HOME/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/My Drive/More Than A Hack 2027 - Initial Challenge"' >> ~/.zshrc
source ~/.zshrc
```

4. In Drive for desktop, right click the `tactical` folder and choose **Available offline**. Skip `broadcast` unless you need it (the video alone is 14 GB).

Sizes: tactical video 5.2 GB, tracking 196 MB, homography 23 MB, tagging 0.3 MB. Most approaches only need the non video files.

Check it works:

```bash
python starter/baseline.py
```

## 3. How we work

* Never commit anything from `data/` or any frame level export. Data is licensed for the competition only.
* One branch per idea: `git checkout -b name/idea`, then open a PR to `main`.
* Submission CSVs go in `evaluation/results/` named `name_idea.csv`. That folder is gitignored, so share the score, not the file. The rules say the repo must regenerate every submission anyway.
* Always run `python evaluation/validate_submission.py <file>.csv` before sharing a score.
* Get official updates (run once: `git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git`), then: `git fetch upstream && git merge upstream/main`

## 4. Rules that override the docs

The organisers' clarification email (early Nov) takes precedence over RULES.md and FAQ.md:

* Training, fitting and tuning use **external data only**. Never the ground truth, never the provided video.
* No checking our output against the video. Watching it to understand what counts as possession is fine.
* No uploading the provided data to AI services.
* The public 5 minutes are for checking the pipeline works and for understanding the definition, never for picking parameters or choosing between versions.

## 5. Who does what

Names TBD at the first meeting. Each role's first version should be usable on its own, so nobody blocks anyone.

| | **Person 1: answer key and scoring** | **Person 2: ball and dead-ball signals** | **Person 3: possession model** |
|---|---|---|---|
| Builds | Labels for the external games (built from their events), splits, noise added to external tracking, results log | Filling ball gaps, dead-ball detection, Smart Tagging features | Control rule, contested-ball handling, possession segmentation, switch timing |
| Answers | "Is this version really better?" | "Where is the ball, and is it in play?" | "Who has it, and when did it change?" |
| Later owns | Reproducible pipeline, technical document | Optional ball detector on the video | Final parameter choices, made on external data |

**External data:** Metrica's open sample games (github.com/metrica-sports/sample-data), 3 matches with tracking and events. Game 3 is in the same EPTS format as ours.

**How we compare versions:** train on 2 external games, score on the 3rd, rotate so each game is tested once. Tune parameters inside the 2 training games (hold out one half at a time). A gap under 0.04 spellF1 is a tie, so keep the simpler version.

| Date | Who | What |
|---|---|---|
| Nov 5 | All | First meeting: the decisions in section 6 |
| Nov 7 | P1 | Baseline scored on the external games |
| Nov 8 | P2, P3 | First versions of the signals and the model |
| Nov 10 | All | Pipeline runs end to end. Register with the same team name |
| Nov 13 | All | Feature freeze: only bug fixes and external tuning after this |
| Nov 15 | P1 | Technical document done; a fresh clone reproduces the same file |
| Nov 16 | All | Final file validated; repo made public, no match data in it |
| Nov 17 | P1 | Submit in the morning, the other two check the upload |

## 6. What to decide at the first meeting

Two kinds of decision. **Rules** we agree by talking, using the definition in CHALLENGE.md and by inspecting the public 5 minutes. **Numbers** we never pick by hand: we tune them on the external games. Write each rule down and freeze it before anyone sees model scores, otherwise we end up tuning the answer key as well as the model.

### Possession rules

The current rules and the evidence behind them live in **[POSSESSION_RULES.md](POSSESSION_RULES.md)**. Summary below.

| Decision | Our starting answer | Number to tune externally |
|---|---|---|
| **Who has the ball** | The team whose player last touched it on purpose (pass, header, tackle, clearance, control). No control needed. Accidental deflections don't count | — |
| **Contested balls:** tackles, loose balls, scrambles | Stay with the last team with a deliberate touch. Internally the model may use a "contested" state, rewritten to that team at the end | — |
| **Where the switch goes** | At the new team's first deliberate touch | — |
| **Dead ball start** | When the ball crosses a line, or play stops for a foul | Thresholds for "ball off the pitch" and "everyone slows down" |
| **Dead ball end** | When the restart is taken (first touch of the set piece) | — |
| **Fouls with advantage** | Play continues, so no `D` | — |
| **Shots and saves** | A keeper catch counts. A parry is open: deliberate touch or deflection? | — |
| **Restarts** | The team taking the restart has the ball. Throw-ins are tagged `N/A` in Smart Tagging, so we work out the team ourselves | — |
| **Second half** | Frames 110,808–113,741 are graded and all `D` (before the second-half kickoff) | — |

### Prediction policy

| Decision | Our starting answer | Number to tune externally |
|---|---|---|
| **Ball missing (41% of graded frames)** | Never write `D` just because the ball is missing; that splits real possessions | Max gap to interpolate across |
| **When to write `X`** | Almost never: the scorer turns `X` into our previous label anyway, and macroF1 (the tie-break) counts it as a miss | — |
| **Minimum possession length / switch cost** | Penalise flicker, but allow real possessions of about 1 s (they exist in the public 5 minutes) | The penalty |
| **Smart Tagging** | Use it as evidence (set pieces mark restarts, phases carry a team), never copy its intervals directly | How much weight it gets |

### Answer key for the external games (P1, agreed by all)

* Map Metrica event types to the rules above: which of PASS, CARRY, RECOVERY, CHALLENGE, BALL LOST, BALL OUT, SET PIECE, FAULT RECEIVED start, keep or end a possession.
* Noise added to external tracking so it looks like ours: ball missing in bursts at roughly the published 41% (test across 30–60%), dropped and swapped player tracks.
* Folds, random seeds, and the results log format: version name, external spellF1 per game, possession count, one line on what changed.

### How we work together

* The feature table: which columns P2 produces and P3 reads, and where it lives (gitignored).
* AI tools: no match data or row-level outputs pasted into hosted AI tools.
* Anything near a rule line: we only do it if we'd describe it in the technical document.

## Read first
CHALLENGE.md, then DATA.md, then EVALUATION.md (the metric rewards whole possessions, not frame accuracy).
