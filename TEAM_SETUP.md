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

|            | **Person 1: answer key and scoring**                                                                     | **Person 2: ball and dead-ball signals**                 | **Person 3: possession model**                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Builds     | Labels for the external games (built from their events), splits, noise added to external tracking, results log | Filling ball gaps, dead-ball detection, Smart Tagging features | Control rule, contested-ball handling, possession segmentation, switch timing |
| Answers    | "Is this version really better?"                                                                               | "Where is the ball, and is it in play?"                        | "Who has it, and when did it change?"                                         |
| Later owns | Reproducible pipeline, technical document                                                                      | Optional ball detector on the video                            | Final parameter choices, made on external data                                |

**External data:** Metrica's open sample games (github.com/metrica-sports/sample-data), 3 matches with tracking and events. Game 3 is in the same EPTS format as ours.

**How we compare versions** (decided Nov 11; the script is [evaluation/score_external.py](evaluation/score_external.py)):

1. **Keep a change** only if the headline goes up **and** no single result (game × answer key × noise level, including live50 and the kick key) drops more than 0.02 against the best run so far. The script checks this for you.
2. **Answer key:** reception is the headline; kick is a check.
3. **Headline:** spellF1 on the reception key, averaged over live20 and live35, then averaged over the 3 test games. It is always shown with each game, e.g. 0.58 (0.62 / 0.60 / 0.52). For the final pick, if two finalists are within about 0.02, take the better worst game.
4. **seed 1 and the sealed DFL games** are milestone tests: run them rarely, log every run, and never tweak in response; investigate instead.
5. **Frames:** every frame with tracking. The unlabelled seconds before kick-off and after the whistle count as D.
6. **Log:** one file per run in `results/runs/`, external scores only. `--summary` prints the table. A daily email digest goes out during Nov 13–15.
7. **Baselines:** the nearest player's team, switching after 0.5 s, with "no ball = dead" (A) or "never dead" (B). The better one is the floor.

The fit step sees the two training games at live20/live35 only. Every test game is scored at every level.

**Revised plan (Nov 11):** the first dates slipped, and Nov 13–15 have almost no one available, so anything needing a person goes on Nov 11–12 or Nov 16. Agents work on external data only and never touch the provided match.

| Date      | Who    | What                                                                                       |
| --------- | ------ | ------------------------------------------------------------------------------------------ |
| Nov 11    | P1     | Scoring set up and baselines logged. Start dead-ball and DFL work in parallel              |
| Nov 12    | P1     | Model v1. **Safety-net submission file** from the best version so far, run locally on our match |
| Nov 13–15 | agents | Tuning, robustness checks, DFL sealed test on finalists, technical document draft; daily email |
| Nov 16    | P1     | Pick the final version, run it on our match, validate, finish the document, make the repo public |
| Nov 17    | P1     | Submit in the morning                                                                      |

## 6. What to decide at the first meeting

Two kinds of decision. **Rules** we agree by talking, using the definition in CHALLENGE.md and by inspecting the public 5 minutes. **Numbers** we never pick by hand: we tune them on the external games. Write each rule down and freeze it before anyone sees model scores, otherwise we end up tuning the answer key as well as the model.

### Possession rules

The current rules and the evidence behind them live in **[POSSESSION_RULES.md](POSSESSION_RULES.md)**. Summary below.

| Decision                                                   | Our starting answer                                                                                                                                | Number to tune externally                                     |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Who has the ball**                                 | The team whose player last touched it on purpose (pass, header, tackle, clearance, control). No control needed. Accidental deflections don't count | —                                                            |
| **Contested balls:** tackles, loose balls, scrambles | Stay with the last team with a deliberate touch. Internally the model may use a "contested" state, rewritten to that team at the end               | —                                                            |
| **Where the switch goes**                            | At the new team's first deliberate touch                                                                                                           | —                                                            |
| **Dead ball start**                                  | When the ball crosses a line, or play stops for a foul                                                                                             | Thresholds for "ball off the pitch" and "everyone slows down" |
| **Dead ball end**                                    | When the restart is taken (first touch of the set piece)                                                                                           | —                                                            |
| **Fouls with advantage**                             | Play continues, so no `D`                                                                                                                        | —                                                            |
| **Shots and saves**                                  | A keeper catch counts. A parry is open: deliberate touch or deflection?                                                                            | —                                                            |
| **Restarts**                                         | The team taking the restart has the ball. Throw-ins are tagged `N/A` in Smart Tagging, so we work out the team ourselves                         | —                                                            |
| **Second half**                                      | Frames 110,808–113,741 are graded and all `D` (before the second-half kickoff)                                                                  | —                                                            |

### Prediction policy

| Decision                                          | Our starting answer                                                                                                  | Number to tune externally     |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Ball missing (41% of graded frames)**     | Never write `D` just because the ball is missing; that splits real possessions                                     | Max gap to interpolate across |
| **When to write `X`**                     | Almost never: the scorer turns `X` into our previous label anyway, and macroF1 (the tie-break) counts it as a miss | —                            |
| **Minimum possession length / switch cost** | Penalise flicker, but allow real possessions of about 1 s (they exist in the public 5 minutes)                       | The penalty                   |
| **Smart Tagging**                           | Use it as evidence (set pieces mark restarts, phases carry a team), never copy its intervals directly                | How much weight it gets       |

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
