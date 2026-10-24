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

## Read first
CHALLENGE.md, then DATA.md, then EVALUATION.md (the metric rewards whole possessions, not frame accuracy).
