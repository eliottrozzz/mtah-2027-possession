"""Score a model on the external games: the one number we compare versions on.

Only external data is read (Metrica's open games and their noisy copies). Never the provided
match, its ground truth or its video.

    python evaluation/score_external.py --model models.baselines:NoBallDead
    python evaluation/score_external.py --model models.baselines:NoBallDead --seed 1   # milestone
    python evaluation/score_external.py --summary                                      # the log
    python evaluation/score_external.py --sealed --oracle                 # check the sealed set
    python evaluation/score_external.py --sealed --model ... --confirm-milestone      # spent
    python evaluation/score_external.py --dfl-test --model ...      # once per candidate (see below)

How it works (agreed 2026-11-11, see TEAM_SETUP.md):
  * Train on two games, test on the third, three rounds. fit() sees the training games at the
    tuning levels (live20, live35) with the reception answer key; predict() then runs on the test
    game at every level.
  * HEADLINE = spellF1 on the reception key, averaged over live20 and live35, averaged over the 3
    test games. Always shown with each game's own number.
  * Checks, never tuned towards: the kick key, live50, and the clean tracking (info only).
  * Frames: every frame with tracking. Unlabelled frames (before kick-off, after the whistle) count
    as D; the two halves are still scored separately.
  * Keep a change only if the headline goes up AND no single result (game x key x level, spellF1,
    clean excluded) drops more than 0.02 against the best run so far. This script checks that.
  * seed 0 is for tuning. seed 1 is the milestone test: run it rarely, log every run, and never
    tweak in response to it.

SEALED TEST (--sealed): the 7 DFL games (external/dfl_*.py), a separate milestone number.
  * fit() sees all 3 Metrica games at the tuning levels (seed 0) and NOTHING from DFL; predict()
    then runs on every DFL game at every level, scored on both keys.
  * SEALED HEADLINE = spellF1 on the reception key, averaged over live20 and live35, averaged over
    the 7 DFL games. It is never mixed with the Metrica headline and never part of the keep rule.
  * Only for the 2-3 final candidates, around Nov 14-15. Nobody looks at a real model's DFL score
    before then, and nothing is tweaked in response to it. A real model needs --confirm-milestone;
    every such run is logged (seed "sealed") and listed separately by --summary.
  * --oracle predicts the answer key itself: the reception key must score 1.0. Never logged.
  * SPENT: looks 1 (v3, v3.1, two-tier) and 2 (v4) were the two looks. A real model can no longer
    be scored on all 7 games, since 3 of them are now the DFL test set below. --oracle still works.

DFL SPLIT (Jordan, 2026-11-14, after both sealed looks): 4 TRAIN and 3 TEST matches, fixed by a
rule, not by scores. TEST = the 3 highest match IDs among the 5 sync-holdout matches (the 2 sync
development matches, external/dfl_sync.DEV, are never test). TRAIN = the other 4.
  * TRAIN (DFL_TRAIN): free to fit and tune on. dfl_train_set() gives their copies (live20/35,
    seed 0) with reception answer keys as (tracking, key) pairs, the same format as Metrica's.
  * TEST (--dfl-test): scored like the sealed set (all levels, both keys, headline = reception,
    live20/35, mean over the 3 games), logged with seed "dfl-test" and listed by --summary.
    fit() sees the 3 Metrica games; add --fit-dfl-train to also give it the 4 TRAIN matches.
    ONE run per candidate, only after it passes both Metrica seeds: the script refuses a model
    with no logged seed-1 run, or one that already has a dfl-test run (--again overrides; say why
    in --note). Nothing is tuned in response to a test score.

Each run writes one file to results/runs/ (one per run, so parallel work doesn't conflict).
"""

import argparse
import concurrent.futures
import contextlib
import datetime
import getpass
import glob
import importlib
import io
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
import possession_metrics as pm  # noqa: E402
import metrica_tracking as mt  # noqa: E402
import dfl_tracking as dfl  # noqa: E402

# Params
GAMES = (1, 2, 3)
KEYS = ("reception", "kick")
HEADLINE_KEY = "reception"
TUNING_LEVELS = (20, 35)
STRESS_LEVELS = (50,)
INFO_LEVELS = ("clean",)              # shown, never part of the keep rule
MAX_DROP = 0.02                       # largest drop allowed on any single result
CLEAN_MAX_DROP = 0.05                 # ... and on any clean result: a change that only helps the
                                      # noisy copies may be fitting our noise model, not football
BOOTSTRAP_N = 200                     # for the headline cells only
RUNS_DIR = os.path.join(ROOT, "results", "runs")
LABEL_PATH = os.path.join(mt.OUT_DIR, "game{}_labels_{}.csv")
NOISY_PATH = os.path.join(mt.OUT_DIR, "noisy", "game{}_live{}_seed{}.npz")
SEALED_GAMES = dfl.MATCHES            # the 7 DFL games: test only, never fitted on
SEALED_SEED = 0                       # the only noisy DFL copies there are
SEALED_LABEL_PATH = os.path.join(dfl.OUT_DIR, "{}_labels_{}.csv")
SEALED_NOISY_PATH = os.path.join(dfl.OUT_DIR, "noisy", "{}_live{}_seed{}.npz")
# DFL split (see the docstring). Fixed by rule: the 3 highest IDs among the 5 sync-holdout matches.
DFL_SYNC_DEV = ("J03WMX", "J03WOH")            # = external/dfl_sync.DEV (checked in test_split)
DFL_TEST = tuple(sorted(m for m in dfl.MATCHES if m not in DFL_SYNC_DEV)[-3:])
DFL_TRAIN = tuple(m for m in dfl.MATCHES if m not in DFL_TEST)
assert DFL_TEST == ("J03WPY", "J03WQQ", "J03WR9"), DFL_TEST
assert DFL_TRAIN == ("J03WMX", "J03WN1", "J03WOH", "J03WOY"), DFL_TRAIN
DFL_TEST_SEED = "dfl-test"                      # the log's seed field for test-3 runs


# -- data --------------------------------------------------------------------------------------

def load_tracking(game, level, seed, sealed=False):
    if level == "clean":
        return dfl.load_game(game) if sealed else mt.load_game(game)
    path = (SEALED_NOISY_PATH if sealed else NOISY_PATH).format(game, level, seed)
    if not os.path.exists(path):
        sys.exit("missing {}: run external/{}_noise.py --calibration data/ball_gap_stats.json"
                 .format(os.path.relpath(path, ROOT), "dfl" if sealed else "metrica"))
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != "ball_reason"}


def load_answer_key(game, key, period, sealed=False):
    """Answer-key codes, one per tracking frame.

    X frames inside a half become D (players lining up or walking off: the organisers grade the
    same moment in our match as D). Frames with no tracking stay X, and the first frame of the
    second half is set to X so the two halves are scored separately, as in the real grading.
    """
    path = (SEALED_LABEL_PATH if sealed else LABEL_PATH).format(game, key)
    if not os.path.exists(path):
        sys.exit("missing {}: run external/{}_labels.py --restart {}".format(
            os.path.relpath(path, ROOT), "dfl" if sealed else "metrica", key))
    labels = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str, usecols=1)
    n = len(period)
    labels = np.concatenate([labels, np.full(max(0, n - len(labels)), "X")])[:n]
    labels = np.where((labels == "X") & (period > 0), "D", labels)
    labels[period == 0] = "X"
    second_half = np.flatnonzero(period == 2)
    if len(second_half):
        labels[second_half[0]] = "X"
    return pm.label_codes(labels)


def metrica_train_set(levels=TUNING_LEVELS, seed=0):
    """[(tracking, reception key)] for the 3 Metrica games: what fit() sees everywhere."""
    out = []
    for g in GAMES:
        for level in levels:
            tracking = load_tracking(g, level, seed)
            out.append((tracking, load_answer_key(g, HEADLINE_KEY, tracking["period"])))
    return out


def dfl_train_set(levels=TUNING_LEVELS, key=HEADLINE_KEY):
    """[(tracking, answer key)] for the 4 DFL TRAIN matches (seed 0, the only DFL copies), same
    format as Metrica's. Never includes a DFL_TEST match."""
    out = []
    for match in DFL_TRAIN:
        assert match not in DFL_TEST
        for level in levels:
            tracking = load_tracking(match, level, SEALED_SEED, sealed=True)
            out.append((tracking, load_answer_key(match, key, tracking["period"], sealed=True)))
    return out


def load_model(spec):
    """'package.module:ClassName' -> an instance."""
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)()


# -- scoring -----------------------------------------------------------------------------------

def score_cell(gt, pred, bootstrap):
    result = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=BOOTSTRAP_N if bootstrap else 0)
    cell = {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_gt_spells": result["n_gt_spells"], "n_pred_spells": result["n_pred_spells"]}
    if bootstrap:
        cell["spell_f1_range"] = [result["spell_bootstrap_low"], result["spell_bootstrap_high"]]
    return cell


def run_fold(model, test, seed, levels):
    """Fit on the two other games, predict game `test` at every level. Returns that game's cells.

    Loads its own tracking, so a worker process never receives big arrays through pickling. Only
    the training games and one test level are held in memory at a time.
    """
    train = []
    for g in GAMES:
        if g == test:
            continue
        for level in TUNING_LEVELS:
            tracking = load_tracking(g, level, seed)
            train.append((tracking, load_answer_key(g, HEADLINE_KEY, tracking["period"])))
    model.fit(train)
    del train

    cells = {key: {} for key in KEYS}
    gts = None
    for level in levels:
        tracking = load_tracking(test, level, seed)
        if gts is None:
            gts = {key: load_answer_key(test, key, load_tracking(test, levels[0], seed)["period"]
                                        if level != levels[0] else tracking["period"])
                   for key in KEYS}
        pred = model.predict(tracking)
        del tracking
        for key in KEYS:
            gt = gts[key]
            if len(pred) != len(gt):
                raise ValueError("model returned {} frames, game {} has {}".format(
                    len(pred), test, len(gt)))
            bootstrap = key == HEADLINE_KEY and level in TUNING_LEVELS
            cells[key][str(level)] = score_cell(gt, pred, bootstrap)
    return cells


def _fold_worker(spec, test, seed, levels):
    """Process-pool entry: a fresh model, its printed output captured, and its fitted settings."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = load_model(spec)
        cells = run_fold(model, test, seed, levels)
    return cells, buffer.getvalue(), model.settings()


def run(spec, seed, levels, jobs=len(GAMES)):
    """({game: {key: {level: cell}}} for every test game, the fitted settings).

    jobs 1 scores the folds one after another with a single model instance. Otherwise each fold
    runs in its own process with its own model instance; their printed output is replayed in game
    order so the log looks the same. The settings are those of the last fold, as before.
    """
    cells = {}
    if jobs <= 1:
        model = load_model(spec)
        for test in GAMES:
            cells[test] = run_fold(model, test, seed, levels)
            print("  game {} done".format(test))
        return cells, model.settings()

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(GAMES))) as pool:
        futures = {test: pool.submit(_fold_worker, spec, test, seed, levels) for test in GAMES}
        for test in GAMES:
            cells[test], output, settings = futures[test].result()
            print(output, end="")
            print("  game {} done".format(test))
    return cells, settings


def run_sealed(spec, levels, oracle=False, games=SEALED_GAMES, fit_dfl_train=False):
    """{dfl game: {key: {level: cell}}}, after one fit on all three Metrica games, plus the 4 DFL
    TRAIN matches if fit_dfl_train (or no fit, for the oracle, which predicts the reception key
    itself). No scored DFL game ever reaches fit()."""
    model = None
    if not oracle:
        model = load_model(spec)
        train = metrica_train_set()
        if fit_dfl_train:
            assert not set(DFL_TRAIN) & set(games), "a scored DFL game would be fitted on"
            train += dfl_train_set()
        model.fit(train)
        del train
    letters = np.array(["H", "A", "D"])
    cells = {}
    for game in games:
        period = load_tracking(game, "clean", SEALED_SEED, sealed=True)["period"]
        gts = {key: load_answer_key(game, key, period, sealed=True) for key in KEYS}
        cells[game] = {key: {} for key in KEYS}
        for level in levels:
            if oracle:
                codes = gts[HEADLINE_KEY]
                pred = np.where((codes >= 0) & (codes < 3), letters[np.clip(codes, 0, 2)], "D")
            else:
                pred = model.predict(load_tracking(game, level, SEALED_SEED, sealed=True))
            for key in KEYS:
                if len(pred) != len(gts[key]):
                    raise ValueError("model returned {} frames, DFL game {} has {}".format(
                        len(pred), game, len(gts[key])))
                bootstrap = key == HEADLINE_KEY and level in TUNING_LEVELS
                cells[game][key][str(level)] = score_cell(gts[key], pred, bootstrap)
        print("  DFL game {} done".format(game))
    return cells, (model.settings() if model else {})


def headline(cells):
    per_game = {g: float(np.mean([cells[g][HEADLINE_KEY][str(l)]["spell_f1"] for l in TUNING_LEVELS]))
                for g in cells}
    return float(np.mean(list(per_game.values()))), per_game


def checked_results(cells):
    """(game, key, level) -> spellF1 for every result the keep rule looks at."""
    return {(g, k, l): cells[g][k][l]["spell_f1"] for g in cells for k in KEYS
            for l in cells[g][k] if l not in INFO_LEVELS}


# -- the log -----------------------------------------------------------------------------------

def git(*args):
    try:
        return subprocess.check_output(["git"] + list(args), cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_runs(seed=None):
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(path) as handle:
            entry = json.load(handle)
        entry["_file"] = os.path.basename(path)
        if seed is None or entry["seed"] == seed:
            runs.append(entry)
    return runs


def cells_from_log(entry):
    return {(g if entry["seed"] in ("sealed", DFL_TEST_SEED) else int(g)): v
            for g, v in entry["cells"].items()}


def retired_runs():
    """Run files listed in results/retired.txt: versions we decided against for football reasons.

    They stay in the log but are never the bar a new run has to beat. One file name per line,
    then '#' and the reason.
    """
    path = os.path.join(ROOT, "results", "retired.txt")
    if not os.path.exists(path):
        return set()
    with open(path) as handle:
        return {line.split("#")[0].strip() for line in handle if line.split("#")[0].strip()}


def keep_rule(cells, seed):
    """Compare against the best earlier run on the same seed. None if there is nothing to beat."""
    retired = retired_runs()
    earlier = [e for e in load_runs(seed) if e["_file"] not in retired]
    if not earlier:
        return None
    best = max(earlier, key=lambda e: e["headline"])
    best_cells = checked_results(cells_from_log(best))
    mine = checked_results(cells)
    drops = {"game{} {} live{}".format(*k): round(best_cells[k] - mine[k], 4)
             for k in mine if k in best_cells and best_cells[k] - mine[k] > MAX_DROP}
    best_clean, mine_clean = clean_results(cells_from_log(best)), clean_results(cells)
    clean_drops = {"game{} {} clean".format(*k): round(best_clean[k] - mine_clean[k], 4)
                   for k in mine_clean
                   if k in best_clean and best_clean[k] - mine_clean[k] > CLEAN_MAX_DROP}
    drops.update(clean_drops)
    gain = headline(cells)[0] - best["headline"]
    return {"against": best["_file"], "against_model": best["model"],
            "headline_gain": round(gain, 4), "drops_over_limit": drops,
            "keep": gain > 0 and not drops}


def clean_results(cells):
    """{(game, key): spellF1} on the clean copies."""
    return {(g, k): cells[g][k]["clean"]["spell_f1"] for g in cells for k in cells[g]
            if "clean" in cells[g][k]}


def show(entry):
    cells = cells_from_log(entry)
    games = list(cells)
    title = {"sealed": ("SEALED DFL TEST", "SEALED HEADLINE"),
             DFL_TEST_SEED: ("DFL TEST-3" + (" (fit +DFL train)" if entry.get("fit_dfl_train")
                                             else ""), "DFL TEST HEADLINE")}
    label, head = title.get(entry["seed"], ("seed {}".format(entry["seed"]), "headline"))
    print("\n{}  {}  {} {:.3f} ({})".format(
        entry["model"], label, head, entry["headline"],
        " / ".join("{:.2f}".format(entry["per_game"][str(g)]) for g in games)))
    levels = list(cells[games[0]][HEADLINE_KEY].keys())
    print("{:<12}".format("spellF1") + "".join(
        "{:>16}".format("{} live{}".format(k[:4], l) if l != "clean" else "{} clean".format(k[:4]))
        for k in KEYS for l in levels))
    for g in games:
        print("{:<12}".format("game {}".format(g)) + "".join(
            "{:>16.3f}".format(cells[g][k][l]["spell_f1"]) for k in KEYS for l in levels))
    print("{:<12}".format("macroF1") + "".join(
        "{:>16.3f}".format(np.mean([cells[g][k][l]["macro_f1"] for g in games]))
        for k in KEYS for l in levels))
    rule = entry.get("keep_rule")
    if rule:
        verdict = "KEEP" if rule["keep"] else "DON'T KEEP"
        print("against best so far ({}): headline {:+.3f}, drops over {} ({} clean): {} -> {}"
              .format(rule["against_model"], rule["headline_gain"], MAX_DROP, CLEAN_MAX_DROP,
            rule["drops_over_limit"] or "none", verdict))
    elif entry["seed"] in ("sealed", DFL_TEST_SEED):
        print("{}: no keep rule, and nothing gets tweaked in response".format(
            "sealed milestone" if entry["seed"] == "sealed" else "DFL test-3 (one run per candidate)"))
    else:
        print("first run on this seed: nothing to compare against")


def summary():
    for seed, title in ((0, "TUNING RUNS (seed 0)"), (1, "MILESTONE RUNS (seed 1)"),
                        ("sealed", "SEALED DFL RUNS (milestone, separate from the headline)"),
                        (DFL_TEST_SEED, "DFL TEST-3 RUNS ({}; one per candidate)".format(
                            ", ".join(DFL_TEST)))):
        runs = sorted(load_runs(seed), key=lambda e: -e["headline"])
        retired = retired_runs()
        print("\n" + title)
        if not runs:
            print("  none yet")
        for e in runs:
            rule = e.get("keep_rule") or {}
            print("{} {:.3f} ({})  {:<28} {:<10} {}  {}  {}".format(
                "R" if e["_file"] in retired else " ",
                e["headline"], " / ".join("{:.2f}".format(v) for v in e["per_game"].values()),
                e["model"], e["who"], e["commit"][:7] + ("*" if e["dirty"] else ""),
                e["time"][:16], "keep" if rule.get("keep") else ("-" if not rule else "no")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="module:Class, e.g. models.baselines:NoBallDead")
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1),
                        help="0 = tuning (default), 1 = milestone test, run rarely")
    parser.add_argument("--note", default="", help="one line for the log")
    parser.add_argument("--who", default=getpass.getuser(), help="who ran it (person or agent)")
    parser.add_argument("--jobs", type=int, default=len(GAMES),
                        help="folds scored in parallel processes (default 3; 1 = one after another)")
    parser.add_argument("--no-log", action="store_true", help="print only, write nothing")
    parser.add_argument("--sealed", action="store_true",
                        help="score on the 7 sealed DFL games (milestone; see the docstring)")
    parser.add_argument("--oracle", action="store_true",
                        help="with --sealed: predict the answer key itself (must score 1.0)")
    parser.add_argument("--confirm-milestone", action="store_true",
                        help="with --sealed and a real model: yes, this is a final candidate")
    parser.add_argument("--dfl-test", action="store_true",
                        help="score on the 3 DFL TEST matches (once per candidate; see docstring)")
    parser.add_argument("--fit-dfl-train", action="store_true",
                        help="with --dfl-test: fit on Metrica plus the 4 DFL TRAIN matches")
    parser.add_argument("--again", action="store_true",
                        help="with --dfl-test: allow a second test run of the same model name")
    parser.add_argument("--summary", action="store_true", help="print the log and stop")
    args = parser.parse_args()

    if args.summary:
        summary()
        return
    if args.sealed and args.dfl_test:
        parser.error("--sealed and --dfl-test are separate modes")
    if args.sealed or args.dfl_test:
        sealed_main(args)
        return
    if args.oracle:
        parser.error("--oracle only works with --sealed or --dfl-test")
    if args.fit_dfl_train or args.again:
        parser.error("--fit-dfl-train and --again only work with --dfl-test")
    if not args.model:
        parser.error("--model is required (or use --summary)")

    model = load_model(args.model)
    levels = list(TUNING_LEVELS) + list(STRESS_LEVELS) + list(INFO_LEVELS)
    print("scoring {} on seed {}".format(model.name, args.seed))
    cells, settings = run(args.model, args.seed, levels, args.jobs)
    total, per_game = headline(cells)

    entry = {
        "model": model.name, "spec": args.model, "settings": settings,
        "seed": args.seed, "who": args.who, "note": args.note,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "models", "evaluation", "external")),
        "headline": total, "per_game": {str(g): v for g, v in per_game.items()},
        "cells": {str(g): v for g, v in cells.items()},
    }
    entry["keep_rule"] = keep_rule(cells, args.seed)
    show(entry)

    if not args.no_log:
        os.makedirs(RUNS_DIR, exist_ok=True)
        name = "{}_{}_seed{}.json".format(
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), model.name, args.seed)
        with open(os.path.join(RUNS_DIR, name), "w") as handle:
            json.dump(entry, handle, indent=1)
        print("logged results/runs/" + name)


def dfl_test_allowed(name, spec, again):
    """None if the model (by name, or by module:Class spec, since models get renamed) may have its
    one DFL test-3 run, else the reason it may not."""
    def same(e):
        return e["model"] == name or e.get("spec") == spec
    if not any(same(e) for e in load_runs(1)):
        return ("{} ({}) has no logged Metrica seed-1 run. The DFL test is only for candidates "
                "that passed both Metrica seeds.".format(name, spec))
    done = [e["_file"] for e in load_runs(DFL_TEST_SEED) if same(e)]
    if done and not again:
        return ("{} already has a DFL test-3 run ({}). One run per candidate; --again only with "
                "a reason in --note.".format(name, ", ".join(done)))
    return None


def sealed_main(args):
    levels = list(TUNING_LEVELS) + list(STRESS_LEVELS) + list(INFO_LEVELS)
    test = args.dfl_test
    games = DFL_TEST if test else SEALED_GAMES
    seed_tag = DFL_TEST_SEED if test else "sealed"
    if args.fit_dfl_train and not test:
        sys.exit("--fit-dfl-train only works with --dfl-test")
    if args.oracle:
        name, spec = "oracle_answer_key", None
    else:
        mode = "--dfl-test" if test else "--sealed"
        if not args.model:
            sys.exit("{} needs --model (or --oracle)".format(mode))
        name, spec = load_model(args.model).name, args.model
        if test:
            reason = dfl_test_allowed(name, spec, args.again)
            if reason:
                sys.exit(reason)
        else:
            sys.exit("--sealed is spent: looks 1 and 2 were the two looks, and 3 of its games are "
                     "now the DFL test set. Use --dfl-test (see the docstring). --oracle still works.")
    print("{}: scoring {} on {} games ({}), fit on Metrica{}".format(
        "DFL TEST-3" if test else "SEALED DFL TEST", name, len(games), ", ".join(games),
        " + the 4 DFL TRAIN matches" if args.fit_dfl_train else " only"))
    cells, settings = run_sealed(spec, levels, oracle=args.oracle, games=games,
                                 fit_dfl_train=args.fit_dfl_train)
    total, per_game = headline(cells)
    entry = {
        "model": name, "spec": spec, "settings": settings, "seed": seed_tag,
        "fit_dfl_train": bool(args.fit_dfl_train), "dfl_games": list(games),
        "who": args.who, "note": args.note,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "models", "evaluation", "external")),
        "headline": total, "per_game": {str(g): v for g, v in per_game.items()},
        "cells": {str(g): v for g, v in cells.items()},
    }
    show(entry)
    if args.oracle:
        ok = all(abs(cells[g][HEADLINE_KEY][l]["spell_f1"] - 1.0) < 1e-9 for g in cells
                 for l in cells[g][HEADLINE_KEY])
        print("oracle check: reception key scores 1.0 everywhere -> {}  (not logged)".format(
            "PASS" if ok else "FAIL"))
        if not ok:
            sys.exit(1)
        return
    if not args.no_log:
        os.makedirs(RUNS_DIR, exist_ok=True)
        file_name = "{}_{}_{}.json".format(
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), name,
            "dfltest" if test else "sealed")
        with open(os.path.join(RUNS_DIR, file_name), "w") as handle:
            json.dump(entry, handle, indent=1)
        print("logged results/runs/" + file_name)


if __name__ == "__main__":
    main()
