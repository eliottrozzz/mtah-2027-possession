"""Score a model on the external games: the one number we compare versions on.

Only external data is read (Metrica's open games and their noisy copies). Never the provided
match, its ground truth or its video.

    python evaluation/score_external.py --model models.baselines:NoBallDead
    python evaluation/score_external.py --model models.baselines:NoBallDead --seed 1   # milestone
    python evaluation/score_external.py --summary                                      # the log

How it works (agreed 2026-11-11):
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
import re
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
VERSIONS_DOC = os.path.join(ROOT, "docs", "model_versions.md")
LABEL_PATH = os.path.join(mt.OUT_DIR, "game{}_labels_{}.csv")
NOISY_PATH = os.path.join(mt.OUT_DIR, "noisy", "game{}_live{}_seed{}.npz")


# -- data --------------------------------------------------------------------------------------

def load_tracking(game, level, seed):
    if level == "clean":
        return mt.load_game(game)
    path = NOISY_PATH.format(game, level, seed)
    if not os.path.exists(path):
        sys.exit("missing {}: run external/metrica_noise.py --calibration data/ball_gap_stats.json"
                 .format(os.path.relpath(path, ROOT)))
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != "ball_reason"}


def load_answer_key(game, key, period):
    """Answer-key codes, one per tracking frame.

    X frames inside a half become D (players lining up or walking off: the organisers grade the
    same moment in our match as D). Frames with no tracking stay X, and the first frame of the
    second half is set to X so the two halves are scored separately, as in the real grading.
    """
    path = LABEL_PATH.format(game, key)
    if not os.path.exists(path):
        sys.exit("missing {}: run external/metrica_labels.py --restart {}".format(
            os.path.relpath(path, ROOT), key))
    labels = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str, usecols=1)
    n = len(period)
    labels = np.concatenate([labels, np.full(max(0, n - len(labels)), "X")])[:n]
    labels = np.where((labels == "X") & (period > 0), "D", labels)
    labels[period == 0] = "X"
    second_half = np.flatnonzero(period == 2)
    if len(second_half):
        labels[second_half[0]] = "X"
    return pm.label_codes(labels)


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
    return {int(g): v for g, v in entry["cells"].items()}


def version_name(run_name):
    """A run's model name as its version (v3.1, v4, ...) from docs/model_versions.md's table, so a
    run logged under an old name (e.g. v4_proto_cap) shows as the version it became. Names that
    never became a version come back unchanged."""
    try:
        with open(VERSIONS_DOC) as handle:
            for line in handle:
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cols) < 3 or not cols[0].lstrip("*").startswith("v"):
                    continue
                if run_name in re.findall(r"`([^`]+)`", cols[2]):
                    return cols[0].strip("*").split()[0].strip("*")
    except OSError:
        pass
    return run_name


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
    print("\n{}  seed {}  headline {:.3f} ({})".format(
        entry["model"], entry["seed"], entry["headline"],
        " / ".join("{:.2f}".format(entry["per_game"][str(g)]) for g in GAMES)))
    levels = list(cells[GAMES[0]][HEADLINE_KEY].keys())
    print("{:<12}".format("spellF1") + "".join(
        "{:>16}".format("{} live{}".format(k[:4], l) if l != "clean" else "{} clean".format(k[:4]))
        for k in KEYS for l in levels))
    for g in GAMES:
        print("{:<12}".format("game {}".format(g)) + "".join(
            "{:>16.3f}".format(cells[g][k][l]["spell_f1"]) for k in KEYS for l in levels))
    print("{:<12}".format("macroF1") + "".join(
        "{:>16.3f}".format(np.mean([cells[g][k][l]["macro_f1"] for g in GAMES]))
        for k in KEYS for l in levels))
    rule = entry.get("keep_rule")
    if rule:
        verdict = "KEEP" if rule["keep"] else "DON'T KEEP"
        stamp = os.path.basename(rule.get("against", ""))[:15]
        print("against best so far ({}, run {}): headline {:+.3f}, drops over {} ({} clean): "
              "{} -> {}".format(version_name(rule["against_model"]), stamp, rule["headline_gain"],
                      MAX_DROP, CLEAN_MAX_DROP,
            rule["drops_over_limit"] or "none", verdict))
    else:
        print("first run on this seed: nothing to compare against")


def summary():
    for seed, title in ((0, "TUNING RUNS (seed 0)"), (1, "MILESTONE RUNS (seed 1)")):
        runs = sorted(load_runs(seed), key=lambda e: -e["headline"])
        retired = retired_runs()
        print("\n" + title)
        if not runs:
            print("  none yet")
        for e in runs:
            rule = e.get("keep_rule") or {}
            print("{} {:.3f} ({})  {:<28} {:<10} {}  {}  {}".format(
                "R" if e["_file"] in retired else " ",
                e["headline"], " / ".join("{:.2f}".format(e["per_game"][str(g)]) for g in GAMES),
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
    parser.add_argument("--summary", action="store_true", help="print the log and stop")
    args = parser.parse_args()

    if args.summary:
        summary()
        return
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


if __name__ == "__main__":
    main()
