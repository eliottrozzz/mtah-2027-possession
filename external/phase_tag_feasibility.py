"""Phase-tag feasibility (Metrica, external data only): how many possession spells per game
pass simple zone and duration rules, to compare with the real Smart Tagging phase counts in
DATA.md. Uses the reception labels and tracking ball x. Prints counts only.

    python external/phase_tag_feasibility.py
"""
import numpy as np, pandas as pd, sys
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "metrica") + os.sep
for g in (1, 2, 3):
    d = np.load(R + f"game{g}_tracking.npz"); lab = pd.read_csv(R + f"game{g}_labels_reception.csv")["label"].values
    ball, pl, per, team = d["ball"], d["players"], d["period"], d["team"]
    if g == 1: print("ball x range", np.nanmin(ball[:, 0]), np.nanmax(ball[:, 0]), "team vals", set(team.tolist()))
    n = min(len(lab), len(ball)); lab = lab[:n]
    # attack direction per period: team with lower mean x attacks +x
    sgn = {}
    for p in (1, 2):
        m = per[:n] == p
        hx = np.nanmean(pl[:n][m][:, team == team[0], 0]); ax = np.nanmean(pl[:n][m][:, team != team[0], 0])
        sgn[(p, "H")] = 1 if hx < ax else -1; sgn[(p, "A")] = -sgn[(p, "H")]
    # spells
    spells = []; i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]: j += 1
        if lab[i] in ("H", "A") and per[i] > 0:
            s = sgn[(per[i], lab[i])]; x = ball[i:j, 0]; x = x[~np.isnan(x)]
            if len(x):
                xa = 0.5 + s * (x - 0.5) if np.nanmax(np.abs(ball[:, 0])) <= 1.5 else 0.5 + s * x / 105
                spells.append(dict(team=lab[i], dur=(j - i) / 25, x0=xa[0], xmax=xa.max(), prev=lab[i-1] if i else "X",
                                   box=bool(((xa > 1 - 16.5/105)).any())))
        i = j
    S = pd.DataFrame(spells)
    print(f"game {g}: spells {len(S)}, direct turnovers {(S.prev.isin(['H','A'])).sum()}")
    for dmin in (3, 5, 8, 10):
        L = S[S.dur >= dmin]
        print(f"  dur>={dmin:2}s n={len(L):3}  start own third {(L.x0<1/3).sum():3}  reach mid {(L.xmax>1/3).sum():3}  reach final third {(L.xmax>2/3).sum():3}  reach box {L.box.sum():3}  after direct turnover {(L.prev.isin(['H','A'])).sum():3}")
