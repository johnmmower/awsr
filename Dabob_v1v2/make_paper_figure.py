#!/usr/bin/env python3
"""
make_paper_figure.py -- single-column (3.5 in) IEEE figure: downwind
first-moment velocity minus the 8 m ADCP current versus wind speed, for the
two near-range grazing bands, with linear fits.

Usage:
  python3 make_paper_figure.py --merged compare_v2_det/merged.csv --out awsr_downwind_vs_wind.pdf
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True)
ap.add_argument("--out", default="awsr_downwind_vs_wind.pdf")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.merged)))
u = np.array([float(r["wind_speed"] or "nan") for r in rows])
bands = [("10-90deg", r"10–33$^\circ$ grazing", "C0", "o"),
         ("3-10deg", r"3–10$^\circ$ grazing", "C3", "s")]

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for key, label, col, mk in bands:
    y = np.array([float(r[f"{key}_downwind_minus_adcp"] or "nan") for r in rows])
    ok = np.isfinite(u) & np.isfinite(y)
    r = np.corrcoef(u[ok], y[ok])[0, 1]
    p = np.polyfit(u[ok], y[ok], 1)
    ax.plot(u[ok], y[ok], mk, ms=2.5, mfc="none", mec=col, mew=0.6, label=f"{label} ($r$ = {r:.2f})")
    xx = np.linspace(0, np.nanmax(u[ok]), 20)
    ax.plot(xx, np.polyval(p, xx), "-", color=col, lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"Wind speed, 30-min mean (m s$^{-1}$)")
ax.set_ylabel(r"Downwind velocity $-$ ADCP (m s$^{-1}$)")
ax.set_ylim(-0.6, 0.9)
ax.legend(fontsize=7, loc="upper left", frameon=False)
ax.grid(alpha=0.3, lw=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(args.out)
print(f"wrote {args.out}")
