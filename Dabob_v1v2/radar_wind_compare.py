#!/usr/bin/env python3
"""
radar_wind_compare.py -- radar wind estimates vs the PISCES2 met buoy.

Three estimates, each compared with the buoy's 30-min mean wind:

 (1) Doppler direction: azimuth of maximum receding first-harmonic velocity
     (dir1 from harmonics.csv) ~ direction the wind blows TOWARD.
     Needs --merged (from radar_met_adcp_compare.py) and --harmonics.
 (2) SNR-asymmetry direction: VV return is brighter looking upwind, so the
     azimuth of the fitted first harmonic of mean SNR across azimuth ~
     direction the wind blows FROM. Independent of the Doppler sign.
     Needs --products (combined npz files; the original products work) and --met.
 (3) Wind speed from the detection count (n_valid column of merged.csv), via
     a log-linear fit U = a + b*log(n), scored by leave-one-out CV.

Directions are compared only when buoy wind >= --min-wind (default 3 m/s):
at lower wind the ripple field is weak and direction is poorly defined.

ASSUMPTIONS -- flagged:
  - harmonics.csv directions are in true coordinates. For the Dabob runs made
    with heading +21 and Doppler sign -1, that holds: the 180-deg azimuth
    error and the Doppler sign error cancel in A1/B1.
  - (2) maps radar azimuth to true with --heading-offset-deg (201 for Dabob)
    and uses slant range --r-min..--r-max (default 400-1300 m, ~3-10 deg
    grazing). Azimuths are restricted to --sector true degrees (default
    180-360, the water side) and PA-off azimuths are dropped. With only a
    half circle, the fitted direction is less certain for winds blowing along
    the sector edge.
  - Window times for (2) come from antenna_*_timestamp_us (UTC).
"""

import argparse
import csv
import glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from radar_met_adcp_compare import load_nc, trailing_vector_mean


def circ_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def circ_stats(d):
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0)
    r = np.radians(d)
    mean = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return dict(n=int(d.size), mean=float(mean), median_abs=float(np.median(np.abs(d))),
                rms=float(np.sqrt(np.mean(d ** 2))), within30=float(np.mean(np.abs(d) <= 30)))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def snr_profile(path, args):
    """Mean SNR (dB) per azimuth over the range window, true azimuths, usable mask, UTC time."""
    with np.load(path) as d:
        snr = np.asarray(d[args.snr_key], float)
        az = np.asarray(d["bin_centers_deg"], float) if "bin_centers_deg" in d.files else np.arange(snr.shape[0]) + 0.5
        pa = [np.asarray(d[k], float) for k in d.files if k.endswith("_pa_enable")]
        ts = [np.asarray(d[k], float).ravel() for k in d.files if k.endswith("_timestamp_us")]
    r = np.arange(snr.shape[1]) * args.dr_m
    rsel = (r >= args.r_min) & (r <= args.r_max)
    prof = 10 * np.log10(np.nanmean(10 ** (snr[:, rsel] / 10), axis=1))
    ok = np.isfinite(prof)
    if pa:
        ok &= np.min(pa, axis=0) >= 1
    t = np.concatenate(ts) if ts else np.array([])
    t = t[np.isfinite(t) & (t > 0)]
    return prof, ok, az, (np.median(t) / 1e6 if t.size else np.nan)


def fit_direction(prof, ok, az, args, offset):
    true_az = (az + offset) % 360
    lo, hi = args.sector
    m = ok & (((true_az - lo) % 360) <= ((hi - lo) % 360)) & np.isfinite(prof)
    if m.sum() < 20:
        return np.nan, np.nan
    th = np.radians(true_az[m])
    X = np.column_stack([np.ones(m.sum()), np.cos(th), np.sin(th)])
    beta, *_ = np.linalg.lstsq(X, prof[m], rcond=None)
    return np.degrees(np.arctan2(beta[2], beta[1])) % 360, float(np.hypot(beta[1], beta[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", help="merged.csv from radar_met_adcp_compare.py")
    ap.add_argument("--harmonics", help="harmonics.csv from dabob_doppler_calibrate.py")
    ap.add_argument("--bands", nargs="+", default=["10-90deg", "3-10deg"])
    ap.add_argument("--products", help="glob of combined npz files for the SNR-asymmetry method")
    ap.add_argument("--met", help="PISCES2_MET nc (needed with --products)")
    ap.add_argument("--snr-key", default="combined_snr_db")
    ap.add_argument("--heading-offset-deg", type=float, default=201.0)
    ap.add_argument("--sector", type=float, nargs=2, default=[180.0, 360.0], metavar=("FROM", "TO"))
    ap.add_argument("--test-offsets", type=float, nargs="+", default=[201.0],
                    help="heading offsets to score for the SNR-asymmetry method")
    ap.add_argument("--keep-mean-profile", action="store_true",
                    help="don't subtract the record-median azimuth profile")
    ap.add_argument("--r-min", type=float, default=400.0)
    ap.add_argument("--r-max", type=float, default=1300.0)
    ap.add_argument("--dr-m", type=float, default=4.8)
    ap.add_argument("--wind-avg-min", type=float, default=30.0)
    ap.add_argument("--min-wind", type=float, default=3.0)
    ap.add_argument("--speed-col", default="signal_frac", help="merged.csv column used as the speed proxy")
    ap.add_argument("--out-dir", default="wind_compare")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = []
    panels = []

    # ---------------- (1) Doppler direction and (3) speed, from merged + harmonics
    if args.merged:
        mrows = list(csv.DictReader(open(args.merged)))
        by_file = {r["file"]: r for r in mrows}
        if args.harmonics:
            hrows = list(csv.DictReader(open(args.harmonics)))
            for band in args.bands:
                rd, bd, ws = [], [], []
                for h in hrows:
                    if h["band"] != band or h.get("n_terms", "0") in ("0", "") or h["file"] not in by_file:
                        continue
                    m = by_file[h["file"]]
                    rd.append(fnum(h["dir1_deg"]))
                    bd.append(fnum(m["wind_to_deg"]))
                    ws.append(fnum(m["wind_speed"]))
                rd, bd, ws = map(np.array, (rd, bd, ws))
                sel = ws >= args.min_wind
                st = circ_stats(circ_diff(rd[sel], bd[sel]))
                lines.append(f"(1) Doppler direction [{band}] vs buoy wind-toward, U >= {args.min_wind:g}: {st}")
                if st.get("n"):
                    st2 = circ_stats(circ_diff(rd[sel] - st["mean"], bd[sel]))
                    lines.append(f"    after removing the mean offset ({st['mean']:+.0f} deg): "
                                 f"median |diff| {st2['median_abs']:.0f} deg, within 30 deg {st2['within30']:.2f}")
                panels.append((f"Doppler dir [{band}]", bd, rd, ws, "buoy wind toward (deg)", "radar max-receding az (deg)"))

        # (3) speed from detection-count proxy, leave-one-out
        x = np.array([fnum(r.get(args.speed_col)) for r in mrows])
        u = np.array([fnum(r["wind_speed"]) for r in mrows])
        ok = np.isfinite(x) & np.isfinite(u) & (x > 0)
        if ok.sum() > 10:
            lx, uu = np.log(x[ok]), u[ok]
            pred = np.empty_like(uu)
            for i in range(len(uu)):
                m = np.ones(len(uu), bool)
                m[i] = False
                b, a = np.polyfit(lx[m], uu[m], 1)
                pred[i] = a + b * lx[i]
            err = pred - uu
            b, a = np.polyfit(lx, uu, 1)
            lines.append(f"(3) speed from log({args.speed_col}): U = {a:.2f} + {b:.2f} ln(n); leave-one-out "
                         f"RMSE {np.sqrt(np.mean(err**2)):.2f} m/s, bias {err.mean():+.2f} m/s, "
                         f"r {np.corrcoef(pred, uu)[0,1]:.2f}, n {len(uu)}")
            fig, axx = plt.subplots(figsize=(4.5, 4.2))
            axx.plot(uu, pred, ".", ms=4)
            lim = [0, max(uu.max(), pred.max()) * 1.05]
            axx.plot(lim, lim, "k-", lw=0.5)
            axx.set_xlabel("buoy wind speed (m/s)")
            axx.set_ylabel("radar wind speed, leave-one-out (m/s)")
            axx.set_title(f"RMSE {np.sqrt(np.mean(err**2)):.2f} m/s, r {np.corrcoef(pred, uu)[0,1]:.2f}", fontsize=9)
            axx.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "radar_wind_speed.png", dpi=170)
            plt.close(fig)

    # ---------------- (2) SNR-asymmetry direction from products
    if args.products:
        if not args.met:
            raise SystemExit("--products needs --met")
        met = load_nc(args.met, ["time", "eastward_wind_speed", "northward_wind_speed", "wind_speed"])
        files = sorted(glob.glob(args.products))
        profs = [snr_profile(f, args) for f in files]
        P = np.array([p[0] for p in profs])
        okm = np.array([p[1] for p in profs])
        az = profs[0][2]
        tt = np.array([p[3] for p in profs])
        if args.keep_mean_profile:
            A = P
        else:
            # remove the record-median profile: static structure (land, near-shore,
            # antenna pattern) otherwise dominates the azimuthal fit
            A = P - np.nanmedian(np.where(okm, P, np.nan), axis=0)[None, :]
        wu, wv = trailing_vector_mean(met["time"], met["eastward_wind_speed"], met["northward_wind_speed"],
                                      np.nan_to_num(tt, nan=-1e12), args.wind_avg_min)
        ws = np.full(len(tt), np.nan)
        for i, t in enumerate(tt):
            s = (met["time"] <= t) & (met["time"] >= t - 60 * args.wind_avg_min) & np.isfinite(met["wind_speed"])
            if s.any():
                ws[i] = met["wind_speed"][s].mean()
        buoy_from = (np.degrees(np.arctan2(wu, wv)) + 180) % 360
        sel = ws >= args.min_wind
        for off in args.test_offsets:
            res = [fit_direction(A[i], okm[i], az, args, off) for i in range(len(files))]
            fdo = np.array([r[0] for r in res])
            ampo = np.array([r[1] for r in res])
            st = circ_stats(circ_diff(fdo[sel], buoy_from[sel]))
            lines.append(f"(2) SNR-asymmetry direction (offset {off:g}, "
                         f"{'raw' if args.keep_mean_profile else 'mean profile removed'}) vs buoy wind-from, "
                         f"U >= {args.min_wind:g}, {len(files)} files: {st}")
            if off == args.heading_offset_deg:
                fd, amp = fdo, ampo
        if args.heading_offset_deg not in args.test_offsets:
            res = [fit_direction(A[i], okm[i], az, args, args.heading_offset_deg) for i in range(len(files))]
            fd = np.array([r[0] for r in res])
            amp = np.array([r[1] for r in res])
        lines.append(f"    asymmetry amplitude (dB): median {np.nanmedian(amp[sel]):.2f} for U >= {args.min_wind:g}, "
                     f"{np.nanmedian(amp[~sel & np.isfinite(ws)]):.2f} below")
        panels.append((f"SNR asymmetry dir (offset {args.heading_offset_deg:g})", buoy_from, fd, ws,
                       "buoy wind from (deg)", "radar brightest az (deg)"))
        with open(out / "snr_direction.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "utc", "radar_from_deg", "asym_db", "buoy_from_deg", "buoy_speed"])
            for f, a, b, c, d_, e in zip(files, tt, fd, amp, buoy_from, ws):
                w.writerow([Path(f).name, datetime.fromtimestamp(a, timezone.utc).isoformat() if np.isfinite(a) else "",
                            f"{b:.1f}", f"{c:.2f}", f"{d_:.1f}", f"{e:.2f}"])

    if panels:
        fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3), squeeze=False)
        for axx, (title, xb, yr, ws, xl, yl) in zip(axs[0], panels):
            sel = np.isfinite(xb) & np.isfinite(yr) & (ws >= args.min_wind)
            sc = axx.scatter(xb[sel], yr[sel], c=ws[sel], s=12, cmap="viridis")
            for k in (-360, 0, 360):
                axx.plot([0, 360], [k, 360 + k], "k-", lw=0.5)
            axx.set_xlim(0, 360)
            axx.set_ylim(0, 360)
            axx.set_xlabel(xl)
            axx.set_ylabel(yl)
            d = circ_diff(yr[sel], xb[sel])
            axx.set_title(f"{title}: median |diff| {np.median(np.abs(d)) if d.size else np.nan:.0f} deg, n={sel.sum()}",
                          fontsize=9)
            axx.grid(alpha=0.3)
            fig.colorbar(sc, ax=axx, label="buoy wind (m/s)")
        fig.tight_layout()
        fig.savefig(out / "radar_wind_direction.png", dpi=160)
        plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"outputs in {out}/")


if __name__ == "__main__":
    main()
