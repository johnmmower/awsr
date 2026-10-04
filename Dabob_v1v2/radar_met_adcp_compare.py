#!/usr/bin/env python3
"""
radar_met_adcp_compare.py

Merge per-window radar diagnostics (dabob_doppler_calibrate.py outputs) with
the PISCES2 met buoy and ADCP (NetCDF), and test:
  1. Does radar signal (fraction of cells with SNR >= threshold) track wind
     speed?  -- the Bragg-roughness expectation
  2. Does the first-harmonic radial Doppler, projected on the downwind
     direction, track wind speed after removing the ADCP current?

Inputs
  --calib-dir     directory with inventory.csv (+ harmonics.csv) from the
                  calibration script
  --met, --adcp   PISCES2_MET_*.nc, PISCES2_ADCP_*.nc
  --products-dir  (recommended) directory of the *_combined.npz files the
                  calibration read: window UTC times are taken from the
                  antenna_*_timestamp_us arrays (epoch microseconds) instead
                  of the filename, which radar_archive_service writes in the
                  NUC's LOCAL time.

ASSUMPTIONS -- flagged:
  - Without --products-dir, filename time = UTC + --radar-utc-offset-hours
    (default -7, PDT). Check once with --products-dir: the script prints the
    offset it finds.
  - Met eastward/northward_wind_speed are the wind vector (blowing TOWARD);
    checked against wind_from_direction in this file (median diff 0.1 deg).
  - Wind is a vector average over the --wind-avg-min minutes BEFORE the
    window centre (short waves respond to wind with some lag; the length is
    a guess).
  - ADCP: nearest good bin to --adcp-depth-m (top good bin in this file is
    8 m; the radar sees the top centimetres, so this is only a proxy).
  - harmonics.csv azimuths are whatever the calibration used; run it with
    --heading-offset-deg 21 so A1 ~ north, B1 ~ east.
  - Met/ADCP QC: values with fill (-555 / -32768) or |v| > 10 m/s dropped;
    wind_speed_qc_agg in {1, 2} kept.
"""

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

R_EARTH = 6.371e6


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_nc(path, names):
    import netCDF4
    d = netCDF4.Dataset(path)
    out = {}
    for n in names:
        if n in d.variables:
            a = np.ma.filled(d[n][:].astype(float), np.nan)
            a[(a <= -555) | (a == -32768)] = np.nan
            out[n] = a
    out["_attrs"] = {k: getattr(d, k) for k in d.ncattrs()}
    return out


def window_utc_from_products(products_dir, pattern):
    """filename stem (first 14 digits) -> UTC epoch s (median of per-azimuth timestamps)."""
    out = {}
    for p in sorted(Path(products_dir).glob(pattern)):
        m = re.search(r"(\d{14})", p.name)
        if not m:
            continue
        try:
            with np.load(p) as d:
                ts = [d[k] for k in d.files if re.fullmatch(r"antenna_\d+_timestamp_us", k)]
            ts = np.concatenate([np.asarray(t, float).ravel() for t in ts]) if ts else np.array([])
            ts = ts[np.isfinite(ts) & (ts > 0)]
            if ts.size:
                out[m.group(1)] = float(np.median(ts)) / 1e6
        except Exception as e:
            print(f"  [warning] {p.name}: {e}")
    return out


def trailing_vector_mean(t, u, v, t_ref, minutes):
    """Vector-mean (u, v) over [t_ref - minutes, t_ref]."""
    uu = np.full(len(t_ref), np.nan)
    vv = np.full(len(t_ref), np.nan)
    ok = np.isfinite(u) & np.isfinite(v)
    t, u, v = t[ok], u[ok], v[ok]
    for i, tr in enumerate(t_ref):
        sel = (t <= tr) & (t >= tr - 60 * minutes)
        if sel.any():
            uu[i], vv[i] = u[sel].mean(), v[sel].mean()
    return uu, vv


def interp_gap(t, x, t_ref, max_gap_s):
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 2:
        return np.full(len(t_ref), np.nan)
    y = np.interp(t_ref, t, x, left=np.nan, right=np.nan)
    j = np.searchsorted(t, t_ref)
    j0, j1 = np.clip(j - 1, 0, len(t) - 1), np.clip(j, 0, len(t) - 1)
    far = (np.abs(t_ref - t[j0]) > max_gap_s) & (np.abs(t[j1] - t_ref) > max_gap_s)
    y[far] = np.nan
    return y


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    rs = np.corrcoef(rx, ry)[0, 1]
    return float(r), float(rs), int(ok.sum())


def binned(x, y, edges):
    cen, med, q1, q3, n = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = np.isfinite(x) & np.isfinite(y) & (x >= a) & (x < b)
        if s.sum() >= 3:
            cen.append(0.5 * (a + b))
            med.append(np.median(y[s]))
            q1.append(np.percentile(y[s], 25))
            q3.append(np.percentile(y[s], 75))
            n.append(int(s.sum()))
    return map(np.array, (cen, med, q1, q3, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--met", required=True)
    ap.add_argument("--adcp", required=True)
    ap.add_argument("--products-dir", help="combined npz dir, for UTC times from timestamp_us")
    ap.add_argument("--pattern", default="*_combined.npz")
    ap.add_argument("--radar-utc-offset-hours", type=float, default=-7.0,
                    help="Filename local time = UTC + this (used only without --products-dir)")
    ap.add_argument("--signal-col", default="frac_snr_ge_signal")
    ap.add_argument("--wind-avg-min", type=float, default=30.0)
    ap.add_argument("--adcp-depth-m", type=float, default=8.0)
    ap.add_argument("--band", default=None, help="harmonics band label to use (default: all bands)")
    ap.add_argument("--radar-lat", type=float, default=47.709487)
    ap.add_argument("--radar-lon", type=float, default=-122.823652)
    ap.add_argument("--height-m", type=float, default=66.0)
    ap.add_argument("--out-dir", default="compare_out")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- radar windows
    inv = read_csv(Path(args.calib_dir) / "inventory.csv")
    stems = [re.search(r"(\d{14})", r["file"]).group(1) for r in inv]
    utc_map = window_utc_from_products(args.products_dir, args.pattern) if args.products_dir else {}
    t_local_naive = np.array([datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
                              for s in stems])
    if utc_map:
        diffs = [(t_local_naive[i] - utc_map[s]) / 3600 for i, s in enumerate(stems) if s in utc_map]
        if diffs:
            print(f"filename minus timestamp_us: median {np.median(diffs):+.2f} h "
                  f"(so filename = UTC {np.median(diffs):+.1f} h)")
    t_rad = np.array([utc_map.get(s, t_local_naive[i] - 3600 * args.radar_utc_offset_hours)
                      for i, s in enumerate(stems)])
    sig = np.array([fnum(r.get(args.signal_col)) for r in inv])

    # ---------------- met
    met = load_nc(args.met, ["time", "wind_speed", "eastward_wind_speed", "northward_wind_speed",
                             "wind_from_direction", "wind_speed_qc_agg"])
    qc_ok = np.isin(np.nan_to_num(met.get("wind_speed_qc_agg", np.ones_like(met["time"])), nan=9), [1, 2])
    mu = np.where(qc_ok, met["eastward_wind_speed"], np.nan)
    mv = np.where(qc_ok, met["northward_wind_speed"], np.nan)
    ms = np.where(qc_ok, met["wind_speed"], np.nan)
    wu, wv = trailing_vector_mean(met["time"], mu, mv, t_rad, args.wind_avg_min)
    # scalar mean speed over the same interval (vector mean understates it when direction wanders)
    wspd = np.full(len(t_rad), np.nan)
    for i, tr in enumerate(t_rad):
        s = (met["time"] <= tr) & (met["time"] >= tr - 60 * args.wind_avg_min) & np.isfinite(ms)
        if s.any():
            wspd[i] = ms[s].mean()
    w_to = np.degrees(np.arctan2(wu, wv)) % 360        # direction wind blows toward, deg true

    # ---------------- adcp
    adcp = load_nc(args.adcp, ["time", "depth", "eastward_sea_water_velocity", "northward_sea_water_velocity"])
    ue, un = adcp["eastward_sea_water_velocity"], adcp["northward_sea_water_velocity"]
    ue[np.abs(ue) > 10] = np.nan
    un[np.abs(un) > 10] = np.nan
    good = np.isfinite(ue).mean(0) > 0.5
    k = int(np.argmin(np.where(good, np.abs(adcp["depth"] - args.adcp_depth_m), np.inf)))
    print(f"ADCP bin used: {adcp['depth'][k]:.1f} m")
    cu = interp_gap(adcp["time"], ue[:, k], t_rad, 1800)
    cv = interp_gap(adcp["time"], un[:, k], t_rad, 1800)

    # ---------------- geometry of the buoy
    a = met["_attrs"]
    blat, blon = float(a.get("buoy_latitude", np.nan)), float(a.get("buoy_longitude", np.nan))
    n_ = np.radians(blat - args.radar_lat) * R_EARTH
    e_ = np.radians(blon - args.radar_lon) * R_EARTH * np.cos(np.radians(args.radar_lat))
    g = np.hypot(n_, e_)
    print(f"buoy from radar: {g:.0f} m ground, bearing {np.degrees(np.arctan2(e_, n_)) % 360:.1f} deg true, "
          f"grazing {np.degrees(np.arctan2(args.height_m, g)):.2f} deg")

    # ---------------- harmonics (optional)
    hpath = Path(args.calib_dir) / "harmonics.csv"
    harm = read_csv(hpath) if hpath.exists() else []
    bands = sorted(set(r["band"] for r in harm)) if args.band is None else [args.band]
    idx = {s: i for i, s in enumerate(stems)}
    rows = []
    for i, s in enumerate(stems):
        rows.append(dict(file=inv[i]["file"], utc=datetime.fromtimestamp(t_rad[i], timezone.utc).isoformat(),
                         signal_frac=sig[i], wind_speed=wspd[i], wind_u=wu[i], wind_v=wv[i],
                         wind_to_deg=w_to[i], adcp_u=cu[i], adcp_v=cv[i]))
    hstats = {}
    for b in bands:
        A1 = np.full(len(stems), np.nan)
        B1 = np.full(len(stems), np.nan)
        A0 = np.full(len(stems), np.nan)
        for r in harm:
            if r["band"] != b or r.get("n_terms", "0") in ("0", ""):
                continue
            m = re.search(r"(\d{14})", r["file"])
            if m and m.group(1) in idx:
                j = idx[m.group(1)]
                A1[j], B1[j], A0[j] = fnum(r["A1"]), fnum(r["B1"]), fnum(r["A0"])
        th = np.radians(w_to)
        down = A1 * np.cos(th) + B1 * np.sin(th)             # radial Doppler, looking downwind
        cur_down = cu * np.sin(th) + cv * np.cos(th)
        resid = down - cur_down
        for i in range(len(stems)):
            rows[i][f"{b}_A0"] = A0[i]
            rows[i][f"{b}_downwind"] = down[i]
            rows[i][f"{b}_downwind_minus_adcp"] = resid[i]
        hstats[b] = dict(down=down, resid=resid, A0=A0)

    with open(out / "merged.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------------- stats + plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = []
    r, rs, n = corr(wspd, sig)
    lines.append(f"signal fraction vs wind speed: Pearson r={r:.2f}, Spearman={rs:.2f}, n={n}")
    cs = np.hypot(cu, cv)
    lines.append(f"ADCP {adcp['depth'][k]:.0f} m speed at radar times: median {np.nanmedian(cs):.3f} m/s, "
                 f"95th pct {np.nanpercentile(cs, 95):.3f} m/s")
    lines.append(f"wind speed at radar times: median {np.nanmedian(wspd):.2f} m/s, "
                 f"95th pct {np.nanpercentile(wspd, 95):.2f} m/s")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(wspd, sig, c=w_to, cmap="twilight", vmin=0, vmax=360, s=10)
    c, md, q1, q3, nn = binned(wspd, sig, np.arange(0, 12, 1.0))
    if len(c):
        ax.errorbar(c, md, yerr=[md - q1, q3 - md], fmt="ks-", ms=4, capsize=3, label="median, IQR (1 m/s bins)")
        ax.legend(fontsize=8)
    ax.set_xlabel(f"wind speed, {args.wind_avg_min:g}-min mean (m/s)")
    ax.set_ylabel(args.signal_col)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="wind toward (deg true)")
    ax.set_title(f"r = {r:.2f}, Spearman = {rs:.2f}, n = {n}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "signal_vs_wind.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    tm = [datetime.fromtimestamp(x, timezone.utc) for x in met["time"]]
    tr = [datetime.fromtimestamp(x, timezone.utc) for x in t_rad]
    axs[0].plot(tm, ms, lw=0.6, color="0.5", label="10-min")
    axs[0].plot(tr, wspd, ".", ms=3, label=f"at radar windows ({args.wind_avg_min:g}-min mean)")
    axs[0].set_ylabel("wind (m/s)")
    axs[0].legend(fontsize=8)
    axs[1].plot(tr, sig, ".", ms=3)
    axs[1].set_ylabel(args.signal_col)
    ta = [datetime.fromtimestamp(x, timezone.utc) for x in adcp["time"]]
    axs[2].plot(ta, ue[:, k], lw=0.6, label="east")
    axs[2].plot(ta, un[:, k], lw=0.6, label="north")
    axs[2].set_ylabel(f"ADCP {adcp['depth'][k]:.0f} m (m/s)")
    axs[2].legend(fontsize=8)
    for axx in axs:
        axx.grid(alpha=0.3)
    axs[0].set_xlim(min(tr), max(tr))
    fig.tight_layout()
    fig.savefig(out / "timeseries_wind_signal_adcp.png", dpi=150)
    plt.close(fig)

    if hstats:
        fig, axs = plt.subplots(1, len(hstats), figsize=(4.5 * len(hstats), 4), squeeze=False)
        for axx, (b, h) in zip(axs[0], hstats.items()):
            r1, rs1, n1 = corr(wspd, h["resid"])
            lines.append(f"[{b}] downwind radial Doppler minus ADCP vs wind speed: r={r1:.2f}, "
                         f"Spearman={rs1:.2f}, n={n1}; A0 median {np.nanmedian(h['A0']):+.3f} m/s")
            ok = np.isfinite(wspd) & np.isfinite(h["resid"])
            axx.scatter(wspd[ok], h["resid"][ok], s=10)
            if ok.sum() >= 5:
                p = np.polyfit(wspd[ok], h["resid"][ok], 1)
                xx = np.linspace(0, np.nanmax(wspd[ok]), 10)
                axx.plot(xx, np.polyval(p, xx), "r-", label=f"{p[0]:.3f} U + {p[1]:.3f}")
                axx.legend(fontsize=8)
                lines.append(f"    linear fit: {p[0]:.4f} * U + {p[1]:.4f} m/s")
            axx.axhline(0, color="k", lw=0.5)
            axx.set_title(f"{b}: r={r1:.2f}, n={n1}", fontsize=9)
            axx.set_xlabel("wind speed (m/s)")
            axx.set_ylabel("downwind radial Doppler − ADCP (m/s)")
            axx.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "downwind_doppler_vs_wind.png", dpi=180)
        plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"outputs in {out}/: merged.csv summary.txt signal_vs_wind.png timeseries_wind_signal_adcp.png"
          + (" downwind_doppler_vs_wind.png" if hstats else ""))


if __name__ == "__main__":
    main()
