#!/usr/bin/env python3
"""
dabob_doppler_calibrate.py

Noise/SNR calibration, land masking, and azimuthal harmonic fits for the
Dabob X-band *_combined.npz products (output of radar_sync_and_process.py).

Implements steps 1-7 of the procedure we discussed. Files are streamed one
at a time, four passes over the input, so ~1000 files is fine on memory.

  pass 1  inventory + per-cell climatology: mean SNR, scan-to-scan
          variability  ->  smooth SNR background, rough land mask
  pass 2  per-scan land flags (catches tide-varying aliased land) and the
          neighbour-slope test for mixed land/water cells  ->  final mask
  pass 3  noise sigma vs SNR (range structure function, extrapolated to
          zero lag; scan-to-scan differences as an upper bound) and
          shrinkage gamma vs SNR (adjacent-cell regression against a
          high-SNR reference)  ->  SNR cutoff, c in gamma = s/(s+c)
  pass 4  per-scan, per-grazing-band weighted harmonic fits of horizontal
          radial velocity: v = A0 + A1 cos(phi) + B1 sin(phi)
                                  + A2 cos(2phi) + B2 sin(2phi)
          (phi clockwise from north; for a uniform current A1 = u_north,
          B1 = u_east)

No ADCP or wind is used here; harmonics.csv is the input for that step.

RUN --inspect FIRST. I have not seen radar_products_combiner.py, so array
key names, axis order, units, and the empty-bin convention are guessed
(candidate lists below) and should be confirmed.

ASSUMPTIONS -- flagging explicitly (all overridable on the command line):
  - Each file holds one "scan": 2-D (azimuth, range) combined fields, time
    taken from the %Y%m%d%H%M%S stamp in the filename (whatever clock the
    NUC writes). 3-D files with a time axis are supported if they carry a
    times array (epoch seconds).
  - moment1 is in Hz. --doppler-sign -1 (default) assumes positive Doppler
    = approaching; everything downstream uses positive = AWAY from radar.
    If this is wrong, A1/B1 flip sign (direction off by 180 deg).
  - Azimuth 0 = true north, clockwise, bin centres at az_start + i*daz
    unless an azimuth array is in the file. --heading-offset-deg adds a
    correction.
  - Range bins are slant range, r = range0 + i*4.8 m unless a range array
    is in the file.
  - Empty bins are non-finite, OR have snr == 0 and moment1 == 0 exactly
    (--no-zero-is-empty to disable).
  - If a per-bin antenna count exists, the noise curve is stratified by
    count (noise of an averaged moment depends on how many antennas went
    into it, which SNR alone doesn't capture). If not, one curve is fit to
    the mixture.
  - Adjacent range bins are assumed to have independent estimator noise.
    If the pulse is longer than 4.8 m they won't; raise --sf-lag0 /
    --shrink-lag to skip correlated lags. The scan-to-scan estimate is
    unaffected by this and is written alongside as a check.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

C_LIGHT = 299_792_458.0
R_EARTH = 6.371e6

KEY_CANDIDATES = {
    "snr": ["combined_snr_db", "snr_db_combined", "snr_db"],
    "m1": ["combined_moment1_hz", "moment1_hz_combined", "moment1_hz"],
    "count": ["combined_num_antennas", "combined_n_antennas", "combined_n_hits", "combined_count",
              "combined_n", "n_antennas", "n_hits", "hit_count"],
    "az": ["bin_centers_deg", "azimuth_deg", "azimuth_centers_deg", "az_deg", "az_centers_deg",
           "angle_deg", "azimuth_bins_deg"],
    "range": ["range_m", "range_centers_m", "range_bins_m", "ranges_m"],
    "time": ["times", "time", "timestamps", "time_s", "epoch_s"],
}


# ----------------------------------------------------------------------------
# file discovery / layout
# ----------------------------------------------------------------------------

def parse_time_from_name(name: str) -> Optional[datetime]:
    m = re.search(r"(\d{14})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def list_files(args):
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    files = [(p, parse_time_from_name(p.name)) for p in paths]
    files.sort(key=lambda x: (x[1] or datetime.min, x[0].name))
    if args.every > 1:
        files = files[::args.every]
    if args.limit_files:
        files = files[:args.limit_files]
    return files


def inspect_file(path):
    print(f"== {path}")
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            try:
                a = d[k]
            except Exception as e:  # object arrays etc.
                print(f"  {k:32s} <unreadable without pickle: {e}>")
                continue
            info = f"  {k:32s} shape={str(a.shape):18s} dtype={a.dtype}"
            if a.size and np.issubdtype(a.dtype, np.number):
                af = a.astype(float)
                fin = np.isfinite(af)
                if fin.any():
                    info += (f"  min={np.nanmin(af[fin]):.4g} max={np.nanmax(af[fin]):.4g}"
                             f"  finite={fin.mean()*100:.1f}%  zeros={np.mean(af == 0)*100:.1f}%")
                else:
                    info += "  (no finite values)"
            elif a.size == 1:
                info += f"  value={a.item()!r}"
            print(info)


def _pick(keys, override, cands, required, what, flag):
    if override:
        if override not in keys:
            sys.exit(f"--{flag} '{override}' not in file keys: {keys}")
        return override
    for c in cands:
        if c in keys:
            return c
    if required:
        sys.exit(f"Could not find the {what} array (tried {cands}). Keys are: {keys}. "
                 f"Run --inspect and pass --{flag}.")
    return None


@dataclass
class Layout:
    snr_key: str
    m1_key: str
    count_key: Optional[str]
    time_key: Optional[str]
    time_axis: Optional[int]
    transpose: bool
    n_az: int
    n_r: int
    az_deg: np.ndarray
    range_m: np.ndarray
    ncls: int
    pa_keys: list


def detect_layout(path, args) -> Layout:
    with np.load(path, allow_pickle=False) as d:
        keys = list(d.files)
        snr_k = _pick(keys, args.snr_key, KEY_CANDIDATES["snr"], True, "SNR", "snr-key")
        m1_k = _pick(keys, args.m1_key, KEY_CANDIDATES["m1"], True, "moment1", "m1-key")
        cnt_k = _pick(keys, args.count_key, KEY_CANDIDATES["count"], False, "count", "count-key")
        az_k = _pick(keys, args.az_key, KEY_CANDIDATES["az"], False, "azimuth", "az-key")
        rg_k = _pick(keys, args.range_key, KEY_CANDIDATES["range"], False, "range", "range-key")
        t_k = _pick(keys, args.time_key, KEY_CANDIDATES["time"], False, "time", "time-key")
        raw = d[snr_k]
        az = np.asarray(d[az_k], float).ravel() if az_k else None
        rg = np.asarray(d[rg_k], float).ravel() if rg_k else None
        tt = np.asarray(d[t_k]).ravel() if t_k else None
        pa_keys = sorted(k for k in keys if re.fullmatch(r"antenna_\d+_pa_enable", k))

    if cnt_k and args.no_count:
        cnt_k = None

    if raw.ndim == 2:
        time_axis = None
        sp = raw.shape
    elif raw.ndim == 3:
        if tt is None:
            sys.exit(f"'{snr_k}' is 3-D {raw.shape} but no time array found. If this is "
                     f"per-antenna (e.g. 4 x az x range), point --snr-key/--m1-key at the "
                     f"combined arrays; if it has a time axis, pass --time-key.")
        if args.time_axis is not None:
            time_axis = args.time_axis
        else:
            cands = [ax for ax in range(3) if raw.shape[ax] == len(tt)]
            if not cands:
                sys.exit(f"No axis of {raw.shape} matches len(times)={len(tt)}; pass --time-axis.")
            time_axis = cands[-1]
        sp = tuple(s for i, s in enumerate(raw.shape) if i != time_axis)
    else:
        sys.exit(f"'{snr_k}' has unsupported ndim {raw.ndim}")

    def n_from_coord(c):
        return None if c is None else len(c)

    if args.axis_order:
        transpose = args.axis_order == "range,az"
    elif az is not None and n_from_coord(az) in (sp[1], sp[1] + 1) and n_from_coord(az) not in (sp[0], sp[0] + 1):
        transpose = True
    elif rg is not None and n_from_coord(rg) in (sp[0], sp[0] + 1) and n_from_coord(rg) not in (sp[1], sp[1] + 1):
        transpose = True
    else:
        transpose = False
    n_az, n_r = (sp[1], sp[0]) if transpose else sp

    def centres(c, n, start, step, name):
        if c is not None:
            if len(c) == n:
                return c
            if len(c) == n + 1:
                return 0.5 * (c[:-1] + c[1:])
            print(f"  [warning] {name} array length {len(c)} doesn't match {n}; using start/step")
        return start + step * np.arange(n)

    az_deg = centres(az, n_az, args.az_start_deg, 360.0 / n_az, "azimuth")
    range_m = centres(rg, n_r, args.range0_m, args.dr_m, "range")
    ncls = args.max_count_class if cnt_k else 1

    if args.no_pa_mask:
        pa_keys = []
    lay = Layout(snr_k, m1_k, cnt_k, t_k, time_axis, transpose, n_az, n_r,
                 az_deg, range_m, ncls, pa_keys)
    print(f"Layout: snr='{snr_k}' m1='{m1_k}' count={cnt_k!r} time={t_k!r} "
          f"time_axis={time_axis} transpose={transpose}")
    print(f"  n_az={n_az} ({az_deg[0]:.2f}..{az_deg[-1]:.2f} deg, from {'file' if az_k else 'args'}), "
          f"n_range={n_r} ({range_m[0]:.1f}..{range_m[-1]:.1f} m, from {'file' if rg_k else 'args'})")
    if cnt_k is None:
        print("  no per-bin count array: noise curve fit to the mixture of antenna counts")
    if pa_keys:
        print(f"  masking azimuths where any of {len(pa_keys)} antennas has pa_enable < {args.pa_min} "
              f"(the combined arrays may include noise-only antennas there)")
    return lay


# ----------------------------------------------------------------------------
# geometry, bins, filters
# ----------------------------------------------------------------------------

@dataclass
class Geometry:
    psi_deg: np.ndarray
    cospsi: np.ndarray
    ground_m: np.ndarray
    ok: np.ndarray
    band: np.ndarray
    band_labels: list
    water2d: Optional[np.ndarray] = None


def make_geometry(range_m, args) -> Geometry:
    r = np.asarray(range_m, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = args.height_m / r - r / (2.0 * args.k_e * R_EARTH)
    ok = (r > args.height_m) & (s > 0) & (s < 1)
    ok &= (r >= args.min_range_m) & (r <= args.max_range_m)
    psi = np.full_like(r, np.nan)
    psi[ok] = np.degrees(np.arcsin(s[ok]))
    cospsi = np.cos(np.radians(np.nan_to_num(psi)))
    ground = np.sqrt(np.maximum(r ** 2 - args.height_m ** 2, 0.0))
    edges = sorted(args.band_edges_deg, reverse=True)
    band = np.full(r.shape, -1, int)
    labels = []
    for i in range(len(edges) - 1):
        sel = ok & (psi <= edges[i]) & (psi > edges[i + 1])
        band[sel] = i
        labels.append(f"{edges[i+1]:g}-{edges[i]:g}deg")
    ok &= band >= 0
    return Geometry(psi, cospsi, ground, ok, band, labels)


def true_az_deg(az_deg, args):
    """Radar azimuth -> true azimuth (deg, clockwise from north).
    normal:   true = radar + heading_offset
    mirrored: true = heading_offset - radar   (radar azimuth runs counter-clockwise)"""
    az = np.asarray(az_deg, float)
    return (args.heading_offset_deg - az) % 360 if args.azimuth_mirror else (az + args.heading_offset_deg) % 360


def cell_latlon(lay, geom, args):
    """Cell-centre lat/lon on a local tangent plane (fine at < ~10 km)."""
    th = np.radians(true_az_deg(lay.az_deg, args))[:, None]
    g = geom.ground_m[None, :]
    east, north = g * np.sin(th), g * np.cos(th)
    lat = args.radar_lat + np.degrees(north / R_EARTH)
    lon = args.radar_lon + np.degrees(east / (R_EARTH * np.cos(np.radians(args.radar_lat))))
    return lat, lon


def water_mask_from_geojson(path, lat, lon):
    """True where the cell centre lies inside any water polygon (holes = islands)."""
    from matplotlib.path import Path as MPath
    with open(path) as fh:
        gj = json.load(fh)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    polys = []
    for f in feats:
        geo = f.get("geometry", f)
        if geo["type"] == "Polygon":
            polys.append(geo["coordinates"])
        elif geo["type"] == "MultiPolygon":
            polys.extend(geo["coordinates"])
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(len(pts), bool)
    for rings in polys:
        cur = MPath(np.asarray(rings[0])[:, :2]).contains_points(pts)
        for hole in rings[1:]:
            cur &= ~MPath(np.asarray(hole)[:, :2]).contains_points(pts)
        inside |= cur
    print(f"  water polygon(s): {len(polys)}; {inside.mean()*100:.1f}% of cells are water")
    return inside.reshape(lat.shape)


class SnrBins:
    def __init__(self, lo, hi, step):
        self.edges = np.arange(lo, hi + 0.5 * step, step)
        self.lo, self.step = lo, step
        self.n = len(self.edges) - 1
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])

    def index(self, snr):
        x = np.where(np.isfinite(snr), snr, self.lo - 1e6)
        k = np.floor((x - self.lo) / self.step).astype(np.int64)
        ok = (k >= 0) & (k < self.n)
        return np.clip(k, 0, self.n - 1), ok


def wrap_filter(func, a, size, **kw):
    """Apply an ndimage filter with azimuth (axis 0) wrapping, range edge-clamped."""
    pa = size[0] // 2
    if pa == 0:
        return func(a, size=size, mode="nearest", **kw)
    ap = np.pad(a, ((pa, pa), (0, 0)), mode="wrap")
    return func(ap, size=size, mode="nearest", **kw)[pa:-pa]


def dilate(mask, k):
    if k <= 0:
        return mask.copy()
    return wrap_filter(ndimage.maximum_filter, mask.astype(np.uint8), (2 * k + 1, 2 * k + 1)) > 0


def nbr_mean(v, m, size):
    """Mean of valid neighbours (excluding the centre cell) and their count."""
    npts = size[0] * size[1]
    vm = np.where(m, v, 0.0)
    mf = m.astype(float)
    sv = wrap_filter(ndimage.uniform_filter, vm, size) * npts - vm
    sm = wrap_filter(ndimage.uniform_filter, mf, size) * npts - mf
    sm = np.rint(sm)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(sm > 0, sv / np.maximum(sm, 1), np.nan)
    return mean, sm


# ----------------------------------------------------------------------------
# scan iterator
# ----------------------------------------------------------------------------

@dataclass
class Scan:
    t: Optional[datetime]
    snr: np.ndarray     # dB, NaN where invalid
    v: np.ndarray       # LOS m/s, positive away, 0 where invalid
    cls: np.ndarray     # count class index (0 if no count)
    valid: np.ndarray
    src: str


def iter_scans(files, lay: Layout, geom: Geometry, args, failures=None):
    lam2 = C_LIGHT / args.carrier_hz / 2.0
    for path, ftime in files:
        try:
            with np.load(path, allow_pickle=False) as d:
                snr = np.asarray(d[lay.snr_key], float)
                f = np.asarray(d[lay.m1_key], float)
                cnt = np.asarray(d[lay.count_key], float) if lay.count_key else None
                tt = np.asarray(d[lay.time_key], float).ravel() if lay.time_axis is not None else None
                pa_min = None
                if lay.pa_keys:
                    pa_min = np.min([np.asarray(d[k], float).ravel() for k in lay.pa_keys], axis=0)
        except Exception as e:
            if failures is not None:
                failures.append((str(path), f"load failed: {e}"))
            continue
        if lay.time_axis is None:
            slices = [(ftime, snr, f, cnt)]
        else:
            ax = lay.time_axis
            slices = [(datetime.fromtimestamp(tt[i], timezone.utc).replace(tzinfo=None),
                       np.take(snr, i, ax), np.take(f, i, ax),
                       np.take(cnt, i, ax) if cnt is not None else None)
                      for i in range(len(tt))]
        for t, s2, f2, c2 in slices:
            if lay.transpose:
                s2, f2 = s2.T, f2.T
                c2 = c2.T if c2 is not None else None
            if s2.shape != (lay.n_az, lay.n_r) or f2.shape != s2.shape:
                if failures is not None:
                    failures.append((str(path), f"shape {s2.shape} != {(lay.n_az, lay.n_r)}"))
                continue
            valid = np.isfinite(s2) & np.isfinite(f2) & geom.ok[None, :]
            if geom.water2d is not None:
                valid &= geom.water2d
            if pa_min is not None and pa_min.shape == (lay.n_az,) and lay.time_axis is None:
                valid &= (pa_min >= args.pa_min)[:, None]
            if args.zero_is_empty:
                valid &= ~((s2 == 0) & (f2 == 0))
            valid &= np.nan_to_num(s2, nan=-np.inf) > args.empty_snr_db
            if c2 is not None:
                c2 = np.nan_to_num(c2)
                if c2.ndim == 1:            # per-azimuth count -> broadcast along range
                    c2 = np.broadcast_to(c2[:, None] if c2.shape[0] == lay.n_az else c2[None, :], s2.shape)
                valid &= c2 >= args.min_count
                cls = (np.clip(c2, 1, lay.ncls) - 1).astype(np.int64)
            else:
                cls = np.zeros(s2.shape, np.int64)
            v = np.where(valid, args.doppler_sign * lam2 * f2, 0.0)
            s2 = np.where(valid, s2, np.nan)
            yield Scan(t, s2, v, cls, valid, str(path))


def dt_seconds(a, b):
    if a is None or b is None:
        return None
    return (a - b).total_seconds()


# ----------------------------------------------------------------------------
# pass 1: inventory + climatology
# ----------------------------------------------------------------------------

def pass1(files, lay, geom, args, failures):
    shp = (lay.n_az, lay.n_r)
    n = np.zeros(shp)
    s_snr = np.zeros(shp)
    s_v = np.zeros(shp)
    s_v2 = np.zeros(shp)
    n_sig = np.zeros(shp)
    n_dt = np.zeros(shp)
    s_dt2 = np.zeros(shp)
    hist_edges = np.arange(-60, 120.25, 0.5)
    snr_hist = np.zeros(len(hist_edges) - 1)
    inv = []
    prev = None
    for sc in iter_scans(files, lay, geom, args, failures):
        val = sc.valid
        n += val
        s_snr += np.where(val, sc.snr, 0.0)
        s_v += sc.v
        s_v2 += sc.v * sc.v
        with np.errstate(invalid="ignore"):
            above = val & (sc.snr >= args.signal_snr_db)
        n_sig += above
        snr_hist += np.histogram(sc.snr[val], hist_edges)[0]
        dt = dt_seconds(sc.t, prev.t) if prev is not None else None
        if dt is not None and 0 < dt <= args.max_dt_s:
            m = val & prev.valid
            d = sc.v - prev.v
            n_dt += m
            s_dt2 += np.where(m, d * d, 0.0)
        inv.append(dict(time=sc.t.isoformat() if sc.t else "", file=Path(sc.src).name,
                        n_valid=int(val.sum()),
                        frac_snr_ge_signal=round(float(above.sum() / max(val.sum(), 1)), 4),
                        az_coverage_deg=round(float(val.any(axis=1).mean() * 360), 1),
                        dt_prev_s="" if dt is None else round(dt, 1)))
        prev = sc
    return dict(n=n, s_snr=s_snr, s_v=s_v, s_v2=s_v2, n_sig=n_sig, n_dt=n_dt, s_dt2=s_dt2,
                snr_hist=snr_hist, hist_edges=hist_edges), inv


def hist_percentile(h, edges, q):
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    i = np.searchsorted(c, q / 100.0 * c[-1])
    return 0.5 * (edges[i] + edges[i + 1])


def derive_background(p1, nscan, args):
    n = p1["n"]
    min_n = max(args.min_obs, args.min_obs_frac * nscan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_snr = np.where(n >= min_n, p1["s_snr"] / np.maximum(n, 1), np.nan)
        tstd = np.where(p1["n_dt"] >= min_n, np.sqrt(p1["s_dt2"] / np.maximum(p1["n_dt"], 1) / 2), np.nan)
        nn = np.maximum(n, 1)
        vmean = np.where(n >= min_n, p1["s_v"] / nn, np.nan)
        vstd = np.where(n >= min_n, np.sqrt(np.maximum(p1["s_v2"] / nn - (p1["s_v"] / nn) ** 2, 0)), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col = np.nanmedian(mean_snr, axis=0)
        glob = np.nanmedian(mean_snr) if np.isfinite(mean_snr).any() else 0.0
    filled = np.where(np.isfinite(mean_snr), mean_snr, col[None, :])
    filled = np.where(np.isfinite(filled), filled, glob)
    bg = wrap_filter(ndimage.median_filter, filled, tuple(args.bg_window))
    anom = mean_snr - bg
    with np.errstate(invalid="ignore"):
        # isolated bright + steady cells (aliased land), OR cells whose velocity
        # barely moves over the whole record (extended real shoreline, where a
        # local-median background is itself land and the anomaly test fails)
        rough = ((anom > args.land_anom_db) & (tstd < args.land_tstd_mps)) | (vstd < args.land_vstd_mps)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig_frac = np.where(n > 0, p1["n_sig"] / np.maximum(n, 1), np.nan)
    return dict(sig_frac=sig_frac, mean_snr=mean_snr, tstd=tstd, vmean=vmean, vstd=vstd, bg=bg, anom=anom, rough=rough, min_n=min_n)


# ----------------------------------------------------------------------------
# pass 2: per-scan land flags + mixed-cell neighbour slope
# ----------------------------------------------------------------------------

def pass2(files, lay, geom, args, clim):
    shp = (lay.n_az, lay.n_r)
    M1 = dilate(clim["rough"], args.dilate)
    n2 = np.zeros(shp)
    land_ct = np.zeros(shp)
    sxy = np.zeros(shp)
    syy = np.zeros(shp)
    svv = np.zeros(shp)
    nnb = np.zeros(shp)
    for sc in iter_scans(files, lay, geom, args):
        val = sc.valid
        a = sc.snr - clim["bg"]
        wm = val & ~M1
        if wm.any():
            a = a - np.nanmedian(a[wm])   # remove scan-wide SNR offset (wind/sea state)
        n2 += val
        with np.errstate(invalid="ignore"):
            land_ct += val & (a > args.land_anom_db) & (np.abs(sc.v) < args.land_vmax_mps)
        nb, cnt = nbr_mean(sc.v, wm, tuple(args.nbr_window))
        use = wm & (cnt >= 3)
        sxy += np.where(use, sc.v * nb, 0.0)
        syy += np.where(use, nb * nb, 0.0)
        svv += np.where(use, sc.v * sc.v, 0.0)
        nnb += use
    with np.errstate(invalid="ignore", divide="ignore"):
        land_frac = np.where(n2 > 0, land_ct / np.maximum(n2, 1), np.nan)
        slope = np.where(syy > 0, sxy / syy, np.nan)
        land2 = (n2 >= clim["min_n"]) & (land_frac > args.land_frac)
    # compare each cell's slope to the local median slope, so ordinary
    # low-SNR noise shrinkage (which neighbours share) isn't flagged
    fill = np.where(np.isfinite(slope), slope, 1.0)
    slope_loc = wrap_filter(ndimage.median_filter, fill, tuple(args.slope_window))
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_rel = slope / np.where(slope_loc > 0.05, slope_loc, np.nan)
        # standard error of the slope; require the low slope to be significant
        resid = np.maximum(svv - slope * sxy, 0.0) / np.maximum(nnb - 1, 1)
        slope_se = np.sqrt(resid / syy)
        slope_rel_se = slope_se / np.abs(slope_loc)
        mixed = ((nnb >= clim["min_n"]) & (slope_rel + 2 * slope_rel_se < args.mixed_slope)
                 & (clim["anom"] > args.mixed_anom_db))
    final = dilate(clim["rough"] | land2 | mixed, args.dilate) | ~geom.ok[None, :]
    if geom.water2d is not None:
        final |= ~geom.water2d
    return dict(land_frac=land_frac, slope=slope, slope_rel=slope_rel, land2=land2, mixed=mixed, final=final)


# ----------------------------------------------------------------------------
# pass 3: noise curve + shrinkage
# ----------------------------------------------------------------------------

def pass3(files, lay, geom, args, final, sb: SnrBins):
    lags = np.arange(args.sf_lag0, args.sf_lag0 + args.sf_nlags)
    ndv = int(round(args.dv_max / args.dv_res))
    nidx = lay.ncls * sb.n * ndv
    sf_hist = np.zeros((len(lags), nidx), np.int64)
    t_hist = np.zeros(nidx, np.int64)
    sxy = np.zeros(sb.n * sb.n)     # [test bin, ref bin]
    sxx = np.zeros(sb.n * sb.n)
    nsh = np.zeros(sb.n * sb.n)

    def accum(hist, s1, s2, v1, v2, c):
        k, ok = sb.index(0.5 * (s1 + s2))
        j = np.minimum((np.abs(v1 - v2) / args.dv_res).astype(np.int64), ndv - 1)
        idx = (c[ok] * sb.n + k[ok]) * ndv + j[ok]
        hist += np.bincount(idx, minlength=nidx)

    prev = None
    for sc in iter_scans(files, lay, geom, args):
        m = sc.valid & ~final
        for li, lag in enumerate(lags):
            mm = m[:, :-lag] & m[:, lag:] & (sc.cls[:, :-lag] == sc.cls[:, lag:])
            if mm.any():
                accum(sf_hist[li], sc.snr[:, :-lag][mm], sc.snr[:, lag:][mm],
                      sc.v[:, :-lag][mm], sc.v[:, lag:][mm], sc.cls[:, :-lag][mm])
        dt = dt_seconds(sc.t, prev[0]) if prev is not None else None
        if dt is not None and 0 < dt <= args.max_dt_s:
            _, pv, pm, ps, pc = prev
            mm = m & pm & (sc.cls == pc)
            if mm.any():
                accum(t_hist, sc.snr[mm], ps[mm], sc.v[mm], pv[mm], sc.cls[mm])
        lag = args.shrink_lag
        mm = m[:, :-lag] & m[:, lag:]
        pairs = ((sc.snr[:, :-lag], sc.v[:, :-lag], sc.snr[:, lag:], sc.v[:, lag:]),
                 (sc.snr[:, lag:], sc.v[:, lag:], sc.snr[:, :-lag], sc.v[:, :-lag]))
        for st, vt, sr, vr in pairs:
            sel = mm
            if not sel.any():
                continue
            k, ok = sb.index(st[sel])
            j, okj = sb.index(sr[sel])
            ok &= okj
            kk = k[ok] * sb.n + j[ok]
            vts, vrs = vt[sel][ok], vr[sel][ok]
            nn = sb.n * sb.n
            sxy += np.bincount(kk, weights=vts * vrs, minlength=nn)
            sxx += np.bincount(kk, weights=vrs * vrs, minlength=nn)
            nsh += np.bincount(kk, minlength=nn)
        prev = (sc.t, sc.v, m, sc.snr, sc.cls)
    return dict(lags=lags, ndv=ndv, sf_hist=sf_hist, t_hist=t_hist, sxy=sxy, sxx=sxx, nsh=nsh)


def hist_mad_sigma(h, dv_res):
    """Robust sigma of a difference from a histogram of |dv| (last bin = overflow)."""
    tot = h.sum(-1)
    cum = np.cumsum(h, -1)
    idx = np.argmax(cum >= (tot[..., None] / 2.0), axis=-1)
    med = (idx + 0.5) * dv_res
    bad = (tot == 0) | (idx == h.shape[-1] - 1)
    sig = np.where(bad, np.nan, 1.4826 * med)
    return sig, tot


def derive_noise(p3, lay, sb, args):
    L = len(p3["lags"])
    ndv = p3["ndv"]
    h = p3["sf_hist"].reshape(L, lay.ncls, sb.n, ndv)
    sig_d, tot = hist_mad_sigma(h, args.dv_res)       # (L, ncls, nsb)
    D = sig_d ** 2 / 2.0                              # structure function
    sigma_sf = np.full((lay.ncls, sb.n), np.nan)
    npairs_sf = tot.min(axis=0)
    for c in range(lay.ncls):
        for k in range(sb.n):
            y = D[:, c, k]
            if np.all(np.isfinite(y)) and npairs_sf[c, k] >= args.min_pairs:
                if L >= 2:
                    slope, icpt = np.polyfit(p3["lags"].astype(float), y, 1)
                else:
                    icpt = y[0]
                sigma_sf[c, k] = np.sqrt(max(icpt, 0.0))
    ht = p3["t_hist"].reshape(lay.ncls, sb.n, ndv)
    sig_t, npairs_t = hist_mad_sigma(ht, args.dv_res)
    sigma_t = np.where(npairs_t >= args.min_pairs, sig_t / np.sqrt(2.0), np.nan)

    # Shrinkage, chained down from high SNR. For a test cell in bin k next to a
    # cell in bin j:  v_k = g_k s + n_k,  v_j = g_j s + n_j, so
    #   g_k = g_j * E[v_k v_j] / (E[v_j^2] - sigma_j^2)
    # (subtracting sigma_j^2 undoes regression attenuation). Bins at or above
    # --ref-snr-db are anchored at g = 1; each lower bin is estimated from all
    # higher bins with known g and sigma, working downward.
    Sxy = p3["sxy"].reshape(sb.n, sb.n)
    Sxx = p3["sxx"].reshape(sb.n, sb.n)
    Nsh = p3["nsh"].reshape(sb.n, sb.n)
    if lay.ncls == 1:
        sig_ref = sigma_sf[0]
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sig_ref = np.nanmedian(sigma_sf, axis=0)
    ref_snr = args.ref_snr_db
    if ref_snr is None:
        # lowest SNR above which the measured noise stays below --ref-sigma-mps
        ref_snr = np.nan
        good_r = np.isfinite(sig_ref)
        for k in range(sb.n):
            if good_r[k] and np.all(sig_ref[k:][good_r[k:]] <= args.ref_sigma_mps):
                ref_snr = float(sb.edges[k])
                break
    gamma = np.full(sb.n, np.nan)
    gamma_raw = np.full(sb.n, np.nan)
    nsh = np.zeros(sb.n)
    anchor = (sb.edges[:-1] >= ref_snr) & (Nsh.sum(1) > 0)
    gamma[anchor] = 1.0
    gamma_raw[anchor] = 1.0
    for k in range(sb.n - 1, -1, -1):
        if anchor[k]:
            continue
        j = np.arange(k + 1, sb.n)
        okj = np.isfinite(gamma[j]) & np.isfinite(sig_ref[j]) & (Sxx[k, j] > 0)
        # only use reference bins whose own noise is a small part of their
        # variance; otherwise the subtraction is a small difference of large
        # numbers and the estimate is unstable
        with np.errstate(invalid="ignore", divide="ignore"):
            okj &= Nsh[k, j] * sig_ref[j] ** 2 <= args.max_ref_noise_frac * Sxx[k, j]
        j = j[okj]
        if j.size == 0:
            continue
        n_k = Nsh[k, j].sum()
        num = (gamma[j] * Sxy[k, j]).sum()
        den_raw = Sxx[k, j].sum()
        den = den_raw - (Nsh[k, j] * sig_ref[j] ** 2).sum()
        nsh[k] = n_k
        if n_k >= args.min_pairs and den > 0:
            gamma[k] = num / den
            gamma_raw[k] = num / den_raw
    s_lin = 10 ** (sb.centres / 10.0)
    good = np.isfinite(gamma) & ~anchor
    c_fit, c_rms = np.nan, np.nan
    if good.sum() >= 3:
        grid = np.logspace(-3, 3, 1201)
        w = nsh[good]
        err = [np.sum(w * (gamma[good] - s_lin[good] / (s_lin[good] + c)) ** 2) / w.sum() for c in grid]
        c_fit = float(grid[int(np.argmin(err))])
        c_rms = float(np.sqrt(min(err)))

    cutoff = np.full(lay.ncls, np.nan)
    if args.snr_cutoff is not None:
        cutoff[:] = args.snr_cutoff
    else:
        for c in range(lay.ncls):
            sig = sigma_sf[c]
            good_c = np.isfinite(sig)
            for k in range(sb.n):
                if not good_c[k]:
                    continue
                above = sig[k:][good_c[k:]]
                if np.all(above <= args.sigma_target_mps):
                    cutoff[c] = sb.edges[k]
                    break
    gamma_snr = np.nan
    if np.isfinite(c_fit):
        # SNR above which fitted shrinkage bias is < (1 - gamma_min)
        gamma_snr = float(10 * np.log10(c_fit * args.gamma_min / (1 - args.gamma_min)))
    return dict(sigma_sf=sigma_sf, npairs_sf=npairs_sf, D=D, sigma_t=sigma_t,
                npairs_t=npairs_t, gamma=gamma, gamma_raw=gamma_raw, nsh=nsh, ref_snr=ref_snr,
                gamma_snr=gamma_snr, c_fit=c_fit,
                c_rms=c_rms, cutoff=cutoff)


# ----------------------------------------------------------------------------
# pass 4: harmonic fits
# ----------------------------------------------------------------------------

def sigma_lookup(noise, sb, snr, cls, ncls, default):
    out = np.full(snr.shape, default)
    x_all = sb.centres
    s = np.nan_to_num(snr, nan=sb.lo)
    for c in range(ncls):
        sig = noise["sigma_sf"][c]
        good = np.isfinite(sig) & (sig > 0)
        sel = cls == c
        if good.any() and sel.any():
            out[sel] = np.interp(s[sel], x_all[good], sig[good])
    return out


def circ_max_gap(has):
    if has.all():
        return 0
    if not has.any():
        return len(has)
    x = np.concatenate([~has, ~has]).astype(int)
    best = run = 0
    for val in x:
        run = run + 1 if val else 0
        best = max(best, run)
    return min(best, len(has))


def harmonic_fit(phi, v, w, nterms):
    cols = [np.ones_like(phi)]
    if nterms >= 3:
        cols += [np.cos(phi), np.sin(phi)]
    if nterms >= 5:
        cols += [np.cos(2 * phi), np.sin(2 * phi)]
    X = np.stack(cols, 1)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = v * sw
    norms = np.linalg.norm(Xw, axis=0)
    cond = float(np.linalg.cond(Xw / norms))
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    r = yw - Xw @ beta
    dof = max(len(v) - X.shape[1], 1)
    chi2r = float(r @ r / dof)
    cov = np.linalg.inv(Xw.T @ Xw) * chi2r
    return beta, np.sqrt(np.diag(cov)), cond, chi2r


def pass4(files, lay, geom, args, final, noise, sb):
    phi_az = np.radians(true_az_deg(lay.az_deg, args))
    daz = 360.0 / lay.n_az
    names = ["A0", "A1", "B1", "A2", "B2"]
    rows = []
    g_ok = np.isfinite(noise["gamma"])
    bc = args.bias_correct and g_ok.sum() >= 2
    if args.bias_correct and not bc:
        print("  [warning] --bias-correct requested but no measured gamma curve; not applied")
    if bc:
        g_x, g_y = sb.centres[g_ok], np.clip(noise["gamma"][g_ok], 0.05, 1.0)
        g_min_snr = sb.edges[:-1][g_ok].min()
        print(f"  bias correction from measured gamma table; cells below {g_min_snr:.1f} dB dropped")
    for sc in iter_scans(files, lay, geom, args):
        cut = noise["cutoff"][sc.cls] if args.fit_snr_min is None else np.full(sc.cls.shape, args.fit_snr_min)
        with np.errstate(invalid="ignore"):
            m = sc.valid & ~final & (sc.snr >= cut)
        sig = sigma_lookup(noise, sb, sc.snr, sc.cls, lay.ncls, args.sigma_default_mps)
        v = sc.v.copy()
        if bc:
            # divide by measured shrinkage (interpolated, not the c model), and
            # don't extrapolate below the lowest SNR where gamma was measured
            with np.errstate(invalid="ignore"):
                m &= sc.snr >= g_min_snr
            fac = 1.0 / np.interp(np.nan_to_num(sc.snr, nan=g_min_snr), g_x, g_y)
            v *= fac
            sig = sig * fac
        nb, cnt = nbr_mean(v, m, tuple(args.despike_window))
        with np.errstate(invalid="ignore"):
            spike = m & (cnt >= 3) & (np.abs(v - nb) > args.despike_k * np.maximum(sig, args.sigma_floor_mps))
        m &= ~spike
        vh = v / geom.cospsi[None, :]
        sig_h = sig / geom.cospsi[None, :]
        w_all = 1.0 / (sig_h ** 2 + args.sigma_geo_mps ** 2)
        for b, label in enumerate(geom.band_labels):
            mb = m & (geom.band[None, :] == b)
            row = dict(time=sc.t.isoformat() if sc.t else "", file=Path(sc.src).name,
                       band=label, n_cells=int(mb.sum()), n_spikes=int((spike & (geom.band[None, :] == b)).sum()))
            for nm in names:
                row[nm] = np.nan
                row["s_" + nm] = np.nan
            row.update(n_terms=0, az_coverage_deg=np.nan, max_az_gap_deg=np.nan,
                       cond=np.nan, chi2_red=np.nan, R1=np.nan, dir1_deg=np.nan)
            if row["n_cells"] >= args.min_cells:
                ia, ir = np.nonzero(mb)
                per_az = np.bincount(ia, minlength=lay.n_az)
                has = per_az >= args.min_cells_per_az
                row["az_coverage_deg"] = float(has.sum() * daz)
                row["max_az_gap_deg"] = float(circ_max_gap(has) * daz)
                phi, vv, ww = phi_az[ia], vh[ia, ir], w_all[ia, ir]
                for nt in (5, 3):
                    beta, se, cond, chi2r = harmonic_fit(phi, vv, ww, nt)
                    if cond <= args.cond_max:
                        for i in range(nt):
                            row[names[i]] = float(beta[i])
                            row["s_" + names[i]] = float(se[i])
                        row.update(n_terms=nt, cond=cond, chi2_red=chi2r)
                        row["R1"] = float(np.hypot(beta[1], beta[2]))
                        row["dir1_deg"] = float(np.degrees(np.arctan2(beta[2], beta[1])) % 360)
                        break
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})


def make_plots(out, lay, geom, clim, masks, noise, sb, rows, args, inv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # noise curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    first = True
    for c in range(lay.ncls):
        if not np.isfinite(noise["sigma_sf"][c]).any():
            continue
        lab = f"count={c+1}" if lay.ncls > 1 else "range structure fn (0-lag)"
        ax[0].semilogy(sb.centres, noise["sigma_sf"][c], "o-", ms=3, label=lab)
        ax[0].semilogy(sb.centres, noise["sigma_t"][c], "x--", ms=3, alpha=0.6,
                       label=("scan-to-scan (upper bd)" if first else None))
        first = False
        if np.isfinite(noise["cutoff"][c]):
            ax[0].axvline(noise["cutoff"][c], color="k", ls=":", lw=1)
    ax[0].axhline(args.sigma_target_mps, color="r", ls=":", lw=1, label="target")
    ax[0].set_xlabel("SNR (dB, as reported)")
    ax[0].set_ylabel("LOS velocity noise sigma (m/s)")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)
    s_lin = 10 ** (sb.centres / 10)
    ax[1].plot(sb.centres, noise["gamma"], "o", ms=3, label="measured gamma (ref-noise corrected)")
    ax[1].plot(sb.centres, noise["gamma_raw"], ".", ms=2, alpha=0.5, label="uncorrected")
    if np.isfinite(noise["c_fit"]):
        ax[1].plot(sb.centres, s_lin / (s_lin + noise["c_fit"]), "-",
                   label=f"s/(s+c), c={noise['c_fit']:.3g}")
        ax[1].plot(sb.centres, s_lin / (s_lin + 1), ":", label="c=1")
    if np.isfinite(noise["ref_snr"]):
        ax[1].axvline(noise["ref_snr"], color="k", ls=":", lw=1, label="anchor SNR (gamma=1 above)")
    ax[1].set_ylim(-0.1, 1.3)
    ax[1].set_xlabel("SNR (dB, as reported)")
    ax[1].set_ylabel("shrinkage gamma")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "noise_and_shrinkage_vs_snr.png", dpi=130)
    plt.close(fig)

    # PPI maps
    daz = 360.0 / lay.n_az
    az_e = np.radians(true_az_deg(np.append(lay.az_deg - daz / 2, lay.az_deg[-1] + daz / 2), args))
    r = lay.range_m
    dr = np.diff(r).mean() if len(r) > 1 else args.dr_m
    r_e = np.append(r - dr / 2, r[-1] + dr / 2)
    g_e = np.sqrt(np.maximum(r_e ** 2 - args.height_m ** 2, 0.0)) / 1000
    X = np.sin(az_e)[:, None] * g_e[None, :]
    Y = np.cos(az_e)[:, None] * g_e[None, :]
    panels = [(f"fraction of scans with SNR >= {args.signal_snr_db:g} dB", clim["sig_frac"], "viridis", (0, 1)),
              ("mean SNR (dB)", clim["mean_snr"], "viridis", None),
              ("SNR anomaly vs background (dB)", clim["anom"], "RdBu_r", (-15, 15)),
              ("per-scan land fraction", masks["land_frac"], "magma", (0, 1)),
              ("neighbour slope / local median", masks["slope_rel"], "RdBu_r", (0, 2)),
              ("scan-to-scan sigma (m/s)", clim["tstd"], "viridis", (0, 0.3)),
              ("record-long velocity std (m/s)", clim["vstd"], "viridis", (0, 1.5)),
              ("final mask (1 = excluded)", masks["final"].astype(float), "gray_r", (0, 1))]
    fig, axs = plt.subplots(2, 4, figsize=(21, 10.5))
    for axx, (title, C, cmap, lim) in zip(axs.ravel(), panels):
        kw = {} if lim is None else dict(vmin=lim[0], vmax=lim[1])
        pc = axx.pcolormesh(X, Y, C, cmap=cmap, shading="flat", rasterized=True, **kw)
        axx.set_aspect("equal")
        axx.set_title(title, fontsize=10)
        axx.set_xlabel("east (km, ground range)")
        axx.set_ylabel("north (km)")
        fig.colorbar(pc, ax=axx, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "masks_ppi.png", dpi=110)
    plt.close(fig)

    # fraction of valid cells above the signal threshold, per scan
    if inv:
        tt = [datetime.fromisoformat(r["time"]) for r in inv if r["time"]]
        ff = [r["frac_snr_ge_signal"] for r in inv if r["time"]]
        fig, axx = plt.subplots(figsize=(13, 3.5))
        axx.plot(tt, ff, ".", ms=4)
        axx.set_ylabel(f"frac cells SNR >= {args.signal_snr_db:g} dB")
        axx.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "signal_fraction_timeseries.png", dpi=120)
        plt.close(fig)

    # harmonic time series
    if rows:
        fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for label in geom.band_labels:
            rr = [x for x in rows if x["band"] == label and x["time"] and x["n_terms"] > 0]
            if not rr:
                continue
            t = [datetime.fromisoformat(x["time"]) for x in rr]
            for axx, key in zip(axs, ["A0", "A1", "B1"]):
                axx.plot(t, [x[key] for x in rr], ".", ms=3, label=label)
        for axx, lab in zip(axs, ["A0 (m/s, mean radial)", "A1 (m/s, ~north)", "B1 (m/s, ~east)"]):
            axx.set_ylabel(lab)
            axx.grid(True, alpha=0.3)
        axs[0].legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(out / "harmonics_timeseries.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("input")
    g.add_argument("--input-dir", required=True, help="Directory of *_combined.npz files")
    g.add_argument("--pattern", default="*_combined.npz")
    g.add_argument("--out-dir", default="dabob_calib_out")
    g.add_argument("--inspect", action="store_true", help="Print keys/shapes of the first two files and exit")
    g.add_argument("--limit-files", type=int, default=0, help="Only use the first N files (testing)")
    g.add_argument("--every", type=int, default=1, help="Use every Nth file (testing)")
    g.add_argument("--snr-key"); g.add_argument("--m1-key"); g.add_argument("--count-key")
    g.add_argument("--az-key"); g.add_argument("--range-key"); g.add_argument("--time-key")
    g.add_argument("--no-count", action="store_true", help="Ignore a count array even if found")
    g.add_argument("--no-pa-mask", action="store_true", help="Don't mask azimuths using antenna_*_pa_enable")
    g.add_argument("--pa-min", type=float, default=1.0,
                   help="Azimuth used only if every antenna's pa_enable >= this (values are averaged per bin)")
    g.add_argument("--axis-order", choices=["az,range", "range,az"])
    g.add_argument("--time-axis", type=int)
    g.add_argument("--zero-is-empty", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--empty-snr-db", type=float, default=-1e9, help="SNR <= this is treated as empty")
    g.add_argument("--min-count", type=float, default=1)
    g.add_argument("--max-count-class", type=int, default=4)

    g = ap.add_argument_group("geometry / conventions")
    g.add_argument("--height-m", type=float, default=66.0)
    g.add_argument("--k-e", type=float, default=4.0 / 3.0)
    g.add_argument("--carrier-hz", type=float, default=9.2e9)
    g.add_argument("--doppler-sign", type=float, default=-1.0,
                   help="v_away = sign * (lambda/2) * f. -1 assumes +Doppler = approaching")
    g.add_argument("--az-start-deg", type=float, default=0.0, help="Centre of azimuth bin 0 if not in file")
    g.add_argument("--heading-offset-deg", type=float, default=0.0,
                   help="Added to file azimuths to get true azimuth (e.g. 21 for Dabob TNoff)")
    g.add_argument("--azimuth-mirror", action="store_true",
                   help="Radar azimuth runs counter-clockwise: true = heading_offset - radar az")
    g.add_argument("--radar-lat", type=float)
    g.add_argument("--radar-lon", type=float)
    g.add_argument("--water-geojson", help="GeoJSON Polygon/MultiPolygon of water (lon/lat); cells outside are excluded")
    g.add_argument("--range0-m", type=float, default=0.0, help="Slant range of bin 0 if not in file")
    g.add_argument("--dr-m", type=float, default=4.8)
    g.add_argument("--min-range-m", type=float, default=0.0)
    g.add_argument("--max-range-m", type=float, default=1e9)
    g.add_argument("--band-edges-deg", type=float, nargs="+", default=[90, 10, 3, 1, 0])

    g = ap.add_argument_group("masking")
    g.add_argument("--max-dt-s", type=float, default=1000, help="Max gap for scan-to-scan pairing")
    g.add_argument("--signal-snr-db", type=float, default=15.0,
                   help="Threshold for the 'fraction of scans with signal' diagnostics")
    g.add_argument("--min-obs", type=int, default=5)
    g.add_argument("--min-obs-frac", type=float, default=0.05)
    g.add_argument("--bg-window", type=int, nargs=2, default=[15, 41], help="az, range bins")
    g.add_argument("--land-anom-db", type=float, default=8.0)
    g.add_argument("--land-tstd-mps", type=float, default=0.03)
    g.add_argument("--land-vstd-mps", type=float, default=0.02,
                   help="Cells whose record-long velocity std is below this are land. Needs the record to span tides")
    g.add_argument("--land-vmax-mps", type=float, default=0.05)
    g.add_argument("--land-frac", type=float, default=0.05)
    g.add_argument("--mixed-slope", type=float, default=0.6,
                   help="Flag cells whose neighbour slope is below this fraction of the local median slope")
    g.add_argument("--slope-window", type=int, nargs=2, default=[5, 21])
    g.add_argument("--mixed-anom-db", type=float, default=5.0)
    g.add_argument("--nbr-window", type=int, nargs=2, default=[3, 9])
    g.add_argument("--dilate", type=int, default=1)

    g = ap.add_argument_group("noise / shrinkage")
    g.add_argument("--snr-bins", type=float, nargs=3, default=[-10, 60, 1], metavar=("LO", "HI", "STEP"))
    g.add_argument("--dv-res", type=float, default=0.002)
    g.add_argument("--dv-max", type=float, default=3.0)
    g.add_argument("--sf-lag0", type=int, default=1)
    g.add_argument("--sf-nlags", type=int, default=4)
    g.add_argument("--shrink-lag", type=int, default=None, help="Default = sf-lag0")
    g.add_argument("--ref-snr-db", type=float, default=None,
                   help="Shrinkage anchor (gamma=1 above). Default: where noise <= --ref-sigma-mps")
    g.add_argument("--ref-sigma-mps", type=float, default=0.08)
    g.add_argument("--min-pairs", type=int, default=500)
    g.add_argument("--max-ref-noise-frac", type=float, default=0.2,
                   help="Shrinkage refs must have noise variance <= this fraction of their total variance")
    g.add_argument("--sigma-target-mps", type=float, default=0.05)
    g.add_argument("--snr-cutoff", type=float, default=None, help="Override the derived cutoff")

    g = ap.add_argument_group("harmonic fits")
    g.add_argument("--bias-correct", action="store_true",
                   help="Divide v by the measured gamma(SNR) table (cells below its range dropped)")
    g.add_argument("--fit-snr-min", type=float, default=None,
                   help="SNR threshold for cells in harmonic fits (default: the derived cutoff). "
                        "Going lower is reasonable with --bias-correct, since weights handle noise")
    g.add_argument("--gamma-min", type=float, default=0.95, help="Reported: SNR where gamma reaches this")
    g.add_argument("--despike-window", type=int, nargs=2, default=[3, 7])
    g.add_argument("--despike-k", type=float, default=4.0)
    g.add_argument("--sigma-floor-mps", type=float, default=0.005)
    g.add_argument("--sigma-default-mps", type=float, default=0.2)
    g.add_argument("--sigma-geo-mps", type=float, default=0.05,
                   help="Added to per-cell noise in weights (real sub-band variability)")
    g.add_argument("--min-cells", type=int, default=200)
    g.add_argument("--min-cells-per-az", type=int, default=3)
    g.add_argument("--cond-max", type=float, default=30.0)
    return ap


def main():
    args = build_parser().parse_args()
    files = list_files(args)
    if not files:
        sys.exit(f"No files matching {args.pattern} in {args.input_dir}")
    if args.inspect:
        for p, _ in files[:2]:
            inspect_file(p)
        return
    if args.shrink_lag is None:
        args.shrink_lag = args.sf_lag0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_notime = sum(1 for _, t in files if t is None)
    print(f"{len(files)} files ({n_notime} without a filename timestamp)")

    lay = None
    for p, _ in files:
        try:
            lay = detect_layout(p, args)
            break
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [warning] could not read {p.name} for layout: {e}")
    if lay is None:
        sys.exit("No readable files")
    geom = make_geometry(lay.range_m, args)
    lat = lon = None
    if args.radar_lat is not None and args.radar_lon is not None:
        lat, lon = cell_latlon(lay, geom, args)
        if args.water_geojson:
            geom.water2d = water_mask_from_geojson(args.water_geojson, lat, lon)
    elif args.water_geojson:
        sys.exit("--water-geojson needs --radar-lat and --radar-lon")
    for i, lab in enumerate(geom.band_labels):
        rr = lay.range_m[geom.band == i]
        if rr.size:
            print(f"  band {lab:10s}: slant {rr.min():7.0f} - {rr.max():7.0f} m ({rr.size} bins)")

    failures = []
    print("Pass 1/4: inventory + climatology ...")
    p1, inv = pass1(files, lay, geom, args, failures)
    nscan = len(inv)
    if nscan == 0:
        sys.exit("No usable scans")
    nv = np.array([r["n_valid"] for r in inv])
    med = np.median(nv)
    for r in inv:
        r["small"] = int(r["n_valid"] < 0.5 * med)
    dts = [r["dt_prev_s"] for r in inv if r["dt_prev_s"] != ""]
    big_gaps = [(inv[i]["time"], inv[i]["dt_prev_s"]) for i in range(nscan)
                if inv[i]["dt_prev_s"] != "" and inv[i]["dt_prev_s"] > args.max_dt_s]
    write_csv(out / "inventory.csv", inv)
    pct = {q: hist_percentile(p1["snr_hist"], p1["hist_edges"], q) for q in (1, 10, 50, 90, 99)}
    print(f"  {nscan} scans, median {med:.0f} valid cells/scan, {sum(r['small'] for r in inv)} small "
          f"(<50% of median), {len(failures)} failures")
    if dts:
        print(f"  median dt {np.median(dts):.0f} s; {len(big_gaps)} gaps > {args.max_dt_s:.0f} s "
              f"(largest {max(dts)/86400:.2f} days)")
    print("  SNR percentiles (dB): " + ", ".join(f"p{q}={v:.1f}" for q, v in pct.items()))

    clim = derive_background(p1, nscan, args)
    print(f"  rough land cells: {int(clim['rough'].sum())}")

    print("Pass 2/4: per-scan land flags + mixed-cell test ...")
    masks = pass2(files, lay, geom, args, clim)
    print(f"  land (per-scan persistence): {int(masks['land2'].sum())}, mixed: {int(masks['mixed'].sum())}, "
          f"final excluded (incl. dilation + geometry): {int(masks['final'].sum())} of {masks['final'].size}")


    print("Pass 3/4: noise curve + shrinkage ...")
    sb = SnrBins(*args.snr_bins)
    p3 = pass3(files, lay, geom, args, masks["final"], sb)
    noise = derive_noise(p3, lay, sb, args)
    print(f"  shrinkage anchored (gamma=1) at SNR >= {noise['ref_snr']:.1f} dB "
          f"({'set' if args.ref_snr_db is not None else f'where noise <= {args.ref_sigma_mps} m/s'})")
    for c in range(lay.ncls):
        if not np.isfinite(noise["sigma_sf"][c]).any():
            continue
        print(f"  class {c}: SNR cutoff for sigma <= {args.sigma_target_mps} m/s: {noise['cutoff'][c]:.1f} dB")
    print(f"  shrinkage fit: c = {noise['c_fit']:.3g} (rms misfit {noise['c_rms']:.3g}); c~1 means SNR scale is consistent")
    print(f"  shrinkage < {100*(1-args.gamma_min):.0f}% above SNR = {noise['gamma_snr']:.1f} dB")

    rows_n = []
    for c in range(lay.ncls):
        for k in range(sb.n):
            if noise["npairs_sf"][c, k] == 0 and noise["npairs_t"][c, k] == 0 and (c > 0 or noise["nsh"][k] == 0):
                continue
            r = dict(count_class=c + 1 if lay.count_key else "", snr_lo=sb.edges[k], snr_hi=sb.edges[k + 1],
                     sigma_los_mps=noise["sigma_sf"][c, k],
                     sigma_hz=noise["sigma_sf"][c, k] / (C_LIGHT / args.carrier_hz / 2),
                     sigma_scan_to_scan_mps=noise["sigma_t"][c, k],
                     n_pairs_sf=int(noise["npairs_sf"][c, k]), n_pairs_t=int(noise["npairs_t"][c, k]))
            for li, lag in enumerate(p3["lags"]):
                r[f"D_lag{lag}"] = float(noise["D"][li, c, k])
            if c == 0:
                r["gamma"] = noise["gamma"][k]
                r["gamma_uncorrected"] = noise["gamma_raw"][k]
                r["n_pairs_gamma"] = int(noise["nsh"][k])
            rows_n.append(r)
    write_csv(out / "noise_curve.csv", rows_n)

    print("Pass 4/4: harmonic fits ...")
    rows = pass4(files, lay, geom, args, masks["final"], noise, sb)
    write_csv(out / "harmonics.csv", rows)
    nfit = sum(1 for r in rows if r["n_terms"] > 0)
    print(f"  {nfit} of {len(rows)} scan-band fits succeeded")

    np.savez_compressed(
        out / "calibration.npz",
        az_deg=lay.az_deg, range_m=lay.range_m, psi_deg=geom.psi_deg, ground_m=geom.ground_m,
        band=geom.band, band_labels=np.array(geom.band_labels),
        sig_frac=clim["sig_frac"], mean_snr=clim["mean_snr"], bg_snr=clim["bg"], anom_db=clim["anom"], tstd_mps=clim["tstd"], vstd_mps=clim["vstd"],
        land_frac=masks["land_frac"], nbr_slope=masks["slope"], nbr_slope_rel=masks["slope_rel"], mask_rough=clim["rough"],
        mask_land=masks["land2"], mask_mixed=masks["mixed"], mask_final=masks["final"],
        snr_bin_edges=sb.edges, sigma_sf_mps=noise["sigma_sf"], sigma_t_mps=noise["sigma_t"],
        gamma=noise["gamma"], gamma_raw=noise["gamma_raw"], c_fit=noise["c_fit"], snr_cutoff=noise["cutoff"],
        cell_lat=lat if lat is not None else np.array(np.nan),
        cell_lon=lon if lon is not None else np.array(np.nan),
        water_mask=geom.water2d if geom.water2d is not None else np.array(np.nan),
        config=json.dumps({k: v for k, v in vars(args).items()}, default=str))
    if failures:
        with open(out / "failures.txt", "w") as fh:
            for p, e in failures:
                fh.write(f"{p}\t{e}\n")
    make_plots(out, lay, geom, clim, masks, noise, sb, rows, args, inv)
    print(f"Outputs in {out}/: inventory.csv noise_curve.csv harmonics.csv calibration.npz "
          f"noise_and_shrinkage_vs_snr.png masks_ppi.png harmonics_timeseries.png signal_fraction_timeseries.png"
          + (" failures.txt" if failures else ""))


if __name__ == "__main__":
    main()
