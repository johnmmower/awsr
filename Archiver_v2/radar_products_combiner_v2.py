#!/usr/bin/env python3
"""
Combine per-antenna products from a *_products_v2.npz (written by
radar_reprocess_v2.py) across antennas, per world-angle bin.

Differences from radar_products_combiner.py (v1), flagged explicitly:
  - Antennas whose bin has pa_enable != 1 are EXCLUDED. v1 averaged them
    in, so noise-only looks (PA off) diluted the combined moments.
    combined_num_antennas now counts only PA-on antennas with data.
  - v1-style fields (combined_power_linear, _noise_floor, _power_db,
    _snr_db, _moment1_hz, _moment2_hz) keep their names and methods
    (linear averaging for power, plain mean for moments), but over PA-on
    antennas only.
  - v2 fields are combined per CELL over antennas where the look was
    detected (detected == 1, moment finite):
      combined_moment1_pk_hz, combined_moment2_pk_hz : plain mean
      combined_snr_band_db, combined_snr_pk_db       : 10 log10 of the mean
                                                       LINEAR value
      combined_n_detected (int8, per cell)           : how many antennas
                                                       went into the mean
    DESIGN CHOICE: unweighted mean, like v1. An SNR- or inverse-variance-
    weighted mean would be better once the noise-vs-SNR curve for v2 is
    measured (dabob_doppler_calibrate.py gives it).
"""

from pathlib import Path

import numpy as np

V1_FIELDS = ("power_linear", "power_db", "snr_db", "noise_floor", "moment1_hz", "moment2_hz")


def _antennas(keys):
    out = set()
    for k in keys:
        parts = k.split("_", 2)
        if len(parts) == 3 and parts[0] == "antenna" and parts[1].isdigit():
            out.add(int(parts[1]))
    return sorted(out)


def combine_v2(d) -> dict:
    ants = _antennas(d.files if hasattr(d, "files") else d.keys())
    if not ants:
        raise ValueError("no antenna_* keys")
    nb, nr = d[f"antenna_{ants[0]}_power_linear"].shape

    # per-(antenna, azimuth) usable: has data and PA on
    use = {}
    for a in ants:
        has = ~np.isnan(d[f"antenna_{a}_power_linear"]).all(axis=1)
        pa_key = f"antenna_{a}_pa_enable"
        pa_ok = (np.nan_to_num(d[pa_key], nan=0.0) >= 1.0) if pa_key in d else np.ones(nb, bool)
        use[a] = has & pa_ok

    n_ant = np.sum([use[a] for a in ants], axis=0).astype(np.int64)

    def stack(name):
        return np.stack([np.where(use[a][:, None], d[f"antenna_{a}_{name}"], np.nan) for a in ants])

    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pl = np.nanmean(stack("power_linear"), axis=0)
            nf = np.nanmean(stack("noise_floor"), axis=0)
            m1 = np.nanmean(stack("moment1_hz"), axis=0)
            m2 = np.nanmean(stack("moment2_hz"), axis=0)
        out = {
            "combined_power_linear": pl,
            "combined_noise_floor": nf,
            "combined_power_db": 10 * np.log10(np.where(pl > 0, pl, np.nan)),
            "combined_snr_db": 10 * np.log10(np.where(nf > 0, pl / nf, np.nan)),
            "combined_moment1_hz": m1,
            "combined_moment2_hz": m2,
            "combined_num_antennas": n_ant,
        }

    v2_ok = all(f"antenna_{a}_moment1_pk_hz" in d for a in ants)
    if v2_ok:
        det = []
        for a in ants:
            dd = np.nan_to_num(d[f"antenna_{a}_detected"], nan=0.0) >= 1.0
            dd &= np.isfinite(d[f"antenna_{a}_moment1_pk_hz"])
            det.append(dd & use[a][:, None])
        det = np.stack(det)
        ndet = det.sum(axis=0)

        def dmean(name, linear_db=False):
            x = np.stack([d[f"antenna_{a}_{name}"].astype(np.float64) for a in ants])
            if linear_db:
                x = 10 ** (x / 10)
            x = np.where(det & np.isfinite(x), x, 0.0)
            n = (det & np.isfinite(np.stack([d[f"antenna_{a}_{name}"] for a in ants]))).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(n > 0, x.sum(axis=0) / np.maximum(n, 1), np.nan)
                return 10 * np.log10(np.where(m > 0, m, np.nan)) if linear_db else m

        out.update({
            "combined_moment1_pk_hz": dmean("moment1_pk_hz"),
            "combined_moment2_pk_hz": dmean("moment2_pk_hz"),
            "combined_snr_band_db": dmean("snr_band_db", linear_db=True),
            "combined_snr_pk_db": dmean("snr_pk_db", linear_db=True),
            "combined_n_detected": ndet.astype(np.int8),
        })
    return out


def add_combined_v2(products_npz_path, output_path=None, compress=True, float32=False, slim=False):
    """slim=True drops the per-antenna 2-D arrays and keeps the combined
    fields, 1-D per-antenna metadata (pa_enable, timestamp_us, hit_count) and
    scalars -- roughly a tenth of the size, for pulling over a slow link."""
    d = np.load(products_npz_path)
    out = {k: d[k] for k in d.files}
    if slim:
        out = {k: v for k, v in out.items() if not (k.startswith("antenna_") and np.ndim(v) == 2)}
    out.update(combine_v2(d))
    if float32:
        out = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64
                   and v.ndim == 2 else v) for k, v in out.items()}
    if output_path is None:
        p = Path(products_npz_path)
        output_path = str(p.with_name(p.stem + "_combined.npz"))
    (np.savez_compressed if compress else np.savez)(output_path, **out)
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products_npz")
    ap.add_argument("--output")
    args = ap.parse_args()
    print(add_combined_v2(args.products_npz, args.output))
