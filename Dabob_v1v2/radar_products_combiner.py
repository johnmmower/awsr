#!/usr/bin/env python3
"""
Combine per-antenna Doppler products that are ALREADY computed and saved
by radar_archive_service.py's live _products.npz -- no raw .bin needed.

For each world-angle bin, averages across whichever antennas actually
hit that bin during the archive window:

  - power_linear, noise_floor: averaged in LINEAR units (these are
    additive power quantities), then power_db / snr_db are recomputed
    from those linear averages.
  - moment1_hz, moment2_hz: averaged directly (already linear, Hz units)
    -- a plain unweighted mean across antennas.

WHY NOT JUST AVERAGE power_db / snr_db DIRECTLY:
  power_db and snr_db are logarithms (10*log10(...)). The mean of
  logarithms is NOT the same as the log of the mean -- averaging dB
  values directly is a well-known biased estimator (Jensen's inequality:
  log is concave, so mean(log(x)) <= log(mean(x)), i.e. dB-averaging
  systematically UNDERESTIMATES the combined linear power/SNR). So this
  script averages in linear space first and converts to dB only at the
  end, for power_db/snr_db specifically.

CAVEAT ON MOMENT AVERAGING (flagging explicitly, a design choice, not a
derived/verified method): moment1_hz/moment2_hz are averaged with a
plain unweighted mean across the antennas that hit each bin. This treats
each antenna's per-bin moment estimate as equally reliable. If one
antenna's last hit at a bin had much lower SNR than another's, its
moment estimate is noisier but weighted the same -- an SNR-weighted
average would be more rigorous but wasn't what was asked for here. Easy
to add later if you want it (see _combine_moments below).

This works directly on the small _products.npz files your archive
service is already producing on the NUC -- nothing else needs to be
downloaded or reprocessed.
"""

from pathlib import Path

import numpy as np

RANGE_RESOLVED = ("power_linear", "power_db", "snr_db", "noise_floor",
                   "moment1_hz", "moment2_hz")


def _discover_antennas(npz_files) -> list:
    indices = set()
    for key in npz_files:
        parts = key.split("_", 2)
        if len(parts) == 3 and parts[0] == "antenna":
            indices.add(int(parts[1]))
    return sorted(indices)


def _combine_moments(moment_stack: np.ndarray) -> np.ndarray:
    """moment_stack: shape (n_antennas_present, num_range_bins).
    Plain unweighted mean across antennas -- see CAVEAT in module docstring
    if you want SNR-weighted averaging instead."""
    return np.mean(moment_stack, axis=0)


def combine_products_npz(products_npz_path):
    """Load an existing _products.npz and return a dict of combined_*
    arrays (does not modify or require the raw .bin file at all)."""
    d = np.load(products_npz_path)
    antennas = _discover_antennas(d.files)
    if not antennas:
        raise ValueError(f"No antenna_* keys found in {products_npz_path}")

    num_bins, num_range_bins = d[f"antenna_{antennas[0]}_power_linear"].shape

    combined_power_linear = np.full((num_bins, num_range_bins), np.nan)
    combined_power_db = np.full((num_bins, num_range_bins), np.nan)
    combined_snr_db = np.full((num_bins, num_range_bins), np.nan)
    combined_noise_floor = np.full((num_bins, num_range_bins), np.nan)
    combined_moment1_hz = np.full((num_bins, num_range_bins), np.nan)
    combined_moment2_hz = np.full((num_bins, num_range_bins), np.nan)
    combined_num_antennas = np.zeros(num_bins, dtype=np.int64)

    per_antenna_power_linear = {a: d[f"antenna_{a}_power_linear"] for a in antennas}
    per_antenna_noise_floor = {a: d[f"antenna_{a}_noise_floor"] for a in antennas}
    per_antenna_moment1 = {a: d[f"antenna_{a}_moment1_hz"] for a in antennas}
    per_antenna_moment2 = {a: d[f"antenna_{a}_moment2_hz"] for a in antennas}

    for b in range(num_bins):
        present = [a for a in antennas if not np.isnan(per_antenna_power_linear[a][b]).all()]
        combined_num_antennas[b] = len(present)
        if not present:
            continue

        power_stack = np.stack([per_antenna_power_linear[a][b] for a in present], axis=0)
        noise_stack = np.stack([per_antenna_noise_floor[a][b] for a in present], axis=0)
        moment1_stack = np.stack([per_antenna_moment1[a][b] for a in present], axis=0)
        moment2_stack = np.stack([per_antenna_moment2[a][b] for a in present], axis=0)

        avg_power_linear = np.mean(power_stack, axis=0)
        avg_noise_floor = np.mean(noise_stack, axis=0)

        combined_power_linear[b] = avg_power_linear
        combined_noise_floor[b] = avg_noise_floor
        with np.errstate(divide='ignore', invalid='ignore'):
            combined_power_db[b] = 10.0 * np.log10(
                np.where(avg_power_linear > 0, avg_power_linear, np.nan))
            snr_linear = np.where(avg_noise_floor > 0,
                                   avg_power_linear / avg_noise_floor, np.inf)
            combined_snr_db[b] = 10.0 * np.log10(snr_linear)

        combined_moment1_hz[b] = _combine_moments(moment1_stack)
        combined_moment2_hz[b] = _combine_moments(moment2_stack)

    return {
        "combined_power_linear": combined_power_linear,
        "combined_power_db": combined_power_db,
        "combined_snr_db": combined_snr_db,
        "combined_noise_floor": combined_noise_floor,
        "combined_moment1_hz": combined_moment1_hz,
        "combined_moment2_hz": combined_moment2_hz,
        "combined_num_antennas": combined_num_antennas,
    }


def add_combined_products(products_npz_path, output_path=None):
    """Load an existing _products.npz, compute combined_* fields, and
    write out a new .npz containing the original per-antenna arrays PLUS
    the combined_* arrays. output_path defaults to
    "<stem>_combined.npz" alongside the input."""
    d = np.load(products_npz_path)
    combined = combine_products_npz(products_npz_path)
    out = {k: d[k] for k in d.files}
    out.update(combined)

    if output_path is None:
        p = Path(products_npz_path)
        output_path = str(p.with_name(p.stem + "_combined.npz"))
    np.savez(output_path, **out)
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("products_npz", help="Path to an existing <timestamp>_products.npz")
    ap.add_argument("--output", default=None, help="Output path (default: <stem>_combined.npz)")
    args = ap.parse_args()
    out_path = add_combined_products(args.products_npz, args.output)
    print(f"Wrote combined products to {out_path}")
