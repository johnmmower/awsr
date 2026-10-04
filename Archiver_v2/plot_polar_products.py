#!/usr/bin/env python3
"""
Render polar (PPI-style) PNG images from a radar archive product file
produced by radar_archive_service.py (the "<timestamp>_products.npz" file).

For each antenna, renders:
  - Range-resolved products (shape (num_bins, num_range_bins)) as a 2-D
    polar heatmap: power_linear, power_db, snr_db, noise_floor,
    moment1_hz, moment2_hz.
  - Angle-only products (shape (num_bins,)) as a thin colored ring:
    pa_enable, hit_count.

Missing data (NaN -- bins never written during the archive window) is
rendered transparent/blank rather than a misleading color.

=== DESIGN CHOICES (flagging explicitly) ===
  - Polar orientation: 0 degrees at top (North), angle increasing
    CLOCKWISE -- a common radar/compass display convention. Not something
    the .npz file specifies; change ORIENTATION_* constants below if you
    want a different convention (e.g. mathematical counterclockwise-from-
    East).
  - Range axis: plotted in range-bin index (0..num_range_bins-1), NOT
    physical distance, since the .npz file doesn't carry a range-bin-to-
    meters conversion. If you have a range resolution (meters/bin), pass
    it via --range-resolution-m to label the radial axis in meters
    instead.
  - Colormaps chosen per product type (see PRODUCT_CMAPS) as a reasonable
    default (viridis for power/SNR, coolwarm diverging for moment1 since
    it's signed velocity/frequency-like, viridis for moment2/spread since
    it's non-negative). These are display choices, not physical
    requirements -- adjust to taste.
  - --log-scale: applies log-scale coloring (LogNorm) to power_linear and
    noise_floor only (LOG_APPLICABLE_PRODUCTS) -- the two range-resolved
    products that are strictly non-negative and can span wide dynamic
    range. Ignored for power_db/snr_db (already log by definition),
    moment1_hz (signed), and moment2_hz/hit_count (not wide-dynamic-range
    power quantities). Non-positive values are masked out (log undefined
    for <= 0), same treatment as NaN.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Range-resolved products: (num_bins, num_range_bins)
RANGE_RESOLVED_PRODUCTS = [
    "power_linear", "power_db", "snr_db",
    "noise_floor", "moment1_hz", "moment2_hz",
]
# Angle-only products: (num_bins,)
ANGLE_ONLY_PRODUCTS = ["pa_enable", "hit_count"]

PRODUCT_CMAPS = {
    "power_linear": "viridis",
    "power_db": "viridis",
    "snr_db": "viridis",
    "noise_floor": "viridis",
    "moment1_hz": "coolwarm",   # signed (Doppler mean freq) -> diverging
    "moment2_hz": "viridis",    # non-negative spread
    "pa_enable": "coolwarm",
    "hit_count": "viridis",
}

PRODUCT_LABELS = {
    "power_linear": "Peak Power (linear)",
    "power_db": "Peak Power (dB)",
    "snr_db": "SNR (dB)",
    "noise_floor": "Noise Floor (linear power)",
    "moment1_hz": "1st Doppler Moment (Hz)",
    "moment2_hz": "2nd Doppler Moment / Spectral Width (Hz)",
    "pa_enable": "PA Enable",
    "hit_count": "Hit Count",
}

# 0 deg at top (North), clockwise -- common radar/compass convention.
ORIENTATION_ZERO_LOCATION = "N"
ORIENTATION_DIRECTION = -1  # -1 = clockwise, 1 = counterclockwise

# Products where log-scale coloring makes sense: strictly non-negative,
# power-like quantities that can span wide dynamic range. Deliberately
# excludes power_db/snr_db (already log-scale by definition), moment1_hz
# (signed), moment2_hz and hit_count (not wide-dynamic-range power
# quantities) -- --log-scale is silently ignored for those.
LOG_APPLICABLE_PRODUCTS = {"power_linear", "noise_floor"}


def _theta_edges_rad(num_bins: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, num_bins + 1)


def plot_range_resolved(ax, data: np.ndarray, theta_edges: np.ndarray,
                         cmap: str, label: str, product: str,
                         range_resolution_m=None, log_scale: bool = False):
    num_bins, num_range_bins = data.shape
    if range_resolution_m is not None:
        r_edges = np.arange(num_range_bins + 1) * range_resolution_m
        r_label = "Range (m)"
    else:
        r_edges = np.arange(num_range_bins + 1)
        r_label = "Range bin"

    masked = np.ma.masked_invalid(data.T)  # shape (num_range_bins, num_bins)

    use_log = log_scale and product in LOG_APPLICABLE_PRODUCTS
    if use_log:
        # log scale requires strictly positive values -- mask <= 0 too,
        # on top of the existing NaN mask.
        masked = np.ma.masked_less_equal(masked, 0.0)
        if masked.count() == 0:
            # nothing positive to show -- fall back to linear rather than
            # crashing on an empty-range LogNorm.
            use_log = False
        else:
            norm = LogNorm(vmin=masked.min(), vmax=masked.max())

    if use_log:
        mesh = ax.pcolormesh(theta_edges, r_edges, masked, cmap=cmap,
                              shading="flat", norm=norm)
        label = label + " [log scale]"
    else:
        mesh = ax.pcolormesh(theta_edges, r_edges, masked, cmap=cmap, shading="flat")

    ax.set_theta_zero_location(ORIENTATION_ZERO_LOCATION)
    ax.set_theta_direction(ORIENTATION_DIRECTION)
    ax.set_title(label)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(label)
    ax.set_ylabel(r_label, labelpad=30)


def plot_angle_only(ax, data: np.ndarray, theta_edges: np.ndarray,
                     cmap: str, label: str):
    masked = np.ma.masked_invalid(data.reshape(1, -1))  # shape (1, num_bins)
    r_edges = np.array([0.0, 1.0])
    mesh = ax.pcolormesh(theta_edges, r_edges, masked, cmap=cmap, shading="flat")
    ax.set_theta_zero_location(ORIENTATION_ZERO_LOCATION)
    ax.set_theta_direction(ORIENTATION_DIRECTION)
    ax.set_title(label)
    ax.set_yticklabels([])
    cbar = plt.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(label)


def render_npz_to_pngs(npz_path: str, output_dir: str = None,
                        range_resolution_m: float = None, dpi: int = 150,
                        log_scale: bool = False):
    data = np.load(npz_path)
    num_bins = int(data["num_bins"])
    antenna_indices = data["antenna_indices"]

    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(npz_path))
    os.makedirs(output_dir, exist_ok=True)

    theta_edges = _theta_edges_rad(num_bins)
    written = []

    for ant in antenna_indices:
        for product in RANGE_RESOLVED_PRODUCTS:
            key = f"antenna_{ant}_{product}"
            if key not in data:
                continue
            arr = data[key]
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="polar")
            plot_range_resolved(ax, arr, theta_edges, PRODUCT_CMAPS[product],
                                 f"Antenna {ant} - {PRODUCT_LABELS[product]}",
                                 product, range_resolution_m=range_resolution_m,
                                 log_scale=log_scale)
            out_path = os.path.join(output_dir, f"{base_name}_antenna{ant}_{product}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(out_path)

        for product in ANGLE_ONLY_PRODUCTS:
            key = f"antenna_{ant}_{product}"
            if key not in data:
                continue
            arr = data[key]
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="polar")
            plot_angle_only(ax, arr, theta_edges, PRODUCT_CMAPS[product],
                             f"Antenna {ant} - {PRODUCT_LABELS[product]}")
            out_path = os.path.join(output_dir, f"{base_name}_antenna{ant}_{product}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(out_path)

    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_path", help="path to a <timestamp>_products.npz file")
    ap.add_argument("--output-dir", default=None,
                     help="where to write PNGs (default: same dir as the npz file)")
    ap.add_argument("--range-resolution-m", type=float, default=None,
                     help="meters per range bin, to label the radial axis in "
                          "meters instead of bin index (optional -- not stored "
                          "in the npz file, so must be supplied if wanted)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--log-scale", action="store_true",
                     help="use log-scale coloring for power_linear and "
                          "noise_floor (ignored for other products, which "
                          "are already dB, signed, or narrow-range)")
    args = ap.parse_args()

    written = render_npz_to_pngs(args.npz_path, args.output_dir,
                                  args.range_resolution_m, args.dpi,
                                  log_scale=args.log_scale)
    print(f"Wrote {len(written)} PNG(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
