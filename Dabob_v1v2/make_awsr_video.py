#!/usr/bin/env python3
"""
AWSR npz frame parser -> polar PPI video generator for SNR (dB) and/or 1st moment
(Doppler, Hz or m/s), with moment display alpha-weighted by SNR (low SNR = faded).

Confirmed structure (from --inspect on real data):
    bin_centers_deg:        (360,)        azimuth bin centers, degrees
    {source}_snr_db:        (360, 1024)   azimuth x range
    {source}_moment1_hz:    (360, 1024)   azimuth x range, Doppler frequency (Hz)
  where {source} in: antenna_0, antenna_1, antenna_2, antenna_3, combined

OPEN ASSUMPTIONS - confirm before trusting output:
  - Range axis is bin INDEX unless --range_bin_width_m is given.
  - moment1_hz stays raw Hz unless --center_freq_hz is given for m/s conversion.
    Do NOT trust a made-up default center frequency here - confirm what was
    actually configured for this specific collect.
  - SNR-based alpha weighting: by default the fade range (fully transparent to
    fully opaque) is set from the 5th/95th percentile of SNR across all loaded
    frames. This is a visualization convenience, not a calibrated detection
    threshold for this radar/hardware. Override with --snr_alpha_floor_db /
    --snr_alpha_ceil_db if you have real thresholds (e.g. a known noise floor).
  - Frame order = sorted filename order (works for timestamp-prefixed filenames
    like 20260908133549_products_combined.npz - verify for all files).

Usage:
    python make_awsr_video.py --input_dir ./data/radar_products --inspect

    python make_awsr_video.py --input_dir ./data/radar_products \
        --output_dir ./out --source combined --fields snr moment --fps 10

    # with real units + explicit alpha thresholds, once confirmed:
    python make_awsr_video.py --input_dir ./data/radar_products \
        --output_dir ./out --source combined --fields snr moment \
        --range_bin_width_m 7.5 --center_freq_hz 9.3e9 \
        --snr_alpha_floor_db 0 --snr_alpha_ceil_db 15
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

C = 299792458.0  # m/s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--source", default="combined",
                     choices=["combined", "antenna_0", "antenna_1", "antenna_2", "antenna_3"])
    ap.add_argument("--fields", nargs="+", choices=["snr", "moment"],
                     default=["snr", "moment"])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--range_bin_width_m", type=float, default=None)
    ap.add_argument("--center_freq_hz", type=float, default=None)
    ap.add_argument("--snr_alpha_floor_db", type=float, default=None,
                     help="SNR (dB) at/below which moment is fully transparent. "
                          "Default: 5th percentile of observed SNR.")
    ap.add_argument("--snr_alpha_ceil_db", type=float, default=None,
                     help="SNR (dB) at/above which moment is fully opaque. "
                          "Default: 95th percentile of observed SNR.")
    ap.add_argument("--min_alpha", type=float, default=0.0,
                     help="Alpha floor for moment display even at lowest SNR (0-1).")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        sys.exit(f"No files matching {args.pattern!r} found in {args.input_dir}")

    first = np.load(files[0])
    keys = list(first.keys())

    if args.inspect:
        print(f"Inspecting: {files[0]}")
        print(f"Total matching files: {len(files)}")
        for k in keys:
            arr = first[k]
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        return

    snr_key = f"{args.source}_snr_db"
    mom_key = f"{args.source}_moment1_hz"
    need_snr = "snr" in args.fields or "moment" in args.fields  # moment always needs snr to weight it
    if need_snr and snr_key not in keys:
        sys.exit(f"Key {snr_key!r} not found. Available keys: {keys}")
    if "moment" in args.fields and mom_key not in keys:
        sys.exit(f"Key {mom_key!r} not found. Available keys: {keys}")

    az_deg = first["bin_centers_deg"]
    n_az, n_range = first[snr_key].shape
    theta = np.deg2rad(az_deg)

    if args.range_bin_width_m is not None:
        r = np.arange(n_range) * args.range_bin_width_m / 1000.0  # km
        r_label = "Range (km)"
    else:
        r = np.arange(n_range)
        r_label = "Range bin index (width unconfirmed)"

    theta_edges = np.append(theta, theta[0] + 2 * np.pi)
    r_edges = np.append(r, r[-1] + (r[-1] - r[-2] if len(r) > 1 else 1))
    Theta, R = np.meshgrid(theta_edges, r_edges, indexing="ij")

    print(f"Frames found: {len(files)}")
    print(f"Source: {args.source}  |  azimuth bins: {n_az}  |  range bins: {n_range}")
    if args.range_bin_width_m is None:
        print("NOTE: range axis is in bin index, not meters (no --range_bin_width_m given).")
    if "moment" in args.fields and args.center_freq_hz is None:
        print("NOTE: moment plotted as raw Doppler Hz, not velocity (no --center_freq_hz given).")

    os.makedirs(args.output_dir, exist_ok=True)

    snr_stack, mom_stack = [], []
    for f in files:
        d = np.load(f)
        if need_snr:
            snr_stack.append(d[snr_key])
        if "moment" in args.fields:
            m = d[mom_key]
            if args.center_freq_hz is not None:
                m = m * C / (2.0 * args.center_freq_hz)  # Hz -> m/s
            mom_stack.append(m)

    def make_snr_video(stack, out_path):
        arr = np.stack(stack)
        vmin, vmax = np.nanpercentile(arr, [2, 98])
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pm = ax.pcolormesh(Theta, R, stack[0], cmap="viridis", vmin=vmin, vmax=vmax,
                            shading="flat", edgecolors="none")
        fig.colorbar(pm, ax=ax, label="SNR (dB)", pad=0.1)
        title = ax.set_title(os.path.basename(files[0]), fontsize=9)
        ax.set_rlabel_position(135)
        fig.text(0.02, 0.02, r_label, fontsize=8)

        def update(i):
            pm.set_array(stack[i].ravel())
            title.set_text(os.path.basename(files[i]))
            return pm, title

        ani = animation.FuncAnimation(fig, update, frames=len(stack), blit=False)
        _save(ani, out_path, args.fps)
        plt.close(fig)

    def make_moment_video_weighted(mom_stack, snr_stack, out_path):
        mom_arr = np.stack(mom_stack)
        vmax = np.nanpercentile(np.abs(mom_arr), 98)
        vmin = -vmax
        unit = "m/s" if args.center_freq_hz is not None else "Hz"

        snr_arr = np.stack(snr_stack)
        floor = args.snr_alpha_floor_db
        ceil = args.snr_alpha_ceil_db
        if floor is None:
            floor = float(np.nanpercentile(snr_arr, 5))
        if ceil is None:
            ceil = float(np.nanpercentile(snr_arr, 95))
        if ceil <= floor:
            ceil = floor + 1.0
        print(f"Moment alpha weighting by SNR: floor={floor:.2f} dB (transparent), "
              f"ceil={ceil:.2f} dB (opaque), min_alpha={args.min_alpha}")

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap("RdBu_r")

        def rgba_frame(mom, snr):
            rgba = cmap_obj(norm(mom))
            alpha = np.clip((snr - floor) / (ceil - floor), 0.0, 1.0)
            alpha = args.min_alpha + (1.0 - args.min_alpha) * alpha
            rgba[..., 3] = alpha
            return rgba.reshape(-1, 4)

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pm = ax.pcolormesh(Theta, R, mom_stack[0], cmap=cmap_obj, norm=norm,
                            shading="flat", edgecolors="none")
        pm.set_facecolors(rgba_frame(mom_stack[0], snr_stack[0]))
        # IMPORTANT: pcolormesh keeps an internal data array and recomputes
        # facecolors from it on every redraw, silently overwriting our manual
        # per-frame alpha blending (this was the bug causing a frozen,
        # single-frame-looking video). Clearing it makes set_facecolors stick.
        pm.set_array(None)
        fig.colorbar(pm, ax=ax, label=f"1st moment ({unit}), faded where SNR is low", pad=0.1)
        title = ax.set_title(os.path.basename(files[0]), fontsize=9)
        ax.set_rlabel_position(135)
        fig.text(0.02, 0.02, r_label, fontsize=8)

        def update(i):
            pm.set_facecolors(rgba_frame(mom_stack[i], snr_stack[i]))
            title.set_text(os.path.basename(files[i]))
            return pm, title

        ani = animation.FuncAnimation(fig, update, frames=len(mom_stack), blit=False)
        _save(ani, out_path, args.fps)
        plt.close(fig)

    if "snr" in args.fields:
        make_snr_video(snr_stack, os.path.join(args.output_dir, f"{args.source}_snr_video.mp4"))

    if "moment" in args.fields:
        make_moment_video_weighted(mom_stack, snr_stack,
                                    os.path.join(args.output_dir, f"{args.source}_moment_video.mp4"))


def _save(ani, out_path, fps):
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(out_path, writer=writer)
    except FileNotFoundError:
        print("ffmpeg not found on PATH - falling back to animated GIF.")
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
        print(f"Wrote {gif_path}")
        return
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
