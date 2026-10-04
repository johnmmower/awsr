#!/usr/bin/env python3
"""
Build a combined per-pixel (azimuth, range) time series from all AWSR npz frames
in a directory, then interactively pick any pixel to plot SNR (dB) and/or
Doppler (1st moment) over time.

ASSUMPTIONS - confirm before trusting results:
  - Filename encodes a 14-digit timestamp prefix (YYYYMMDDHHMMSS), e.g.
    20260908133549_products_combined.npz. This is parsed to epoch seconds.
  - That timestamp is treated as UTC by default (--timezone UTC). If your
    capture system logs local time instead, pass --timezone local.
  - Range axis is bin index unless --range_bin_width_m is given.
  - moment1_hz stays raw Hz unless --center_freq_hz is given (v = f_d*c/(2*f0)).
    Confirm the actual center frequency used for this collect before converting.
  - Arrays are stored as float32 by default (source data is float64) to keep
    memory/disk size down for ~500+ frame stacks. Use --dtype float64 to keep
    full precision if you need it.

USAGE

  Step 1 - build the combined time-series file:
    python3 awsr_timeseries.py build --input_dir ./data/radar_products \
        --output_file ./out/combined_timeseries.npz --source combined \
        --range_bin_width_m 4.8 --center_freq_hz 9200e6

  Step 2 - interactively pick a pixel and plot its time series:
    python3 awsr_timeseries.py view --timeseries_file ./out/combined_timeseries.npz
"""

import argparse
import datetime
import glob
import os
import re
import sys

import numpy as np

C = 299792458.0  # m/s
TS_RE = re.compile(r"(\d{14})")


def parse_epoch(filename, timezone):
    m = TS_RE.match(os.path.basename(filename))
    if not m:
        return None
    dt = datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if timezone.lower() == "utc":
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone()  # interpret naive dt as local time
    return dt.timestamp()


def cmd_build(args):
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        sys.exit(f"No files matching {args.pattern!r} found in {args.input_dir}")

    snr_key = f"{args.source}_snr_db"
    mom_key = f"{args.source}_moment1_hz"

    first = np.load(files[0])
    keys = list(first.keys())
    if "snr" in args.fields and snr_key not in keys:
        sys.exit(f"Key {snr_key!r} not found. Available keys: {keys}")
    if "moment" in args.fields and mom_key not in keys:
        sys.exit(f"Key {mom_key!r} not found. Available keys: {keys}")

    bin_centers_deg = first["bin_centers_deg"]
    n_az, n_range = first[snr_key if "snr" in args.fields else mom_key].shape

    dtype = np.float32 if args.dtype == "float32" else np.float64

    epochs, good_files = [], []
    for f in files:
        e = parse_epoch(f, args.timezone)
        if e is None:
            print(f"WARNING: could not parse timestamp from {os.path.basename(f)}; skipping")
            continue
        epochs.append(e)
        good_files.append(f)

    n = len(good_files)
    print(f"Building time series: {n} frames  |  az={n_az}  range={n_range}  "
          f"source={args.source}  dtype={dtype.__name__}")
    print(f"Filename timestamp interpreted as: {args.timezone}")

    snr_ts = np.empty((n, n_az, n_range), dtype=dtype) if "snr" in args.fields else None
    mom_ts = np.empty((n, n_az, n_range), dtype=dtype) if "moment" in args.fields else None

    for i, f in enumerate(good_files):
        d = np.load(f)
        if snr_ts is not None:
            snr_ts[i] = d[snr_key].astype(dtype)
        if mom_ts is not None:
            m = d[mom_key]
            if args.center_freq_hz is not None:
                m = m * C / (2.0 * args.center_freq_hz)
            mom_ts[i] = m.astype(dtype)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  loaded {i + 1}/{n}")

    if args.range_bin_width_m is not None:
        range_axis = np.arange(n_range) * args.range_bin_width_m / 1000.0  # km
        range_unit = "km"
    else:
        range_axis = np.arange(n_range)
        range_unit = "bin_index"

    out = dict(
        epoch_s=np.array(epochs, dtype=np.float64),
        bin_centers_deg=bin_centers_deg,
        range_axis=range_axis,
        range_unit=range_unit,
        source=args.source,
        moment_unit=("m/s" if args.center_freq_hz is not None else "Hz"),
        timezone_assumed=args.timezone,
        source_files=np.array([os.path.basename(f) for f in good_files]),
    )
    if snr_ts is not None:
        out["snr_db"] = snr_ts
    if mom_ts is not None:
        out["moment"] = mom_ts

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    np.savez_compressed(args.output_file, **out)
    size_mb = os.path.getsize(args.output_file) / 1e6
    print(f"Wrote {args.output_file} ({size_mb:.1f} MB)")


def cmd_view(args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons

    d = np.load(args.timeseries_file, allow_pickle=True)
    epoch_s = d["epoch_s"]
    bin_centers_deg = d["bin_centers_deg"]
    range_axis = d["range_axis"]
    range_unit = str(d["range_unit"])
    moment_unit = str(d["moment_unit"]) if "moment_unit" in d else "Hz"
    has_snr = "snr_db" in d
    has_mom = "moment" in d

    if not has_snr and not has_mom:
        sys.exit("Timeseries file has neither snr_db nor moment arrays.")

    if not (has_snr and has_mom):
        print("NOTE: SNR-threshold overlay needs both snr_db and moment arrays; "
              "this file has only one, so the slider/overlay will be skipped.")

    bg = d["snr_db"].mean(axis=0) if has_snr else d["moment"].mean(axis=0)
    times = np.array([datetime.datetime.fromtimestamp(e, tz=datetime.timezone.utc)
                       for e in epoch_s])

    fig = plt.figure(figsize=(8, 10))
    show_slider = has_snr and has_mom
    if show_slider:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 0.06], hspace=0.4)
        ax_img = fig.add_subplot(gs[0])
        ax_ts = fig.add_subplot(gs[1])
        slider_ax = fig.add_subplot(gs[2])
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.35)
        ax_img = fig.add_subplot(gs[0])
        ax_ts = fig.add_subplot(gs[1])
        slider_ax = None

    im = ax_img.imshow(bg, origin="lower", aspect="auto",
                        extent=[range_axis[0], range_axis[-1],
                                bin_centers_deg[0], bin_centers_deg[-1]],
                        cmap="viridis")
    ax_img.set_xlabel(f"Range ({range_unit})")
    ax_img.set_ylabel("Azimuth (deg)")
    ax_img.set_title("Click a pixel below (background = time-mean). "
                      f"{len(epoch_s)} frames loaded.")
    fig.colorbar(im, ax=ax_img, label="mean SNR (dB)" if has_snr else f"mean moment ({moment_unit})")

    # Precompute vertical marks every 4 hours of the day (00, 04, 08, 12, 16, 20)
    # across the full span of the loaded data. These are drawn in whatever
    # timezone the epoch seconds were interpreted as at build time (check
    # d["timezone_assumed"] if unsure) - default was UTC.
    day0 = times.min().replace(hour=0, minute=0, second=0, microsecond=0)
    four_hour_marks = []
    t = day0
    while t <= times.max():
        if t >= times.min():
            four_hour_marks.append(t)
        t += datetime.timedelta(hours=4)
    tz_note = str(d["timezone_assumed"]) if "timezone_assumed" in d else "unknown"
    print(f"4-hour gridlines drawn assuming timestamps are: {tz_note}")

    px_marker, = ax_img.plot([], [], "r+", markersize=14, markeredgewidth=2)

    # Create the twin axis ONCE, outside the click handler. Calling ax_ts.twinx()
    # inside on_click would stack a new overlapping axis on every click, leaving
    # the previous click's line still drawn underneath (data would appear to
    # never be removed). We reuse the same pair of axes and clear both each click.
    ax_mom = ax_ts.twinx() if (has_snr and has_mom) else None
    target_ax = ax_mom if ax_mom is not None else ax_ts  # axis the moment/red trace lives on

    slider = None
    if show_slider:
        snr_min = float(np.nanmin(d["snr_db"]))
        snr_max = float(np.nanmax(d["snr_db"]))
        slider = Slider(slider_ax, "Min SNR (dB) for + markers", snr_min, snr_max,
                         valinit=snr_min)

    radio = None
    if has_mom:
        radio_ax = fig.add_axes([0.015, 0.45, 0.12, 0.1])
        radio_ax.set_title("Doppler trace", fontsize=8)
        radio = RadioButtons(radio_ax, ("Show", "Hide"), active=0)

    state = {"az_idx": None, "range_idx": None, "overlay": None,
             "mom_line": None, "show_trace": True}

    def update_overlay():
        if state["overlay"] is not None:
            state["overlay"].remove()
            state["overlay"] = None
        if slider is None or state["az_idx"] is None:
            fig.canvas.draw_idle()
            return
        az_idx, range_idx = state["az_idx"], state["range_idx"]
        snr_vals = d["snr_db"][:, az_idx, range_idx]
        mom_vals = d["moment"][:, az_idx, range_idx]
        mask = snr_vals >= slider.val
        if mask.any():
            overlay, = target_ax.plot(times[mask], mom_vals[mask], "+", color="red",
                                       markersize=10, markeredgewidth=2, linestyle="None",
                                       label=f"SNR >= {slider.val:.1f} dB")
            state["overlay"] = overlay
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None:
            return
        range_idx = int(np.argmin(np.abs(range_axis - event.xdata)))
        az_idx = int(np.argmin(np.abs(bin_centers_deg - event.ydata)))
        state["az_idx"], state["range_idx"] = az_idx, range_idx
        state["overlay"] = None  # axes are about to be cleared below
        px_marker.set_data([range_axis[range_idx]], [bin_centers_deg[az_idx]])

        ax_ts.clear()
        if ax_mom is not None:
            ax_mom.clear()

        for m in four_hour_marks:
            ax_ts.axvline(m, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

        if has_snr:
            ax_ts.plot(times, d["snr_db"][:, az_idx, range_idx], color="tab:blue", label="SNR (dB)")
            ax_ts.set_ylabel("SNR (dB)", color="tab:blue")
        if has_mom:
            mom_line, = target_ax.plot(times, d["moment"][:, az_idx, range_idx], color="tab:red",
                                        label=f"Moment ({moment_unit})")
            mom_line.set_visible(state["show_trace"])
            state["mom_line"] = mom_line
            target_ax.set_ylabel(f"Doppler moment ({moment_unit})", color="tab:red")
            target_ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                               label="Zero Doppler")

        ax_ts.set_xlabel("Time (UTC, per filename-timestamp assumption)")
        ax_ts.set_title(f"az={bin_centers_deg[az_idx]:.1f} deg, "
                         f"range={range_axis[range_idx]:.2f} {range_unit}")
        fig.autofmt_xdate()

        update_overlay()

    fig.canvas.mpl_connect("button_press_event", on_click)
    if slider is not None:
        slider.on_changed(lambda val: update_overlay())
    if radio is not None:
        def on_radio(label):
            state["show_trace"] = (label == "Show")
            if state["mom_line"] is not None:
                state["mom_line"].set_visible(state["show_trace"])
                fig.canvas.draw_idle()
        radio.on_clicked(on_radio)

    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Parse all npz frames into one time-series file")
    b.add_argument("--input_dir", required=True)
    b.add_argument("--output_file", required=True)
    b.add_argument("--pattern", default="*.npz")
    b.add_argument("--source", default="combined",
                   choices=["combined", "antenna_0", "antenna_1", "antenna_2", "antenna_3"])
    b.add_argument("--fields", nargs="+", choices=["snr", "moment"], default=["snr", "moment"])
    b.add_argument("--range_bin_width_m", type=float, default=None)
    b.add_argument("--center_freq_hz", type=float, default=None)
    b.add_argument("--timezone", choices=["UTC", "local"], default="UTC",
                   help="How to interpret the filename timestamp (default UTC - confirm this!)")
    b.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("view", help="Interactively pick a pixel and plot its time series")
    v.add_argument("--timeseries_file", required=True)
    v.set_defaults(func=cmd_view)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
