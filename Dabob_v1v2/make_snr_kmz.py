#!/usr/bin/env python3
"""
make_snr_kmz.py -- Google Earth overlay of combined-antenna SNR, for checking
the heading offset against the shoreline.

Writes one KMZ with two ground-overlay layers you can toggle:
    "SNR, true = radar az + 21"   and   "SNR, true = radar az - 21"
(plus placemarks for the radar and the PISCES2 buoy). The right sign is the
one whose bright returns line up with the shoreline.

The image is rendered on a regular lat/lon grid (what Google Earth's
LatLonBox expects), alpha 0.5 where there is data, transparent elsewhere.
SNR is the median over the selected files (dB), so persistent features
(land, shoreline) stand out.

Usage:
  python3 make_snr_kmz.py --files 'data/radar_products/*_combined.npz' --max-files 50 \
      --out snr_overlay.kmz

ASSUMPTIONS -- flagged: azimuth bin centres from bin_centers_deg in the file;
slant range r = i * 4.8 m (bin 0 at 0 m); flat-earth local tangent plane.
"""

import argparse
import glob
import io
import zipfile

import numpy as np

R_EARTH = 6.371e6


def load_median_snr(files, key):
    stack = []
    az = None
    for f in files:
        with np.load(f) as d:
            stack.append(np.asarray(d[key], np.float32))
            if az is None and "bin_centers_deg" in d.files:
                az = np.asarray(d["bin_centers_deg"], float)
    snr = np.nanmedian(np.stack(stack), axis=0)
    if az is None:
        az = np.arange(snr.shape[0]) + 0.5
    return snr, az


def render(snr, az_deg, args, offset_deg, mirror=False):
    n_az, n_r = snr.shape
    rmax = args.dr_m * (n_r - 1)
    gmax = np.sqrt(max(rmax ** 2 - args.height_m ** 2, 0))
    dlat = np.degrees(gmax / R_EARTH)
    dlon = dlat / np.cos(np.radians(args.radar_lat))
    n = args.pixels
    lats = np.linspace(args.radar_lat + dlat, args.radar_lat - dlat, n)      # top row = north
    lons = np.linspace(args.radar_lon - dlon, args.radar_lon + dlon, n)
    LON, LAT = np.meshgrid(lons, lats)
    north = np.radians(LAT - args.radar_lat) * R_EARTH
    east = np.radians(LON - args.radar_lon) * R_EARTH * np.cos(np.radians(args.radar_lat))
    g = np.hypot(east, north)
    az_true = np.degrees(np.arctan2(east, north)) % 360
    # normal: true = radar + offset;  mirrored (radar azimuth counter-clockwise):
    # true = offset - radar
    az_radar = ((offset_deg - az_true) if mirror else (az_true - offset_deg)) % 360
    daz = 360.0 / n_az
    ia = np.floor((az_radar - (az_deg[0] - daz / 2)) / daz).astype(int) % n_az
    r = np.sqrt(g ** 2 + args.height_m ** 2)
    ir = np.rint(r / args.dr_m).astype(int)
    inside = (g <= gmax) & (r > args.height_m)
    val = np.full(g.shape, np.nan, np.float32)
    val[inside] = snr[ia[inside], np.clip(ir[inside], 0, n_r - 1)]
    box = dict(north=args.radar_lat + dlat, south=args.radar_lat - dlat,
               east=args.radar_lon + dlon, west=args.radar_lon - dlon)
    return val, box


def to_png(val, vmin, vmax, alpha, cmap_name, alpha_min=None, alpha_power=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    x = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(np.nan_to_num(x))
    if alpha_min is None:
        a = np.full(val.shape, alpha)                       # constant alpha
    else:
        # alpha rises with SNR: alpha_min at vmin -> alpha at vmax, shaped by alpha_power
        a = alpha_min + (alpha - alpha_min) * np.nan_to_num(x) ** alpha_power
    rgba[..., 3] = np.where(np.isfinite(val), a, 0.0)
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return buf.getvalue()


def colorbar_png(vmin, vmax, cmap_name, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.3, 4))
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, cax=ax, label=label)
    fig.subplots_adjust(left=0.05, right=0.35, top=0.97, bottom=0.03)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=False)
    plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", required=True, help="glob of combined npz files")
    ap.add_argument("--key", default="combined_snr_db")
    ap.add_argument("--max-files", type=int, default=50, help="evenly spaced subset")
    ap.add_argument("--offsets", type=float, nargs="+", default=[21.0, -21.0])
    ap.add_argument("--mirror", action="store_true",
                    help="also add mirrored layers (radar azimuth counter-clockwise)")
    ap.add_argument("--radar-lat", type=float, default=47.709487)
    ap.add_argument("--radar-lon", type=float, default=-122.823652)
    ap.add_argument("--buoy-lat", type=float, default=47.692312)
    ap.add_argument("--buoy-lon", type=float, default=-122.865058)
    ap.add_argument("--height-m", type=float, default=66.0)
    ap.add_argument("--dr-m", type=float, default=4.8)
    ap.add_argument("--pixels", type=int, default=1200)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="constant alpha, or the maximum alpha (at vmax) when --alpha-min is given")
    ap.add_argument("--alpha-min", type=float, default=None,
                    help="make alpha vary with SNR: this alpha at vmin, --alpha at vmax (e.g. 0)")
    ap.add_argument("--alpha-power", type=float, default=1.0,
                    help=">1 keeps weak returns more transparent (e.g. 2)")
    ap.add_argument("--vmin", type=float, default=None, help="default: 5th percentile")
    ap.add_argument("--vmax", type=float, default=None, help="default: 99th percentile")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--out", default="snr_overlay.kmz")
    args = ap.parse_args()

    files = sorted(glob.glob(args.files))
    if not files:
        raise SystemExit(f"no files match {args.files}")
    if len(files) > args.max_files:
        files = [files[i] for i in np.linspace(0, len(files) - 1, args.max_files).astype(int)]
    print(f"median of {len(files)} files, key '{args.key}'")
    snr, az = load_median_snr(files, args.key)
    fin = snr[np.isfinite(snr)]
    vmin = args.vmin if args.vmin is not None else float(np.percentile(fin, 5))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(fin, 99))
    print(f"colour range {vmin:.1f} .. {vmax:.1f} dB")

    overlays, images = [], {}
    layers = [(o, False) for o in args.offsets] + ([(o, True) for o in args.offsets] if args.mirror else [])
    for k, (off, mir) in enumerate(layers):
        val, box = render(snr, az, args, off, mir)
        name = (f"snr_off{off:+g}" + ("_mirror" if mir else "") + ".png").replace("+", "p").replace("-", "m")
        images[name] = to_png(val, vmin, vmax, args.alpha, args.cmap,
                              args.alpha_min, args.alpha_power)
        label = (f"SNR, true = {off:g} - radar az (mirrored)" if mir else
                 f"SNR, true = radar az {'+' if off >= 0 else '-'} {abs(off):g}")
        overlays.append(f"""
    <GroundOverlay>
      <name>{label}</name>
      <visibility>{1 if k == 0 else 0}</visibility>
      <Icon><href>{name}</href></Icon>
      <LatLonBox><north>{box['north']:.7f}</north><south>{box['south']:.7f}</south>
        <east>{box['east']:.7f}</east><west>{box['west']:.7f}</west></LatLonBox>
    </GroundOverlay>""")
    images["colorbar.png"] = colorbar_png(vmin, vmax, args.cmap, f"median {args.key} (dB)")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Dabob radar SNR overlay (heading check)</name>
    <description>Median {args.key} over {len(files)} windows. Toggle the layers; the correct heading sign is the one whose bright returns follow the shoreline.</description>
    {''.join(overlays)}
    <ScreenOverlay>
      <name>colour bar</name>
      <Icon><href>colorbar.png</href></Icon>
      <overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>
      <screenXY x="0.01" y="0.95" xunits="fraction" yunits="fraction"/>
      <size x="0" y="0" xunits="pixels" yunits="pixels"/>
    </ScreenOverlay>
    <Placemark><name>radar</name><Point><coordinates>{args.radar_lon},{args.radar_lat},0</coordinates></Point></Placemark>
    <Placemark><name>PISCES2 buoy</name><Point><coordinates>{args.buoy_lon},{args.buoy_lat},0</coordinates></Point></Placemark>
  </Document>
</kml>
"""
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        for n_, b in images.items():
            z.writestr(n_, b)
    print(f"wrote {args.out}  ({len(layers)} layers)")


if __name__ == "__main__":
    main()
