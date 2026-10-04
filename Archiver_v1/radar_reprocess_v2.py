#!/usr/bin/env python3
"""
Offline reprocessing of raw archive windows (<ts>.bin) into v2 products.

For each .bin (one archive window, packets concatenated exactly as received,
PACKET_SIZE bytes each, no framing):
  1. assemble 64-pulse looks with the SAME rules as RadarBlockParser
     (counter step 16 mod 65536, same antenna; a break drops the partial
     block; a new block starts unconditionally after a completed one;
     look angle / timestamp from the first packet, pa_enable from the packet
     that completes the block -- as radar_archive_service.py records it)
  2. run doppler_processor_v2.DopplerProcessor on each look
  3. bin by look angle per antenna, last look into a bin wins (as live)
  4. write <ts>_products_v2.npz with the live per-antenna keys (v1 fields,
     recomputed -- should match the live _products.npz) PLUS per-antenna v2
     fields, processor parameters, and packet/look counts
  5. write <ts>_products_v2_combined.npz via radar_products_combiner_v2

Block assembly is vectorised over packet headers (memory-mapped file), so a
multi-GB .bin is never loaded whole. Looks are processed in batches.

Sources:
  local:  --bin-dir DIR
  remote: --remote-host HOST --remote-dir DIR   (scp one .bin at a time into
          --local-scratch, delete after processing -- never more than one
          large file locally). Because .bin files are large, running this
          script on the machine that holds them and pulling back only the
          small npz is likely much faster than pulling the .bin files.

ASSUMPTIONS -- flagged:
  - .bin contents are whole packets back to back; a trailing partial packet
    (e.g. file truncated mid-write) is ignored.
  - Packets whose size != PACKET_SIZE can't be identified in a raw
    concatenation; if the live service ever received odd-sized datagrams,
    record boundaries would shift. --check-v1 detects that (v1 fields would
    stop matching the live products).
  - "Already done" = <ts>_products_v2_combined.npz exists in --out-dir.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from radar_packet_parser import (ANTENNA_OFFSETS_DEG, HEADER_SIZE, NUM_RANGE_BINS,
                                 PACKET_SIZE, RadarBlockParser)
from doppler_processor_v2 import DopplerProcessor, PROCESSOR_VERSION
from radar_products_combiner_v2 import add_combined_v2

BLOCK = RadarBlockParser.BLOCK_SIZE
STEP = RadarBlockParser.COUNTER_STEP
WRAP = RadarBlockParser.COUNTER_WRAP

PKT_DTYPE = np.dtype([("counter", "<u8"), ("timestamp_us", "<u8"), ("antenna", "<u2"),
                      ("pa_enable", "<u2"), ("reserved", "<u4"), ("encoder", "<f8"),
                      ("samples", "<i4", (2 * NUM_RANGE_BINS,))])
assert PKT_DTYPE.itemsize == PACKET_SIZE and HEADER_SIZE == 32

V2_FIELDS = ("noise_floor_v2", "moment1_pk_hz", "moment2_pk_hz", "n_bins_pk",
             "detected", "snr_band_db", "snr_pk_db")
V1_FIELDS = ("power_linear", "power_db", "snr_db", "noise_floor", "moment1_hz", "moment2_hz")


# ----------------------------------------------------------------------------
# block assembly
# ----------------------------------------------------------------------------

def find_blocks(counter, antenna, block=BLOCK):
    """Start indices of complete looks, replicating RadarBlockParser.feed().

    brk[i] is True when packet i does not continue packet i-1. A block
    started at s completes iff no break occurs at s+1 .. s+block-1; if one
    does at j, the partial block is dropped and a new one starts at j. After a
    completed block the next packet starts a new block regardless of
    continuity.
    """
    n = len(counter)
    if n < block:
        return np.empty(0, np.int64), 0
    c = counter.astype(np.int64) % WRAP
    brk = np.ones(n, bool)
    brk[1:] = (c[1:] != (c[:-1] + STEP) % WRAP) | (antenna[1:] != antenna[:-1])
    brk_idx = np.flatnonzero(brk)
    starts = []
    dropped = 0
    s = 0
    while s + block <= n:
        # first break strictly after s
        k = np.searchsorted(brk_idx, s, side="right")
        j = brk_idx[k] if k < len(brk_idx) else n
        if j < s + block:
            dropped += 1
            s = j
        else:
            starts.append(s)
            s += block
    return np.asarray(starts, np.int64), dropped


# ----------------------------------------------------------------------------
# one .bin -> products
# ----------------------------------------------------------------------------

def process_bin(bin_path, proc: DopplerProcessor, num_bins=360, antennas=(0, 1, 2, 3),
                batch_looks=32, verbose=True):
    size = os.path.getsize(bin_path)
    npk = size // PACKET_SIZE
    if npk == 0:
        raise ValueError("no complete packets")
    mm = np.memmap(bin_path, dtype=PKT_DTYPE, mode="r", shape=(npk,))
    counter = np.asarray(mm["counter"])
    antenna = np.asarray(mm["antenna"])
    starts, dropped = find_blocks(counter, antenna)

    bw = 360.0 / num_bins
    nr = NUM_RANGE_BINS
    f64 = lambda: np.full((num_bins, nr), np.nan)
    f32 = lambda: np.full((num_bins, nr), np.nan, np.float32)
    bins = {a: dict(**{k: f64() for k in V1_FIELDS}, **{k: f32() for k in V2_FIELDS},
                    pa_enable=np.full(num_bins, np.nan), timestamp_us=np.full(num_bins, np.nan),
                    hit_count=np.zeros(num_bins, np.int64)) for a in antennas}
    unknown_ant = 0
    t0 = time.time()

    for b0 in range(0, len(starts), batch_looks):
        st = starts[b0:b0 + batch_looks]
        # (n_looks, 64, 2048) -> complex (n_looks*1024, 64), range-major per look
        idx = st[:, None] + np.arange(BLOCK)[None, :]
        samp = np.asarray(mm["samples"][idx.ravel()]).reshape(len(st), BLOCK, 2 * nr)
        cplx = samp[..., :nr].astype(np.float64) + 1j * samp[..., nr:].astype(np.float64)
        data = np.transpose(cplx, (0, 2, 1)).reshape(len(st) * nr, BLOCK)
        # rows are independent in the processor, so a batch can be one call
        prod = proc.process({"look_angle_deg": 0.0, "timestamp_us": 0, "complex_data": data})
        for i, s in enumerate(st):
            ant = int(antenna[s])
            if ant not in ANTENNA_OFFSETS_DEG or ant not in bins:
                unknown_ant += 1
                continue
            ang = (float(mm["encoder"][s]) + ANTENNA_OFFSETS_DEG[ant]) % 360.0
            b = min(int((ang % 360.0) // bw), num_bins - 1)
            sl = slice(i * nr, (i + 1) * nr)
            arc = bins[ant]
            arc["power_linear"][b] = prod.power_linear[sl]
            arc["power_db"][b] = prod.power_db[sl]
            arc["snr_db"][b] = prod.snr_db[sl]
            arc["noise_floor"][b] = prod.noise_floor[sl]
            arc["moment1_hz"][b] = prod.moment1_hz[sl]
            arc["moment2_hz"][b] = prod.moment2_hz[sl]
            arc["noise_floor_v2"][b] = prod.noise_floor_v2[sl]
            arc["moment1_pk_hz"][b] = prod.moment1_pk_hz[sl]
            arc["moment2_pk_hz"][b] = prod.moment2_pk_hz[sl]
            arc["n_bins_pk"][b] = prod.n_bins_pk[sl]
            arc["detected"][b] = prod.detected[sl]
            arc["snr_band_db"][b] = prod.snr_band_db[sl]
            arc["snr_pk_db"][b] = prod.snr_pk_db[sl]
            arc["timestamp_us"][b] = float(mm["timestamp_us"][s])
            arc["pa_enable"][b] = float(mm["pa_enable"][s + BLOCK - 1])  # completing packet
            arc["hit_count"][b] += 1
    del mm

    stats = dict(n_packets=npk, trailing_bytes=size - npk * PACKET_SIZE, n_looks=len(starts),
                 dropped_blocks=dropped, unknown_antenna_looks=unknown_ant,
                 seconds=round(time.time() - t0, 1))
    if verbose:
        print(f"    {npk} packets, {len(starts)} looks, {dropped} dropped partial blocks, "
              f"{stats['seconds']} s")
    return bins, stats


def save_products(path, bins, stats, proc, num_bins, src_name):
    bw = 360.0 / num_bins
    out = {"num_bins": num_bins, "bin_width_deg": bw,
           "bin_centers_deg": (np.arange(num_bins) + 0.5) * bw,
           "antenna_indices": np.array(sorted(bins)),
           "processor_version": PROCESSOR_VERSION,
           "source_bin": src_name}
    for k, v in proc.params().items():
        out[f"param_{k}"] = v
    for k, v in stats.items():
        out[f"stat_{k}"] = v
    for a, arc in bins.items():
        for k, v in arc.items():
            out[f"antenna_{a}_{k}"] = v
    np.savez_compressed(path, **out)


def check_v1(v2_path, live_path):
    """Compare recomputed v1 fields to the live _products.npz."""
    a, b = np.load(v2_path), np.load(live_path)
    worst = 0.0
    msgs = []
    for k in b.files:
        if not k.startswith("antenna_") or not any(k.endswith(f) for f in ("moment1_hz", "snr_db", "hit_count")):
            continue
        if k not in a.files:
            msgs.append(f"{k} missing")
            continue
        x, y = a[k].astype(float), b[k].astype(float)
        both = np.isfinite(x) & np.isfinite(y)
        nan_mismatch = int((np.isfinite(x) != np.isfinite(y)).sum())
        d = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
        worst = max(worst, d)
        if d > 1e-6 or nan_mismatch:
            msgs.append(f"{k}: max|diff|={d:.3g}, finite-mismatch={nan_mismatch}")
    return worst, msgs


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

TS_RE = re.compile(r"^(\d{14})\.bin$")


def list_local(bin_dir):
    return sorted(p.name for p in Path(bin_dir).glob("*.bin") if TS_RE.match(p.name))


def list_remote(host, remote_dir):
    remote_dir = remote_dir.rstrip("/") + "/"
    cmd = ["ssh", host, f"find {remote_dir} -maxdepth 1 -type f -name '*.bin' -printf '%f\\n'"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    files = [l for l in r.stdout.splitlines() if TS_RE.match(l.strip())]
    if r.returncode != 0:
        if files:
            print(f"  [warning] remote find exited {r.returncode} (continuing with {len(files)} files): "
                  f"{r.stderr.strip()}")
        else:
            raise RuntimeError(f"ssh listing failed (exit {r.returncode}): {r.stderr.strip()}")
    return sorted(files)


def _one(work):
    i, n, name, job = work
    ts = TS_RE.match(name).group(1)
    p = job["proc"]
    proc = DopplerProcessor(fs_hz=p["fs_hz"], tail_fraction=p["tail_fraction"], window=p["window"],
                            region_k=p["region_k"], detect_k=p["detect_k"])
    out_dir = Path(job["out_dir"])
    log = [f"[{i}/{n}] {name}"]
    local = None
    try:
        if job["remote"]:
            local = Path(job["scratch"]) / name
            r = subprocess.run(["scp", "-q", f"{job['host']}:{job['rdir'].rstrip('/')}/{name}", str(local)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"scp failed: {r.stderr.strip()}")
        else:
            local = Path(job["bin_dir"]) / name
        bins, stats = process_bin(local, proc, num_bins=job["num_bins"],
                                  batch_looks=job["batch_looks"], verbose=False)
        log.append(f"    {stats['n_packets']} packets, {stats['n_looks']} looks, "
                   f"{stats['dropped_blocks']} dropped partial blocks, {stats['seconds']} s")
        prod_path = out_dir / f"{ts}_products_v2.npz"
        save_products(prod_path, bins, stats, proc, job["num_bins"], name)
        if job["check_v1"]:
            live = Path(job["check_v1"]) / f"{ts}_products.npz"
            if not live.exists():
                live = Path(job["check_v1"]) / f"{ts}_products_combined.npz"
            if live.exists():
                worst, msgs = check_v1(prod_path, live)
                log.append(f"    v1 check vs {live.name}: " + ("MATCH" if not msgs else "; ".join(msgs[:4])))
            else:
                log.append(f"    v1 check: no live products for {ts}")
        if job["combined"]:
            add_combined_v2(str(prod_path), str(out_dir / f"{ts}_products_v2_combined.npz"),
                            float32=job["float32"], slim=job["slim"])
            prod_path.unlink()   # the combined file contains all per-antenna fields too
        err = None
    except Exception as e:
        log.append(f"    FAILED: {e}")
        err = str(e)
    finally:
        if job["remote"] and local is not None and local.exists() and not job["keep_bin"]:
            local.unlink()
    print("\n".join(log), flush=True)
    return name, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--bin-dir", help="Local directory of <ts>.bin files")
    src.add_argument("--remote-host", help="ssh host alias (e.g. radar-jump)")
    src.add_argument("--remote-dir", help="Remote directory of .bin files")
    src.add_argument("--local-scratch", default="/tmp/radar_bin_scratch")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--since", help="Only timestamps >= this (YYYYmmddHHMMSS)")
    ap.add_argument("--until", help="Only timestamps <= this (YYYYmmddHHMMSS)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files")
    ap.add_argument("--every", type=int, default=1, help="Every Nth file (after since/until)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-v1", metavar="LIVE_PRODUCTS_DIR",
                    help="Compare recomputed v1 fields with <ts>_products.npz found here")
    ap.add_argument("--no-combined", action="store_true", help="Skip writing _combined.npz")
    ap.add_argument("--keep-bin", action="store_true", help="Remote mode: keep scratch copy")
    g = ap.add_argument_group("processor")
    g.add_argument("--fs-hz", type=float, default=1562.5)
    g.add_argument("--tail-fraction", type=float, default=0.25)
    g.add_argument("--window", default="none", choices=["none", "hann"])
    g.add_argument("--region-k", type=float, default=3.0)
    g.add_argument("--detect-k", type=float, default=10.0)
    g.add_argument("--num-bins", type=int, default=360)
    g.add_argument("--batch-looks", type=int, default=32)
    ap.add_argument("--float32", action="store_true",
                    help="Store 2-D float fields as float32 in the combined file (~half the size)")
    ap.add_argument("--slim", action="store_true",
                    help="Combined file keeps only combined_* fields + metadata (~10x smaller)")
    ap.add_argument("--min-age-min", type=float, default=15.0,
                    help="Local mode: skip .bin files modified within this many minutes "
                         "(the live service may still be writing them)")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel files (local mode only)")
    args = ap.parse_args()

    remote = bool(args.remote_host)
    if remote == bool(args.bin_dir):
        sys.exit("Give exactly one of --bin-dir or --remote-host/--remote-dir")
    if remote and not args.remote_dir:
        sys.exit("--remote-host needs --remote-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list_remote(args.remote_host, args.remote_dir) if remote else list_local(args.bin_dir)
    if not remote and args.min_age_min > 0:
        now = time.time()
        young = [n for n in names if now - os.path.getmtime(Path(args.bin_dir) / n) < 60 * args.min_age_min]
        if young:
            print(f"  skipping {len(young)} .bin file(s) modified in the last {args.min_age_min:g} min: "
                  f"{', '.join(young[:3])}")
        names = [n for n in names if n not in young]
    ts_of = lambda n: TS_RE.match(n).group(1)
    if args.since:
        names = [n for n in names if ts_of(n) >= args.since]
    if args.until:
        names = [n for n in names if ts_of(n) <= args.until]
    names = names[::max(args.every, 1)]
    done = {p.name[:14] for p in out_dir.glob("*_products_v2_combined.npz")} if not args.no_combined \
        else {p.name[:14] for p in out_dir.glob("*_products_v2.npz")}
    todo = [n for n in names if ts_of(n) not in done]
    if args.limit:
        todo = todo[:args.limit]
    n_done = sum(ts_of(n) in done for n in names)
    print(f"{len(names)} .bin files selected, {n_done} already done, {len(todo)} to process")
    if args.dry_run:
        for n in todo:
            print("   ", n)
        return

    proc = DopplerProcessor(fs_hz=args.fs_hz, tail_fraction=args.tail_fraction, window=args.window,
                            region_k=args.region_k, detect_k=args.detect_k)
    print(f"processor v{PROCESSOR_VERSION}: {proc.params()}")
    scratch = Path(args.local_scratch)
    if remote:
        scratch.mkdir(parents=True, exist_ok=True)

    job = dict(remote=remote, host=args.remote_host, rdir=args.remote_dir, scratch=str(scratch),
               bin_dir=args.bin_dir, out_dir=str(out_dir), check_v1=args.check_v1,
               combined=not args.no_combined, float32=args.float32, slim=args.slim, keep_bin=args.keep_bin, num_bins=args.num_bins,
               batch_looks=args.batch_looks, proc=proc.params())
    work = [(i, len(todo), name, job) for i, name in enumerate(todo, 1)]
    if args.jobs > 1 and not remote:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.map(_one, work, chunksize=1)
    else:
        if args.jobs > 1:
            print("  --jobs ignored in remote mode (one .bin locally at a time)")
        results = [_one(w) for w in work]
    ok = sum(1 for r in results if r[1] is None)
    failed = [r for r in results if r[1] is not None]
    print(f"\nDone: {ok} ok, {len(failed)} failed")
    for n, e in failed:
        print(f"  {n}: {e}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
