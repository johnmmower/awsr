#!/usr/bin/env python3
"""
Sync + combine pipeline, same shape as the earlier .bin-based version,
now aimed at the small _products.npz files radar_archive_service.py
already produces on the NUC -- no raw .bin transfer or reprocessing.

  1. ssh to the remote host and list *_products.npz files in the remote
     share dir.
  2. Diff against which ones already have a corresponding
     "<stem>_combined.npz" in the local output dir.
  3. For each MISSING products.npz: scp it down to a local scratch dir,
     run radar_products_combiner.add_combined_products() over it (adds
     combined_* fields averaged across whichever antennas hit each bin),
     save the result to the output dir, then delete the local scratch
     copy of the original.

ASSUMPTIONS -- flagging explicitly:
  - Passwordless/key-based ssh to the host alias already works.
  - Transfer uses `scp`. Swap _scp_download() if you're on rsync instead.
  - "Already processed" is decided by filename only (<stem>_products.npz
    -> <stem>_products_combined.npz), no size/mtime/checksum check
    against the remote copy.
  - Requires radar_products_combiner.py importable (same directory or on
    PYTHONPATH).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from radar_products_combiner import add_combined_products


def list_remote_products_files(host: str, remote_dir: str) -> list:
    """ssh to `host` and list *_products.npz filenames in remote_dir.

    `find` can exit non-zero while still having printed valid results on
    stdout (e.g. a symlink loop on one unrelated entry) -- that's a
    partial-failure warning, not a reason to discard everything it did
    find. Only raise if stdout came back empty.
    """
    remote_dir = remote_dir.rstrip("/") + "/"
    cmd = ["ssh", host,
           f"find {remote_dir} -maxdepth 1 -type f -name '*_products.npz' -printf '%f\\n'"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    files = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0:
        if files:
            print(f"  [warning] remote find exited {result.returncode} "
                  f"(partial failure, continuing with {len(files)} files found): "
                  f"{result.stderr.strip()}")
        else:
            raise RuntimeError(
                f"ssh file listing failed (exit {result.returncode}), no files "
                f"returned: {result.stderr.strip()}"
            )
    return files


def already_processed(output_dir: Path) -> set:
    """Return the set of <stem>_products.npz stems that already have a
    _combined.npz output (e.g. '20260913143000_products' is done if
    '20260913143000_products_combined.npz' exists)."""
    done = set()
    for combined_path in output_dir.glob("*_combined.npz"):
        stem = combined_path.name[:-len("_combined.npz")]
        done.add(stem)
    return done


def scp_download(host: str, remote_dir: str, filename: str, local_dir: Path) -> Path:
    remote_path = f"{host}:{remote_dir.rstrip('/')}/{filename}"
    local_path = local_dir / filename
    cmd = ["scp", "-q", remote_path, str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scp failed for {filename} (exit {result.returncode}): {result.stderr.strip()}")
    return local_path


def process_one(npz_path: Path, output_dir: Path):
    stem = npz_path.stem  # "<timestamp>_products" (strips .npz only)
    out_path = output_dir / f"{stem}_combined.npz"
    add_combined_products(str(npz_path), output_path=str(out_path))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="ssh host/alias, e.g. radar-jump")
    ap.add_argument("--remote-dir", required=True, help="Remote directory containing _products.npz files")
    ap.add_argument("--local-scratch", default="/tmp/radar_incoming",
                     help="Local scratch dir for the products.npz during processing (deleted after each file)")
    ap.add_argument("--output-dir", required=True, help="Local directory for _combined.npz outputs")
    ap.add_argument("--keep-npz", action="store_true",
                     help="Don't delete the local scratch copy after processing (default: delete)")
    ap.add_argument("--dry-run", action="store_true",
                     help="List what would be downloaded/processed without doing it")
    args = ap.parse_args()

    scratch_dir = Path(args.local_scratch)
    output_dir = Path(args.output_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing remote _products.npz files on {args.host}:{args.remote_dir} ...")
    remote_files = list_remote_products_files(args.host, args.remote_dir)
    print(f"  {len(remote_files)} _products.npz files found remotely")

    done_stems = already_processed(output_dir)
    print(f"  {len(done_stems)} already processed locally (found in {output_dir})")

    missing = [f for f in remote_files if Path(f).stem not in done_stems]
    print(f"  {len(missing)} missing -> {'would process' if args.dry_run else 'processing'}")

    if args.dry_run:
        for f in missing:
            print(f"    {f}")
        return

    succeeded, failed = [], []
    for i, filename in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {filename} ...", end=" ", flush=True)
        try:
            local_npz = scp_download(args.host, args.remote_dir, filename, scratch_dir)
            out_path = process_one(local_npz, output_dir)
            print(f"OK -> {out_path.name}")
            succeeded.append(filename)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((filename, str(e)))
        finally:
            if not args.keep_npz:
                local_npz_path = scratch_dir / filename
                if local_npz_path.exists():
                    local_npz_path.unlink()

    print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed files:")
        for filename, err in failed:
            print(f"  {filename}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
