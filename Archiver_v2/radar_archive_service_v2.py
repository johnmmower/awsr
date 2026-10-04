#!/usr/bin/env python3
"""
Two-thread radar archive service -- v2.

Same as radar_archive_service.py (v1) except:
  - uses doppler_processor_v2 (v1 per-antenna fields unchanged, plus the v2
    fields: noise_floor_v2, moment1_pk_hz, moment2_pk_hz, n_bins_pk,
    detected, snr_band_db, snr_pk_db -- stored as float32)
  - product file is <ts>_products_v2.npz and also stores processor_version
    and param_* (the processor settings)
  - at window close it ALSO writes <ts>_products_v2_combined.npz: combined_*
    fields from radar_products_combiner_v2 (PA-off antennas excluded) plus
    1-D metadata only, float32 -- small, meant for syncing over the slow link.
    --no-full-products skips the large per-antenna file.
  - logs queue drops / parse errors / looks per window at each window close
    (v1 only printed drops on shutdown)
  - plot_polar_products is imported only when PNGs are rendered, so the
    service runs without it
  - default --output-dir /mnt/share/v2
  - --start-time lets you anchor the cycle phase (v1 always anchored to
    process start, so each restart shifted the window times)

Original v1 description follows.

Two-thread radar archive service.

Thread 1 (listener): binds a UDP socket, does nothing but recvfrom() in a
tight loop and pushes (raw_bytes, recv_time) onto a queue. Kept minimal so
it can keep draining the OS socket buffer as fast as possible; UDP will be
silently dropped by the kernel if this thread falls behind, independent of
anything downstream.

Thread 2 (worker): pops (raw_bytes, recv_time) off the queue. If currently
inside an active archive window:
  - appends the raw payload bytes to a per-window .bin file (payload only,
    exactly as received off the socket -- no added framing/headers)
  - parses the packet, feeds it into a per-antenna 64-pulse block
    accumulator, and on every completed block (RadarBlockParser guarantees
    all 64 packets share one antenna_index and are counter-contiguous)
    runs the Doppler processor and bins the resulting product by angle +
    antenna, OVERWRITING whatever was previously in that bin (last look
    into a bin wins -- matches "just overwrite" requirement, no averaging)
When the active window ends (transition from archiving -> not archiving),
the worker finalizes: closes the raw .bin file, saves the final,
overwritten binned-product state to a companion file, and (by default)
renders that file's polar PNG images via plot_polar_products.py into a
separate directory (default: <output_dir>/pngs).
Outside an active window, incoming data is simply dropped (not written,
not processed) until the next window starts.

=== NAMING -- FLAGGING A LIKELY TYPO ===
The requested raw-file name format was "%Y%d%H%S.bin" -- note this omits
month and minute entirely (Year, Day-of-month, Hour, Second only). Two
different runs on the same day-of-month/hour/second in different months
(or even different hours' worth of same day/second combos across enough
time) would silently overwrite each other's archive. ASSUMING this was a
typo for a full timestamp and using "%Y%m%d%H%M%S" by default instead
(configurable via FILENAME_TIMESTAMP_FMT below) -- flagging this
explicitly rather than silently guessing, since a filename collision here
means silently losing archived data.

The product file uses the same timestamp base with a "_products.npz"
suffix, so the two files for a given window are easy to pair up by eye.

=== BINNING / SCHEDULING ===
  - num_bins angular bins spanning 0-360 degrees, bin i centered at
    (i + 0.5) * (360 / num_bins) degrees.
  - is_archiving(now): True for the first `archive_minutes` minutes of
    each `cycle_minutes`-minute cycle, anchored to `start_time` (defaults
    to construction time -- pass an explicit epoch time if you need
    cycles anchored to e.g. the top of the hour instead).
  - Antenna identity for binning comes from the packet that completes each
    64-pulse block (guaranteed single antenna per block by
    RadarBlockParser), since DopplerProduct itself doesn't carry it.
"""

import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from radar_packet_parser import RadarPacketParser, RadarBlockParser, RadarPacket
from doppler_processor_v2 import DopplerProcessor, DopplerProduct, PROCESSOR_VERSION
from radar_products_combiner_v2 import combine_v2

V2_FIELDS = ("noise_floor_v2", "moment1_pk_hz", "moment2_pk_hz", "n_bins_pk",
             "detected", "snr_band_db", "snr_pk_db")

FILENAME_TIMESTAMP_FMT = "%Y%m%d%H%M%S"  # see NAMING note above


@dataclass
class _AntennaBins:
    power_linear: np.ndarray
    power_db: np.ndarray
    snr_db: np.ndarray
    noise_floor: np.ndarray
    moment1_hz: np.ndarray
    moment2_hz: np.ndarray
    pa_enable: np.ndarray
    timestamp_us: np.ndarray
    hit_count: np.ndarray
    v2: dict = None


class RadarArchiveService:
    def __init__(self,
                 bind_ip: str = "0.0.0.0",
                 bind_port: int = 34952,
                 output_dir: str = ".",
                 cycle_minutes: float = 60.0,
                 archive_minutes: float = 5.0,
                 num_bins: int = 360,
                 antenna_indices=(0, 1, 2, 3),
                 fs_hz: float = 1562.5,
                 queue_maxsize: int = 200_000,
                 start_time: Optional[float] = None,
                 generate_pngs: bool = True,
                 png_output_dir: Optional[str] = None,
                 png_log_scale: bool = False,
                 png_range_resolution_m: Optional[float] = None,
                 processor_kwargs: Optional[dict] = None,
                 write_full_products: bool = True):
        """
        Args:
            bind_ip/bind_port: UDP socket to listen on.
            output_dir: directory the .bin / _products.npz files go in.
            cycle_minutes: length of the repeating schedule cycle.
            archive_minutes: how much of each cycle is actively archived
                (must be <= cycle_minutes).
            num_bins: number of angular bins spanning 0-360 degrees.
            antenna_indices: antennas to pre-allocate bin storage for.
            fs_hz: pulse rate passed through to DopplerProcessor.
            queue_maxsize: bounds the listener->worker queue. If the
                worker falls behind, new items are dropped (counted in
                dropped_queue_full) rather than blocking the listener --
                blocking the listener would risk the OS socket buffer
                overflowing instead, which is worse (silent kernel-level
                drops with no visibility at all).
            start_time: epoch seconds anchoring cycle phase. Defaults to
                time.time() at construction.
            generate_pngs: if True (default), automatically render polar
                PNG images (via plot_polar_products.render_npz_to_pngs)
                for every completed window, right after the products.npz
                is saved. Set False to skip PNG generation entirely.
            png_output_dir: directory for the PNGs. Defaults to
                f"{output_dir}/pngs" if not given -- kept separate from
                the raw .bin/.npz files by default since PNGs are a
                derived/disposable product, not the archive of record.
            png_log_scale: passed through to render_npz_to_pngs
                (log-scale coloring for power_linear/noise_floor).
            png_range_resolution_m: passed through to render_npz_to_pngs
                (labels the radial axis in meters instead of bin index).
        """
        if archive_minutes > cycle_minutes:
            raise ValueError("archive_minutes must be <= cycle_minutes")

        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.output_dir = output_dir
        self.cycle_seconds = cycle_minutes * 60.0
        self.archive_seconds = archive_minutes * 60.0
        self.num_bins = num_bins
        self.bin_width_deg = 360.0 / num_bins
        self.antenna_indices = list(antenna_indices)
        self.start_time = start_time if start_time is not None else time.time()

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()

        self._packet_parser = RadarPacketParser(strict=True)
        self._block_parser = RadarBlockParser(strict=True)
        self._doppler = DopplerProcessor(fs_hz=fs_hz, **(processor_kwargs or {}))
        self.write_full_products = write_full_products
        self._looks_this_window = 0
        self._drops_at_open = (0, 0)

        self._num_range_bins: Optional[int] = None
        self._bins: dict = {}

        self._raw_file = None
        self._window_active = False
        self._current_product_ts_str: Optional[str] = None
        self.dropped_queue_full = 0
        self.dropped_parse_errors = 0

        self.generate_pngs = generate_pngs
        self.png_output_dir = png_output_dir or os.path.join(self.output_dir, "pngs")
        self.png_log_scale = png_log_scale
        self.png_range_resolution_m = png_range_resolution_m

        os.makedirs(self.output_dir, exist_ok=True)
        if self.generate_pngs:
            os.makedirs(self.png_output_dir, exist_ok=True)

    # -- scheduling -------------------------------------------------------

    def is_archiving(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        elapsed = (now - self.start_time) % self.cycle_seconds
        return elapsed < self.archive_seconds

    def _bin_index(self, angle_deg: float) -> int:
        idx = int((angle_deg % 360.0) // self.bin_width_deg)
        return min(idx, self.num_bins - 1)

    def _bin_center_deg(self, idx: int) -> float:
        return (idx + 0.5) * self.bin_width_deg

    # -- bin storage --------------------------------------------------------

    def _allocate_bins(self, num_range_bins: int):
        self._num_range_bins = num_range_bins
        nan2d = lambda: np.full((self.num_bins, num_range_bins), np.nan)
        nan1d = lambda: np.full(self.num_bins, np.nan)
        self._bins = {
            ant: _AntennaBins(
                power_linear=nan2d(), power_db=nan2d(), snr_db=nan2d(),
                noise_floor=nan2d(), moment1_hz=nan2d(), moment2_hz=nan2d(),
                pa_enable=nan1d(), timestamp_us=nan1d(),
                hit_count=np.zeros(self.num_bins, dtype=np.int64),
                v2={k: np.full((self.num_bins, num_range_bins), np.nan, np.float32) for k in V2_FIELDS},
            )
            for ant in self.antenna_indices
        }

    def _reset_bins(self):
        if self._num_range_bins is not None:
            self._allocate_bins(self._num_range_bins)

    # -- window lifecycle -----------------------------------------------

    def _open_new_window(self, window_start_time: float):
        ts_str = time.strftime(FILENAME_TIMESTAMP_FMT, time.localtime(window_start_time))
        raw_path = os.path.join(self.output_dir, f"{ts_str}.bin")
        self._raw_file = open(raw_path, "ab")
        self._reset_bins()
        self._block_parser = RadarBlockParser(strict=True)  # fresh accumulator per window
        self._current_product_ts_str = ts_str
        self._looks_this_window = 0
        self._drops_at_open = (self.dropped_queue_full, self.dropped_parse_errors)
        print(f"[RadarArchiveService] window opened: {raw_path}", flush=True)

    def _close_window(self):
        if self._raw_file is not None:
            self._raw_file.close()
            self._raw_file = None
        dq = self.dropped_queue_full - self._drops_at_open[0]
        dp = self.dropped_parse_errors - self._drops_at_open[1]
        print(f"[RadarArchiveService] window {self._current_product_ts_str}: "
              f"{self._looks_this_window} looks, dropped_queue_full={dq}, "
              f"parse_errors={dp}, queue now {self._queue.qsize()}", flush=True)
        if self._num_range_bins is not None and self._current_product_ts_str is not None:
            base = os.path.join(self.output_dir, self._current_product_ts_str)
            kw = self._product_arrays()
            try:
                comb_path = f"{base}_products_v2_combined.npz"
                self._save_combined(comb_path, kw)
                print(f"[RadarArchiveService] combined saved: {comb_path}", flush=True)
            except Exception as e:
                print(f"[RadarArchiveService] combine failed: {e}", flush=True)
            product_path = None
            if self.write_full_products:
                product_path = f"{base}_products_v2.npz"
                np.savez_compressed(product_path, **kw)
                print(f"[RadarArchiveService] products saved: {product_path}", flush=True)

            if self.generate_pngs and product_path is not None:
                # Runs synchronously in the worker thread (brief pause in queue
                # draining once per window). Wrapped so plotting can't take
                # down archiving; imported here so the module is optional.
                try:
                    from plot_polar_products import render_npz_to_pngs
                    written = render_npz_to_pngs(
                        product_path,
                        output_dir=self.png_output_dir,
                        range_resolution_m=self.png_range_resolution_m,
                        log_scale=self.png_log_scale,
                    )
                    print(f"[RadarArchiveService] wrote {len(written)} PNG(s) "
                          f"to {self.png_output_dir}", flush=True)
                except Exception as e:
                    print(f"[RadarArchiveService] PNG generation failed: {e}", flush=True)

    def _product_arrays(self) -> dict:
        kw = {
            "num_bins": self.num_bins,
            "bin_width_deg": self.bin_width_deg,
            "bin_centers_deg": np.array(
                [self._bin_center_deg(i) for i in range(self.num_bins)]
            ),
            "antenna_indices": np.array(self.antenna_indices),
            "processor_version": PROCESSOR_VERSION,
            "stat_n_looks": self._looks_this_window,
        }
        for k, v in self._doppler.params().items():
            kw[f"param_{k}"] = v
        for ant, b in self._bins.items():
            p = f"antenna_{ant}_"
            kw[p + "power_linear"] = b.power_linear
            kw[p + "power_db"] = b.power_db
            kw[p + "snr_db"] = b.snr_db
            kw[p + "noise_floor"] = b.noise_floor
            kw[p + "moment1_hz"] = b.moment1_hz
            kw[p + "moment2_hz"] = b.moment2_hz
            kw[p + "pa_enable"] = b.pa_enable
            kw[p + "timestamp_us"] = b.timestamp_us
            kw[p + "hit_count"] = b.hit_count
            for k, v in b.v2.items():
                kw[p + k] = v
        return kw

    def _save_combined(self, path: str, kw: dict):
        comb = combine_v2(kw)
        out = {k: v for k, v in kw.items() if not (k.startswith("antenna_") and np.ndim(v) == 2)}
        for k, v in comb.items():
            out[k] = v.astype(np.float32) if (v.dtype == np.float64 and v.ndim == 2) else v
        np.savez_compressed(path, **out)

    def _save_products(self, path: str):
        """Kept for compatibility: full per-antenna v2 products."""
        np.savez_compressed(path, **self._product_arrays())

    # -- per-item processing (kept separate from the thread loop so it can
    #    be exercised directly, without real sockets/threads, in tests) --

    def process_item(self, raw_bytes: bytes, recv_time: float):
        archiving_now = self.is_archiving(recv_time)

        if archiving_now and not self._window_active:
            self._open_new_window(recv_time)
            self._window_active = True
        elif not archiving_now and self._window_active:
            self._close_window()
            self._window_active = False

        if not archiving_now:
            return  # outside window: drop, no writing, no processing

        # 1. raw payload -- write exactly as received.
        self._raw_file.write(raw_bytes)

        # 2. parse + block accumulation + Doppler + binning.
        try:
            pkt: RadarPacket = self._packet_parser.parse(raw_bytes)
        except ValueError:
            self.dropped_parse_errors += 1
            return

        look = self._block_parser.feed(pkt)
        if look is None:
            return

        antenna_index = pkt.antenna_index  # packet that completed the block
        product: DopplerProduct = self._doppler.process(look)

        if self._num_range_bins is None:
            self._allocate_bins(product.power_linear.shape[0])

        b = self._bin_index(product.look_angle_deg)
        arc = self._bins[antenna_index]
        arc.power_linear[b, :] = product.power_linear
        arc.power_db[b, :] = product.power_db
        arc.snr_db[b, :] = product.snr_db
        arc.noise_floor[b, :] = product.noise_floor
        arc.moment1_hz[b, :] = product.moment1_hz
        arc.moment2_hz[b, :] = product.moment2_hz
        arc.timestamp_us[b] = product.timestamp_us
        arc.pa_enable[b] = float(pkt.pa_enable)
        arc.hit_count[b] += 1
        for k in V2_FIELDS:
            arc.v2[k][b, :] = getattr(product, k)
        self._looks_this_window += 1

    # -- threads ------------------------------------------------------------

    def _listener_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        sock.settimeout(0.5)
        print(f"[RadarArchiveService] listening on {self.bind_ip}:{self.bind_port}")
        while not self._stop_event.is_set():
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            recv_time = time.time()
            try:
                self._queue.put_nowait((data, recv_time))
            except queue.Full:
                self.dropped_queue_full += 1
        sock.close()

    def _worker_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                data, recv_time = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.process_item(data, recv_time)
        if self._window_active:
            self._close_window()
            self._window_active = False

    def run(self):
        listener = threading.Thread(target=self._listener_loop, daemon=True)
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        listener.start()
        worker.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[RadarArchiveService] stopping...")
            self._stop_event.set()
            listener.join(timeout=2.0)
            worker.join(timeout=5.0)
            print(f"[RadarArchiveService] dropped_queue_full={self.dropped_queue_full} "
                  f"dropped_parse_errors={self.dropped_parse_errors}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--bind-port", type=int, default=34952)
    ap.add_argument("--output-dir", default="/mnt/share/v2")
    ap.add_argument("--cycle-minutes", type=float, default=60.0)
    ap.add_argument("--archive-minutes", type=float, default=5.0)
    ap.add_argument("--num-bins", type=int, default=360)
    ap.add_argument("--png-output-dir", default=None,
                     help="directory for auto-generated PNGs "
                          "(default: <output-dir>/pngs)")
    ap.add_argument("--no-png", action="store_true",
                     help="disable automatic PNG generation on window close")
    ap.add_argument("--png-log-scale", action="store_true",
                     help="use log-scale coloring for power_linear/noise_floor PNGs")
    ap.add_argument("--png-range-resolution-m", type=float, default=None,
                     help="meters per range bin, to label PNG radial axis in meters")
    ap.add_argument("--start-time", type=float, default=None,
                     help="epoch seconds anchoring the cycle phase (default: process start, as v1). "
                          "e.g. 0 aligns windows to multiples of --cycle-minutes past the hour (UTC)")
    ap.add_argument("--no-full-products", action="store_true",
                     help="write only the small _products_v2_combined.npz, not the per-antenna file")
    ap.add_argument("--window", default="none", choices=["none", "hann"])
    ap.add_argument("--region-k", type=float, default=3.0)
    ap.add_argument("--detect-k", type=float, default=10.0)
    ap.add_argument("--tail-fraction", type=float, default=0.25)
    args = ap.parse_args()

    service = RadarArchiveService(
        bind_ip=args.bind_ip, bind_port=args.bind_port,
        output_dir=args.output_dir, cycle_minutes=args.cycle_minutes,
        archive_minutes=args.archive_minutes, num_bins=args.num_bins,
        generate_pngs=not args.no_png,
        png_output_dir=args.png_output_dir,
        png_log_scale=args.png_log_scale,
        png_range_resolution_m=args.png_range_resolution_m,
        start_time=args.start_time,
        processor_kwargs=dict(window=args.window, region_k=args.region_k,
                              detect_k=args.detect_k, tail_fraction=args.tail_fraction),
        write_full_products=not args.no_full_products,
    )
    print(f"[RadarArchiveService v2] output_dir={args.output_dir} "
          f"processor={service._doppler.params()}", flush=True)
    service.run()


if __name__ == "__main__":
    main()
