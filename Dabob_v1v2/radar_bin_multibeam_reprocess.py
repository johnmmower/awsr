#!/usr/bin/env python3
"""
Offline reprocessor for a raw <timestamp>.bin archive file.

Rebuilds the same per-antenna, per-angle-bin Doppler products that
radar_archive_service.py produces live (power_linear, power_db, snr_db,
noise_floor, moment1_hz, moment2_hz), keeping only the LAST look to land
in each (antenna, angle bin) -- same overwrite semantics as the live
service.

ADDITIONALLY (this is the new part):
  - Each bin's last magnitude Doppler spectrum (|DFT|, not just the
    scalar stats derived from it) is retained per antenna.
  - After the whole file is processed, for each world-angle bin, the
    magnitude spectra from whichever antennas hit that bin are averaged
    together (incoherent / "video" integration), and stats (power_db,
    snr_db, moments) are recomputed from that averaged spectrum.

Rationale (per user): with a 2-minute archive window and ~2-3 hits per
bin per antenna, using just the last hit per antenna is fine -- but
averaging the last hit ACROSS the 4 antennas at a shared world-angle bin
gives an incoherent integration gain of sqrt(N_antennas) (sqrt(4) = 2),
since each antenna's last look at that bin is a statistically
independent noise realization of the same underlying scene.

CAVEATS -- flagging explicitly, not verified against any published
reference:
  - "Averaging the spectrum (magnitude)" is implemented literally as an
    element-wise mean of |spectrum| across contributing antennas (not
    mean of power, not RMS). This is a common video-integration approach,
    but it's a design choice, not the only valid one -- if you intended
    mean-of-power instead, that's a one-line change (see
    _combine_magnitude below).
  - The sqrt(N) incoherent-integration-gain figure is the standard
    radar-theory result for averaging N independent noise-like samples;
    it is a property of the math, not something this script verifies
    against your actual data.
  - Antennas that never hit a given world-angle bin during the window
    are simply excluded from that bin's average (NaN-safe), so the
    number of antennas actually averaged varies per bin --
    `combined_num_antennas` records that count per bin.

Reuses the same header/packet format and Doppler-processing formulas as
radar_packet_parser.py / doppler_processor.py. Imports them if available
alongside this file; otherwise falls back to inline copies (flagged).
"""

from dataclasses import dataclass, field
from pathlib import Path
import struct

import numpy as np

# ---------------------------------------------------------------------------
# Packet / header format
# ---------------------------------------------------------------------------
try:
    from radar_packet_parser import (
        HEADER_STRUCT, HEADER_SIZE, NUM_RANGE_BINS,
        SAMPLES_STRUCT, PAYLOAD_SIZE, PACKET_SIZE,
        ANTENNA_OFFSETS_DEG,
    )
except ImportError:
    HEADER_STRUCT = struct.Struct('<QQHHId')
    HEADER_SIZE = HEADER_STRUCT.size  # 32 bytes
    NUM_RANGE_BINS = 1024
    SAMPLES_COUNT = 2 * NUM_RANGE_BINS
    SAMPLES_STRUCT = struct.Struct(f'<{SAMPLES_COUNT}i')
    PAYLOAD_SIZE = SAMPLES_STRUCT.size  # 8192 bytes
    PACKET_SIZE = HEADER_SIZE + PAYLOAD_SIZE  # 8224 bytes
    ANTENNA_OFFSETS_DEG = {0: 180.0, 1: 90.0, 2: 0.0, 3: -90.0}

BLOCK_SIZE = 64
COUNTER_STEP = 16
COUNTER_WRAP = 65536


@dataclass
class RadarPacket:
    counter: int
    timestamp_us: int
    antenna_index: int
    pa_enable: int
    reserved: int
    encoder_position_deg: float
    range_profile: np.ndarray  # complex128, shape (NUM_RANGE_BINS,)


def _parse_packet(chunk: bytes) -> RadarPacket:
    counter, ts_us, antenna, pa, reserved, encoder = HEADER_STRUCT.unpack(
        chunk[:HEADER_SIZE]
    )
    samples = np.array(
        SAMPLES_STRUCT.unpack(chunk[HEADER_SIZE:HEADER_SIZE + PAYLOAD_SIZE])
    )
    real = samples[:NUM_RANGE_BINS]
    imag = samples[NUM_RANGE_BINS:]
    range_profile = (real + 1j * imag).astype(np.complex128)
    return RadarPacket(counter, ts_us, antenna, pa, reserved, encoder, range_profile)


def iter_packets_from_bin(path, skip_malformed: bool = True):
    """Read a .bin archive file and yield RadarPacket records."""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(PACKET_SIZE)
            if not chunk:
                return
            if len(chunk) < PACKET_SIZE:
                print(f"[iter_packets_from_bin] trailing partial record "
                      f"({len(chunk)} bytes), discarding")
                return
            try:
                yield _parse_packet(chunk)
            except struct.error as e:
                if skip_malformed:
                    print(f"[iter_packets_from_bin] skipping malformed record: {e}")
                    continue
                raise


# ---------------------------------------------------------------------------
# Block accumulation -> RadarLook (faithful reimplementation of
# radar_packet_parser.RadarBlockParser)
# ---------------------------------------------------------------------------
@dataclass
class RadarLook:
    look_angle_deg: float
    timestamp_us: int
    complex_data: np.ndarray  # complex128, shape (NUM_RANGE_BINS, BLOCK_SIZE)


class _BlockAccumulator:
    def __init__(self):
        self._buffer = []
        self._expected_counter = None
        self._block_antenna = None
        self.dropped_blocks = 0
        self.dropped_packets = 0

    def _expected_next(self, counter):
        return (counter + COUNTER_STEP) % COUNTER_WRAP

    def _start_new_block(self, pkt):
        self._buffer = [pkt]
        self._expected_counter = self._expected_next(pkt.counter)
        self._block_antenna = pkt.antenna_index

    def _build_look(self, block):
        first = block[0]
        if first.antenna_index not in ANTENNA_OFFSETS_DEG:
            raise ValueError(f"unknown antenna_index {first.antenna_index}")
        offset = ANTENNA_OFFSETS_DEG[first.antenna_index]
        look_angle_deg = (first.encoder_position_deg + offset) % 360.0
        complex_data = np.stack([p.range_profile for p in block], axis=1)
        look = RadarLook(look_angle_deg, first.timestamp_us, complex_data)
        # antenna_index isn't part of the upstream RadarLook schema (by
        # original design), but this script needs it to know which
        # antenna's bin array to write into -- attached separately rather
        # than changing RadarLook's shape.
        return first.antenna_index, look

    def feed(self, pkt):
        """Returns (antenna_index, RadarLook) if this packet completes a
        block, else None."""
        if not self._buffer:
            self._start_new_block(pkt)
            return None

        counter_ok = (pkt.counter == self._expected_counter)
        antenna_ok = (pkt.antenna_index == self._block_antenna)

        if not (counter_ok and antenna_ok):
            self.dropped_packets += len(self._buffer)
            self.dropped_blocks += 1
            self._start_new_block(pkt)
            return None

        self._buffer.append(pkt)
        self._expected_counter = self._expected_next(pkt.counter)

        if len(self._buffer) == BLOCK_SIZE:
            block = self._buffer
            self._buffer = []
            self._expected_counter = None
            self._block_antenna = None
            return self._build_look(block)

        return None


# ---------------------------------------------------------------------------
# Doppler processing (faithful reimplementation of doppler_processor.py)
# ---------------------------------------------------------------------------
def _spectrum_stats(power: np.ndarray, freq_hz: np.ndarray, tail_fraction: float):
    """Shared stats computation given a power array (range_bins, pulses)
    and its frequency axis. Used both per-look (power = |DFT|^2) and for
    the cross-antenna combined spectrum (power = combined_magnitude^2).
    Formulas copied verbatim from doppler_processor.py's DopplerProcessor.process().
    """
    num_pulses = power.shape[1]
    n_tail = max(1, int(round(num_pulses * tail_fraction)))
    tail_idx = np.concatenate([np.arange(0, n_tail),
                                np.arange(num_pulses - n_tail, num_pulses)])
    noise_floor = np.mean(power[:, tail_idx], axis=1)

    peak_power = np.max(power, axis=1)
    power_linear = peak_power
    with np.errstate(divide='ignore', invalid='ignore'):
        power_db = 10.0 * np.log10(np.where(power_linear > 0, power_linear, np.nan))
        snr_linear = np.where(noise_floor > 0, peak_power / noise_floor, np.inf)
        snr_db = 10.0 * np.log10(snr_linear)

    noise_sub_power = np.clip(power - noise_floor[:, np.newaxis], a_min=0.0, a_max=None)
    total_power = np.sum(noise_sub_power, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        moment1_hz = np.where(
            total_power > 0,
            np.sum(noise_sub_power * freq_hz[np.newaxis, :], axis=1) / total_power,
            0.0,
        )
        variance_hz2 = np.where(
            total_power > 0,
            np.sum(noise_sub_power * (freq_hz[np.newaxis, :] - moment1_hz[:, np.newaxis]) ** 2,
                   axis=1) / total_power,
            0.0,
        )
    moment2_hz = np.sqrt(variance_hz2)

    return dict(noise_floor=noise_floor, power_linear=power_linear, power_db=power_db,
                snr_db=snr_db, moment1_hz=moment1_hz, moment2_hz=moment2_hz)


def _look_products(look: RadarLook, fs_hz: float, tail_fraction: float):
    """Per-look Doppler processing. Returns dict of stats plus the full
    magnitude spectrum (range_bins, pulses) -- the extra thing the live
    archive service discards after computing stats."""
    num_range_bins, num_pulses = look.complex_data.shape
    spectrum = np.fft.fftshift(np.fft.fft(look.complex_data, axis=1), axes=1)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    freq_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1.0 / fs_hz))
    stats = _spectrum_stats(power, freq_hz, tail_fraction)
    stats["magnitude"] = magnitude
    stats["freq_hz"] = freq_hz
    return stats


def _combine_magnitude(magnitude_stack: np.ndarray):
    """magnitude_stack: shape (n_antennas_present, range_bins, pulses).
    Element-wise mean across antennas -- incoherent/video integration.
    Change to sqrt(mean(magnitude**2)) here if you want mean-of-power
    (RMS) combination instead of mean-of-magnitude."""
    return np.mean(magnitude_stack, axis=0)


# ---------------------------------------------------------------------------
# Per-bin storage
# ---------------------------------------------------------------------------
RANGE_RESOLVED = ("power_linear", "power_db", "snr_db", "noise_floor",
                   "moment1_hz", "moment2_hz")


@dataclass
class _AntennaBins:
    num_bins: int
    num_range_bins: int
    num_pulses: int
    power_linear: np.ndarray = None
    power_db: np.ndarray = None
    snr_db: np.ndarray = None
    noise_floor: np.ndarray = None
    moment1_hz: np.ndarray = None
    moment2_hz: np.ndarray = None
    pa_enable: np.ndarray = None
    timestamp_us: np.ndarray = None
    hit_count: np.ndarray = None
    spectrum_magnitude: np.ndarray = None  # (num_bins, num_range_bins, num_pulses)

    def __post_init__(self):
        for name in RANGE_RESOLVED:
            setattr(self, name, np.full((self.num_bins, self.num_range_bins), np.nan))
        self.pa_enable = np.full(self.num_bins, np.nan)
        self.timestamp_us = np.full(self.num_bins, np.nan)
        self.hit_count = np.zeros(self.num_bins, dtype=np.int64)
        # float32 to keep this from becoming enormous: num_bins * range_bins * pulses
        self.spectrum_magnitude = np.full(
            (self.num_bins, self.num_range_bins, self.num_pulses), np.nan, dtype=np.float32
        )


class RadarBinMultibeamReprocessor:
    """Reads a raw .bin archive, rebuilds per-antenna binned Doppler
    products (last-hit-wins per bin, matching the live archive service),
    retains each bin's last magnitude spectrum per antenna, then combines
    (incoherently averages) that spectrum across antennas per
    world-angle bin for an sqrt(N_antennas) SNR gain.
    """

    def __init__(self, num_bins: int = 360, fs_hz: float = 1562.5,
                 tail_fraction: float = 0.25,
                 antenna_indices=(0, 1, 2, 3),
                 num_range_bins: int = NUM_RANGE_BINS,
                 num_pulses: int = BLOCK_SIZE):
        self.num_bins = num_bins
        self.fs_hz = fs_hz
        self.tail_fraction = tail_fraction
        self.antenna_indices = tuple(antenna_indices)
        self.num_range_bins = num_range_bins
        self.num_pulses = num_pulses
        self.freq_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1.0 / fs_hz))

        self._bins = {
            a: _AntennaBins(num_bins, num_range_bins, num_pulses)
            for a in self.antenna_indices
        }
        self.dropped_blocks = 0
        self.dropped_packets = 0
        self.num_looks_processed = 0

        # Filled in by combine() after the whole file is processed.
        self.combined_power_linear = None
        self.combined_power_db = None
        self.combined_snr_db = None
        self.combined_noise_floor = None
        self.combined_moment1_hz = None
        self.combined_moment2_hz = None
        self.combined_num_antennas = None
        self.combined_spectrum_magnitude = None

    def _bin_index(self, angle_deg: float) -> int:
        bin_width = 360.0 / self.num_bins
        return int(angle_deg // bin_width) % self.num_bins

    def process_bin_file(self, path):
        """Read the whole .bin file, rebuild looks, bin per antenna
        (last-write-wins), then compute the cross-antenna combined
        spectrum. Call this once."""
        accumulator = _BlockAccumulator()
        for pkt in iter_packets_from_bin(path):
            result = accumulator.feed(pkt)
            if result is not None:
                antenna, look = result
                self._process_look(antenna, look)
        self.dropped_blocks = accumulator.dropped_blocks
        self.dropped_packets = accumulator.dropped_packets
        self._combine()

    def _process_look(self, antenna: int, look: RadarLook):
        if antenna not in self._bins:
            return

        products = _look_products(look, self.fs_hz, self.tail_fraction)
        bin_idx = self._bin_index(look.look_angle_deg)
        ab = self._bins[antenna]

        for name in RANGE_RESOLVED:
            getattr(ab, name)[bin_idx, :] = products[name]
        ab.timestamp_us[bin_idx] = look.timestamp_us
        ab.hit_count[bin_idx] += 1
        ab.spectrum_magnitude[bin_idx, :, :] = products["magnitude"].astype(np.float32)

        self.num_looks_processed += 1

    def _combine(self):
        num_bins, num_range_bins, num_pulses = self.num_bins, self.num_range_bins, self.num_pulses
        combined_mag = np.full((num_bins, num_range_bins, num_pulses), np.nan, dtype=np.float32)
        num_antennas = np.zeros(num_bins, dtype=np.int64)

        for b in range(num_bins):
            stack = []
            for a in self.antenna_indices:
                m = self._bins[a].spectrum_magnitude[b]
                if not np.isnan(m).all():
                    stack.append(m)
            if stack:
                combined_mag[b] = _combine_magnitude(np.stack(stack, axis=0))
                num_antennas[b] = len(stack)

        self.combined_spectrum_magnitude = combined_mag
        self.combined_num_antennas = num_antennas

        power_linear = np.full((num_bins, num_range_bins), np.nan)
        power_db = np.full((num_bins, num_range_bins), np.nan)
        snr_db = np.full((num_bins, num_range_bins), np.nan)
        noise_floor = np.full((num_bins, num_range_bins), np.nan)
        moment1_hz = np.full((num_bins, num_range_bins), np.nan)
        moment2_hz = np.full((num_bins, num_range_bins), np.nan)

        for b in range(num_bins):
            if num_antennas[b] == 0:
                continue
            power = combined_mag[b].astype(np.float64) ** 2
            stats = _spectrum_stats(power, self.freq_hz, self.tail_fraction)
            power_linear[b] = stats["power_linear"]
            power_db[b] = stats["power_db"]
            snr_db[b] = stats["snr_db"]
            noise_floor[b] = stats["noise_floor"]
            moment1_hz[b] = stats["moment1_hz"]
            moment2_hz[b] = stats["moment2_hz"]

        self.combined_power_linear = power_linear
        self.combined_power_db = power_db
        self.combined_snr_db = snr_db
        self.combined_noise_floor = noise_floor
        self.combined_moment1_hz = moment1_hz
        self.combined_moment2_hz = moment2_hz

    def save_npz(self, path):
        """Saves per-antenna and combined scalar Doppler products only --
        NOT the underlying magnitude spectrum arrays (per-antenna or
        combined). Those are still used internally during combine() to
        compute the combined stats correctly (you can't average per-antenna
        snr_db/moments after the fact and get the same answer as averaging
        the actual spectra first), they're just not persisted to disk.
        This is what took the file from ~560MB down to a few tens of MB.
        """
        out = {}
        for a in self.antenna_indices:
            ab = self._bins[a]
            for name in RANGE_RESOLVED:
                out[f"antenna_{a}_{name}"] = getattr(ab, name)
            out[f"antenna_{a}_pa_enable"] = ab.pa_enable
            out[f"antenna_{a}_timestamp_us"] = ab.timestamp_us
            out[f"antenna_{a}_hit_count"] = ab.hit_count
        out["combined_power_linear"] = self.combined_power_linear
        out["combined_power_db"] = self.combined_power_db
        out["combined_snr_db"] = self.combined_snr_db
        out["combined_noise_floor"] = self.combined_noise_floor
        out["combined_moment1_hz"] = self.combined_moment1_hz
        out["combined_moment2_hz"] = self.combined_moment2_hz
        out["combined_num_antennas"] = self.combined_num_antennas
        out["freq_hz"] = self.freq_hz
        np.savez(path, **out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bin_path", help="Path to a raw <timestamp>.bin archive file")
    ap.add_argument("--output", default=None, help="Output .npz path (default: <bin_path>_multibeam.npz)")
    ap.add_argument("--num-bins", type=int, default=360)
    args = ap.parse_args()

    out_path = args.output or (str(Path(args.bin_path).with_suffix("")) + "_multibeam.npz")
    proc = RadarBinMultibeamReprocessor(num_bins=args.num_bins)
    proc.process_bin_file(args.bin_path)
    proc.save_npz(out_path)
    print(f"Processed {proc.num_looks_processed} looks, "
          f"dropped {proc.dropped_blocks} incoherent blocks. Saved to {out_path}")
