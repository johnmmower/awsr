#!/usr/bin/env python3
"""
Doppler spectral processor.

Consumes "look" data -- anything with the three fields:
    look_angle_deg : float, antenna pointing angle for this look
    timestamp_us   : int, look start time
    complex_data   : complex128 ndarray, shape (num_range_bins, num_pulses)
                      (range axis first, slow-time/pulse axis second)

Duck typed (object or dict), no import coupling to radar_packet_parser.py.

PROCESSOR_VERSION 2 -- what changed and why
-------------------------------------------
The original outputs (noise_floor, power_linear, power_db, snr_db,
moment1_hz, moment2_hz) are computed EXACTLY as before, from the same
unwindowed spectrum, so existing files/code stay comparable.

Why new moments were needed: the v1 moments weight every bin by
max(P - N, 0). For a noise-only bin, P is exponentially distributed with
mean N, and E[max(P - N, 0)] = N/e ~= 0.37 N. So all ~64 bins together add
~23 N of weight spread symmetrically across the band, which pulls the
centroid toward 0 Hz. On Dabob data the measured shrinkage was ~0.5 at
22 dB reported SNR. The reported SNR is peak-bin / mean noise, which for
pure noise is ~ln(64)+0.58 ~= 4.7 (6.7 dB) -- matching the observed median
SNR of 6.8 dB, i.e. most cells are noise.

New per-range-bin outputs (all on an optionally windowed spectrum):
  noise_floor_v2   robust noise estimate from tail bins: median / ln 2
                   (for exponential power the median is N ln 2), less
                   sensitive than the mean to signal leaking into the tails
  moment1_pk_hz    centroid of (P - N) over the contiguous region around the
  moment2_pk_hz    spectral peak where P > region_k * N; region may wrap
                   across +/- fs/2 (frequencies are unwrapped relative to
                   the peak, result rewrapped into [-fs/2, fs/2))
  n_bins_pk        width of that region, in bins
  detected         peak > detect_k * N (else moment1_pk/moment2_pk = NaN)
  snr_band_db      sum over ALL bins of (P - N) / (num_pulses * N): signal
                   power over total in-band noise power. This, not peak SNR,
                   is what governs a full-band centroid's noise bias.
  snr_pk_db        same numerator restricted to the peak region

DESIGN CHOICES / GUESSES (not derived from data, flagged explicitly):
  - fs_hz = 1562.5 Hz (640 us/pulse, from sender timing), passed in.
  - tail_fraction = 0.25 per side for the noise estimate (unchanged).
  - window = "none" for the v2 products. In the self-test (sea-like return,
    no strong interferer) the rectangular window gave ~25% lower moment noise
    and higher detection than Hann. "hann" is available: rectangular
    sidelobes are -13 dB, so a strong narrowband return in the same range bin
    (e.g. aliased land at 0 Hz) leaks across the band, and Hann (~-31 dB
    sidelobes) may be better near land. Not tested on real data.
  - region_k = 3.0: a noise-only bin exceeds 3 N with probability e^-3 ~= 5%,
    but it only enters if contiguous with the peak.
  - detect_k = 10.0: measured false-detection rate on pure noise ~2% (the
    ideal-noise-floor figure would be lower; the 32-sample tail estimate
    scatters). 8 gives ~6%, 12 gives ~0.6%. Lower keeps more weak returns
    and more false detections.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

PROCESSOR_VERSION = 2


@dataclass
class DopplerProduct:
    look_angle_deg: float
    timestamp_us: int
    doppler_freq_hz: np.ndarray   # shape (num_pulses,), fftshifted frequency axis
    spectrum: np.ndarray          # complex128, (num_range_bins, num_pulses), fftshifted, UNwindowed
    noise_floor: np.ndarray       # v1: mean of tail-bin power
    power_linear: np.ndarray      # v1: absolute peak spectral power
    power_db: np.ndarray          # v1: 10*log10(power_linear)
    snr_db: np.ndarray            # v1: peak power / noise_floor
    moment1_hz: np.ndarray        # v1: centroid of max(P - N, 0) over full band
    moment2_hz: np.ndarray        # v1: width of same
    # --- v2 additions (defaults keep old constructors working) ---
    noise_floor_v2: Optional[np.ndarray] = None
    moment1_pk_hz: Optional[np.ndarray] = None
    moment2_pk_hz: Optional[np.ndarray] = None
    n_bins_pk: Optional[np.ndarray] = None
    detected: Optional[np.ndarray] = None
    snr_band_db: Optional[np.ndarray] = None
    snr_pk_db: Optional[np.ndarray] = None
    processor_version: int = PROCESSOR_VERSION


def _get_field(look: Any, name: str):
    if isinstance(look, dict):
        return look[name]
    return getattr(look, name)


def _window(name: str, n: int) -> np.ndarray:
    if name in (None, "none", "rect"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    raise ValueError(f"unknown window {name!r}")


class DopplerProcessor:
    """Stateless across looks; safe to call process() from a worker thread."""

    def __init__(self, fs_hz: float = 1562.5, tail_fraction: float = 0.25,
                 window: str = "none", region_k: float = 3.0, detect_k: float = 10.0,
                 compute_v2: bool = True):
        if not (0.0 < tail_fraction <= 0.5):
            raise ValueError("tail_fraction must be in (0, 0.5]")
        if region_k <= 0 or detect_k <= 0:
            raise ValueError("region_k and detect_k must be > 0")
        self.fs_hz = fs_hz
        self.tail_fraction = tail_fraction
        self.window = window
        self.region_k = region_k
        self.detect_k = detect_k
        self.compute_v2 = compute_v2

    def params(self) -> dict:
        """Parameters to store alongside products, so files are self-describing."""
        return dict(processor_version=PROCESSOR_VERSION, fs_hz=self.fs_hz,
                    tail_fraction=self.tail_fraction, window=str(self.window),
                    region_k=self.region_k, detect_k=self.detect_k)

    # ------------------------------------------------------------------ v1
    def _v1(self, power, freq_hz, tail_idx):
        noise_floor = np.mean(power[:, tail_idx], axis=1)
        peak_power = np.max(power, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            power_db = 10.0 * np.log10(np.where(peak_power > 0, peak_power, np.nan))
            snr_linear = np.where(noise_floor > 0, peak_power / noise_floor, np.inf)
            snr_db = 10.0 * np.log10(snr_linear)
        nsp = np.clip(power - noise_floor[:, np.newaxis], a_min=0.0, a_max=None)
        total = np.sum(nsp, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            m1 = np.where(total > 0, np.sum(nsp * freq_hz[None, :], axis=1) / total, 0.0)
            var = np.where(total > 0,
                           np.sum(nsp * (freq_hz[None, :] - m1[:, None]) ** 2, axis=1) / total, 0.0)
        return noise_floor, peak_power, power_db, snr_db, m1, np.sqrt(var)

    # ------------------------------------------------------------------ v2
    def _v2(self, power, freq_hz, tail_idx):
        nr, n = power.shape
        noise = np.median(power[:, tail_idx], axis=1) / np.log(2.0)
        noise = np.where(noise > 0, noise, np.nan)

        pk = np.argmax(power, axis=1)
        peak = power[np.arange(nr), pk]
        detected = peak > self.detect_k * noise

        # Re-index each row so its peak sits at column c; then the peak region
        # is the run of above-threshold bins contiguous with column c
        # (circular, so a region may wrap across +/- fs/2).
        c = n // 2
        idx = (np.arange(n)[None, :] + pk[:, None] - c) % n
        p_rot = np.take_along_axis(power, idx, axis=1)
        above = p_rot > self.region_k * noise[:, None]
        right = np.cumprod(above[:, c:], axis=1).astype(bool)
        left = np.cumprod(above[:, :c + 1][:, ::-1], axis=1)[:, ::-1].astype(bool)
        region = np.concatenate([left[:, :-1], right], axis=1)
        # guard against a region covering the whole circle twice
        region &= above

        df = self.fs_hz / n
        f_rel = (np.arange(n) - c) * df           # frequency relative to the peak bin
        w = np.where(region, p_rot - noise[:, None], 0.0)
        wsum = w.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_rel = (w * f_rel[None, :]).sum(axis=1) / wsum
            var = (w * (f_rel[None, :] - mu_rel[:, None]) ** 2).sum(axis=1) / wsum
        f_pk = freq_hz[pk]
        half = self.fs_hz / 2.0
        m1 = (f_pk + mu_rel + half) % self.fs_hz - half
        ok = detected & (wsum > 0)
        m1 = np.where(ok, m1, np.nan)
        m2 = np.where(ok, np.sqrt(var), np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            band = (power - noise[:, None]).sum(axis=1) / (n * noise)
            snr_band_db = 10.0 * np.log10(np.where(band > 0, band, np.nan))
            pkr = wsum / (n * noise)
            snr_pk_db = 10.0 * np.log10(np.where(pkr > 0, pkr, np.nan))
        return noise, m1, m2, region.sum(axis=1).astype(np.int16), detected, snr_band_db, snr_pk_db

    def process(self, look: Any) -> DopplerProduct:
        look_angle_deg = _get_field(look, "look_angle_deg")
        timestamp_us = _get_field(look, "timestamp_us")
        complex_data = np.asarray(_get_field(look, "complex_data"))
        if complex_data.ndim != 2:
            raise ValueError(f"complex_data must be 2-D (range_bins, pulses), got {complex_data.shape}")
        num_range_bins, num_pulses = complex_data.shape

        spectrum = np.fft.fftshift(np.fft.fft(complex_data, axis=1), axes=1)
        power = np.abs(spectrum) ** 2
        freq_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1.0 / self.fs_hz))
        n_tail = max(1, int(round(num_pulses * self.tail_fraction)))
        tail_idx = np.concatenate([np.arange(0, n_tail), np.arange(num_pulses - n_tail, num_pulses)])

        nf, pp, pdb, snr, m1, m2 = self._v1(power, freq_hz, tail_idx)
        prod = DopplerProduct(look_angle_deg=look_angle_deg, timestamp_us=timestamp_us,
                              doppler_freq_hz=freq_hz, spectrum=spectrum, noise_floor=nf,
                              power_linear=pp, power_db=pdb, snr_db=snr,
                              moment1_hz=m1, moment2_hz=m2)
        if self.compute_v2:
            win = _window(self.window, num_pulses)
            if self.window in (None, "none", "rect"):
                power_w = power
            else:
                sw = np.fft.fftshift(np.fft.fft(complex_data * win[None, :], axis=1), axes=1)
                power_w = np.abs(sw) ** 2
            (prod.noise_floor_v2, prod.moment1_pk_hz, prod.moment2_pk_hz, prod.n_bins_pk,
             prod.detected, prod.snr_band_db, prod.snr_pk_db) = self._v2(power_w, freq_hz, tail_idx)
        return prod


def process_look(look: Any, fs_hz: float = 1562.5, tail_fraction: float = 0.25,
                 **kw) -> DopplerProduct:
    return DopplerProcessor(fs_hz=fs_hz, tail_fraction=tail_fraction, **kw).process(look)


# ---------------------------------------------------------------------- tests
def _sea_like(rng, nr, n, fs, f0, width, amp2):
    """Complex Gaussian process with a Gaussian-shaped Doppler PSD (circular)."""
    f = np.fft.fftfreq(n, d=1.0 / fs)
    d = (f - f0 + fs / 2) % fs - fs / 2
    psd = np.exp(-0.5 * (d / width) ** 2)
    psd *= amp2 * n / psd.sum()                      # mean |x|^2 per pulse = amp2
    z = (rng.normal(size=(nr, n)) + 1j * rng.normal(size=(nr, n))) / np.sqrt(2)
    return np.fft.ifft(z * np.sqrt(psd)[None, :] * np.sqrt(n), axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs, n = 1562.5, 64

    # 1. original self-test: strong tone
    t = np.arange(n) / fs
    data = 40.0 * np.exp(1j * 2 * np.pi * 120.0 * t)[None, :] + \
        5.0 * (rng.normal(size=(8, n)) + 1j * rng.normal(size=(8, n)))
    p = process_look({"look_angle_deg": 42.0, "timestamp_us": 1, "complex_data": data}, fs_hz=fs)
    print(f"tone 120 Hz: v1 moment1 {p.moment1_hz.mean():.1f} Hz, v2 moment1_pk {np.nanmean(p.moment1_pk_hz):.1f} Hz")
    assert abs(p.moment1_hz.mean() - 120) < 10 and abs(np.nanmean(p.moment1_pk_hz) - 120) < 10

    # 2. pure noise: peak-over-mean SNR statistic and false detection rate
    nr = 20000
    noise = (rng.normal(size=(nr, n)) + 1j * rng.normal(size=(nr, n))) / np.sqrt(2)
    p = process_look({"look_angle_deg": 0, "timestamp_us": 0, "complex_data": noise}, fs_hz=fs)
    print(f"pure noise: median v1 snr_db {np.median(p.snr_db):.2f} dB "
          f"(expect ~6.7), v2 false-detect rate {p.detected.mean()*100:.1f}%")

    # 3. shrinkage vs SNR for a sea-like return at +60 Hz, 25 Hz wide
    f0, width = 60.0, 25.0
    print("\n band SNR  v1 snr_db   gamma_v1  gamma_v2  sd_v1(Hz) sd_v2(Hz) detected")
    for sb_db in [-12, -8, -4, 0, 4, 8, 12, 20]:
        amp2 = 10 ** (sb_db / 10)                    # signal / noise power per pulse
        x = _sea_like(rng, nr, n, fs, f0, width, amp2)
        x += (rng.normal(size=(nr, n)) + 1j * rng.normal(size=(nr, n))) / np.sqrt(2)
        p = process_look({"look_angle_deg": 0, "timestamp_us": 0, "complex_data": x}, fs_hz=fs)
        g1 = np.mean(p.moment1_hz) / f0
        g2 = np.nanmean(p.moment1_pk_hz) / f0
        print(f"  {sb_db:+4d} dB  {np.median(p.snr_db):6.1f} dB  {g1:8.3f}  {g2:8.3f}  "
              f"{np.std(p.moment1_hz):8.1f}  {np.nanstd(p.moment1_pk_hz):8.1f}  {p.detected.mean()*100:5.1f}%")
    print("\nSelf-test passed.")
