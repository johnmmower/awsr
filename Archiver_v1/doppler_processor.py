#!/usr/bin/env python3
"""
Doppler spectral processor.

Consumes "look" data -- anything with the three fields:
    look_angle_deg : float, antenna pointing angle for this look
    timestamp_us   : int, look start time
    complex_data   : complex128 ndarray, shape (num_range_bins, num_pulses)
                      (range axis first, slow-time/pulse axis second)

This is intentionally decoupled from radar_packet_parser.py's RadarLook --
it works with any object or dict exposing those three fields (duck typed),
since the intended usage is one thread producing looks onto a queue and a
separate thread (running this processor) consuming them, with no direct
import coupling between the two stages.

Per look, per range bin, this computes:
  1. DFT along the slow-time (pulse) axis -> Doppler spectrum
  2. Noise floor estimate, from the "tail" bins of the (fftshifted) spectrum
     -- i.e. the bins at the extremes of the Doppler frequency axis, on the
     assumption (per prior radar/clutter discussion) that any real Doppler
     signal is narrowband and sits away from those extreme frequencies, so
     the tails are unlikely to contain signal and are dominated by the
     noise floor.
  3. SNR per range bin: peak spectral power vs. that noise floor, in dB.
     Also reports the absolute peak spectral power itself (linear and dB),
     independent of the noise floor -- useful when you need actual power
     level rather than just a noise-relative ratio.
  4. First and second Doppler moments (mean Doppler frequency, and spectral
     width / standard deviation around that mean), computed as
     power-weighted moments of the frequency axis, on the noise-floor-
     subtracted spectrum (any bin's noise-subtracted power clipped to >= 0
     before weighting, which is a common approach to keep the flat noise
     background from dominating/biasing the moment estimates -- not a
     specific published method being invoked here, just a reasonable
     design choice).

DESIGN CHOICES / GUESSES (not derived from data, flagged explicitly):
  - `fs_hz` (pulse repetition rate feeding the slow-time DFT) defaults to
    1562.5 Hz, matching the known system timing (640 us per pulse) from
    the sender/generator script. This is passed in, not measured from the
    look object itself -- if your pulse spacing changes, update it.
  - `tail_fraction` (how much of each spectrum edge counts as "tail" for
    noise estimation) defaults to 0.25 (outer 25% of bins on each side of
    the shifted spectrum, i.e. the bins farthest from DC). This is a
    tunable parameter, not a fixed/measured value.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DopplerProduct:
    look_angle_deg: float
    timestamp_us: int
    doppler_freq_hz: np.ndarray   # shape (num_pulses,), fftshifted frequency axis
    spectrum: np.ndarray          # complex128, shape (num_range_bins, num_pulses), fftshifted
    noise_floor: np.ndarray       # float64, shape (num_range_bins,), power units
    power_linear: np.ndarray      # float64, shape (num_range_bins,) -- absolute peak spectral power
    power_db: np.ndarray          # float64, shape (num_range_bins,) -- 10*log10(power_linear)
    snr_db: np.ndarray            # float64, shape (num_range_bins,) -- power_linear vs noise_floor
    moment1_hz: np.ndarray        # float64, shape (num_range_bins,) -- mean Doppler freq
    moment2_hz: np.ndarray        # float64, shape (num_range_bins,) -- spectral width (std dev)


def _get_field(look: Any, name: str):
    """Duck-typed field access: works with an object (attribute access) or
    a dict (key access), so this module has no hard dependency on any
    particular "look" class."""
    if isinstance(look, dict):
        return look[name]
    return getattr(look, name)


class DopplerProcessor:
    """Computes per-look Doppler spectral products (SNR, first/second
    Doppler moment) from slow-time DFTs. Stateless across looks -- safe to
    call `process()` from a worker thread pulling looks off a queue and
    pushing DopplerProduct results onto another queue."""

    def __init__(self, fs_hz: float = 1562.5, tail_fraction: float = 0.25):
        """
        Args:
            fs_hz: pulse repetition rate (Hz) along the slow-time axis used
                   for the DFT frequency axis. DEFAULT/GUESS: 1562.5 Hz,
                   matching 640us/pulse from the known sender timing --
                   not measured from the look data itself.
            tail_fraction: fraction (0 < f <= 0.5) of spectrum bins on
                   EACH side (after fftshift) treated as "tail" bins for
                   noise floor estimation. DEFAULT/GUESS: 0.25.
        """
        if not (0.0 < tail_fraction <= 0.5):
            raise ValueError("tail_fraction must be in (0, 0.5]")
        self.fs_hz = fs_hz
        self.tail_fraction = tail_fraction

    def process(self, look: Any) -> DopplerProduct:
        look_angle_deg = _get_field(look, "look_angle_deg")
        timestamp_us = _get_field(look, "timestamp_us")
        complex_data = np.asarray(_get_field(look, "complex_data"))

        if complex_data.ndim != 2:
            raise ValueError(
                f"complex_data must be 2-D (range_bins, pulses), got shape "
                f"{complex_data.shape}"
            )

        num_range_bins, num_pulses = complex_data.shape

        # 1. DFT along slow-time (pulse) axis, per range bin.
        spectrum = np.fft.fftshift(
            np.fft.fft(complex_data, axis=1), axes=1
        )  # shape (num_range_bins, num_pulses)
        power = np.abs(spectrum) ** 2  # shape (num_range_bins, num_pulses)

        freq_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1.0 / self.fs_hz))

        # 2. Noise floor from tail bins (extremes of the frequency axis).
        n_tail = max(1, int(round(num_pulses * self.tail_fraction)))
        tail_idx = np.concatenate([np.arange(0, n_tail),
                                    np.arange(num_pulses - n_tail, num_pulses)])
        noise_floor = np.mean(power[:, tail_idx], axis=1)  # shape (num_range_bins,)

        # 3. Absolute power per range bin (peak spectral power, no noise
        #    reference), plus SNR relative to the noise floor.
        peak_power = np.max(power, axis=1)
        power_linear = peak_power
        with np.errstate(divide='ignore', invalid='ignore'):
            power_db = 10.0 * np.log10(np.where(power_linear > 0, power_linear, np.nan))
            snr_linear = np.where(noise_floor > 0, peak_power / noise_floor, np.inf)
            snr_db = 10.0 * np.log10(snr_linear)

        # 4. First and second Doppler moments, on noise-subtracted power.
        noise_sub_power = np.clip(power - noise_floor[:, np.newaxis], a_min=0.0, a_max=None)
        total_power = np.sum(noise_sub_power, axis=1)  # shape (num_range_bins,)

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

        return DopplerProduct(
            look_angle_deg=look_angle_deg,
            timestamp_us=timestamp_us,
            doppler_freq_hz=freq_hz,
            spectrum=spectrum,
            noise_floor=noise_floor,
            power_linear=power_linear,
            power_db=power_db,
            snr_db=snr_db,
            moment1_hz=moment1_hz,
            moment2_hz=moment2_hz,
        )


def process_look(look: Any, fs_hz: float = 1562.5,
                  tail_fraction: float = 0.25) -> DopplerProduct:
    """Convenience function wrapper around DopplerProcessor, for simple
    one-off calls without instantiating a class."""
    return DopplerProcessor(fs_hz=fs_hz, tail_fraction=tail_fraction).process(look)


if __name__ == "__main__":
    # Self-test with a synthetic look: a known Doppler tone + white noise,
    # confirming the recovered moment1 lands close to the injected tone.
    rng = np.random.default_rng(0)
    num_range_bins = 8
    num_pulses = 64
    fs_hz = 1562.5

    injected_freq_hz = 120.0  # somewhere well inside the non-tail region
    t = np.arange(num_pulses) / fs_hz
    tone = np.exp(1j * 2 * np.pi * injected_freq_hz * t)

    noise_std = 5.0
    data = np.zeros((num_range_bins, num_pulses), dtype=np.complex128)
    for rb in range(num_range_bins):
        noise = (rng.normal(0, noise_std, num_pulses)
                 + 1j * rng.normal(0, noise_std, num_pulses))
        data[rb] = 40.0 * tone + noise  # strong tone well above noise floor

    fake_look = {
        "look_angle_deg": 42.0,
        "timestamp_us": 123456789,
        "complex_data": data,
    }

    product = process_look(fake_look, fs_hz=fs_hz)
    print(f"look_angle_deg={product.look_angle_deg}")
    print(f"timestamp_us={product.timestamp_us}")
    print(f"freq axis: {product.doppler_freq_hz.min():.1f} .. "
          f"{product.doppler_freq_hz.max():.1f} Hz, {len(product.doppler_freq_hz)} bins")
    print(f"noise_floor (mean over range bins): {product.noise_floor.mean():.2f}")
    print(f"power_linear (mean over range bins): {product.power_linear.mean():.2f}")
    print(f"power_db (mean over range bins): {product.power_db.mean():.2f} dB")
    print(f"snr_db (mean over range bins): {product.snr_db.mean():.2f} dB")
    print(f"moment1_hz (mean over range bins): {product.moment1_hz.mean():.2f} Hz "
          f"(injected tone = {injected_freq_hz} Hz)")
    print(f"moment2_hz (mean over range bins): {product.moment2_hz.mean():.2f} Hz")

    assert abs(product.moment1_hz.mean() - injected_freq_hz) < 10.0, \
        "moment1 should land close to the injected tone frequency"
    print("Self-test passed: recovered Doppler moment matches injected tone.")
