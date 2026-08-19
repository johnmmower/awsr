#!/usr/bin/env python3
"""
Generate and continuously stream SYNTHETIC radar UDP data matching the
header/payload format reverse-engineered from radar_dump.pcapng.

This does NOT replay the real capture -- it generates fake data forever,
following a physical model built up and confirmed with the user over several
turns. Everything below is either:
  (a) taken directly from the real capture's confirmed header format, or
  (b) a design choice / tunable parameter for the synthetic model, marked
      GUESS/DEFAULT where the user didn't specify an exact value.

=== Confirmed header format (from parse_radar_pcapng.py) ===
  counter        : uint64 LE, +16 per packet, wraps at 65536
  timestamp_us   : uint64 LE, epoch microseconds. Per user: FIXED time-step
                   clock (+640 us per packet exactly), not wall-clock --
                   intentionally allowed to drift from real elapsed time.
  antenna_index  : uint16 LE, cycles 0,1,2,3, held for 64 packets each (one
                   "look" per antenna)
  pa_enable      : uint16 LE, 0 or 1 (see PA logic below)
  reserved       : uint32 LE, always 0
  encoder_position : float64 LE, degrees. Per user: encoder rotates at
                   ROTATION_RATE_DEG_PER_S, range -180..+180 wrapping (matches
                   the real capture's ~-86.8 starting value and slow drift).
                   CONFIRMED: real capture updates this field every 8
                   packets exactly (measured run-length, not a guess) --
                   matched here via ENCODER_UPDATE_PERIOD_PACKETS = 8.
  samples        : 2048x int32 LE, block format (samples[:1024]=real,
                   samples[1024:]=imag) -- CONFIRMED format from prior
                   analysis.

=== Antenna look-angle geometry (from user) ===
  anglea = (encoder_angle + antenna_offset) % 360
  offsets: {0: +180, 1: +90, 2: 0, 3: -90}

=== PA enable logic (from user, for testing) ===
  pa_enable = 0 if anglea (mod 360) is in [90, 180], else 1
  This is evaluated once per look (64 packets), using the look's angle.

=== Doppler / sea clutter model (from user) ===
  Wind direction: 45 deg (fixed for the whole run), same angular convention
  as anglea/encoder.
  Bragg scatterer radial speed: 3 m/s toward/away from radar, projected via
  cos(anglea - wind_dir).
  Carrier: 9400 MHz -> wavelength = c/f = 0.03191 m
  f_doppler = 2 * v_radial / wavelength  (Hz)
  Each 64-packet look gets a FRESH random draw (independent of all other
  looks) of a complex clutter process per range bin, generated as an AR(1)
  recursion:
      x[n] = pole * exp(j*2*pi*f_d/fs) * x[n-1] + w[n]
  where fs = 1/640us = 1562.5 Hz, and pole sets the pulse-to-pulse
  correlation (coherence time). This gives a slow, narrow, mostly-DC-ish
  spectral line at f_doppler riding on the noise floor, matching "some
  doppler, but not much."

  GUESS/DEFAULT (not specified by user): coherence time = 5 ms (~8 pulses
  at 640us spacing) -> AR(1) pole = exp(-dt/coherence_time). This is a
  tunable constant (COHERENCE_TIME_S below), not a measured value.

  Noise floor target power (E[|x|^2] per range bin) = ~2300, matching the
  measured real/imag std of ~34 in the real capture's pa_enable=0 regions.
  All 1024 range bins get independent noise realizations (white in range).

=== Close-in return (from user) ===
  "Basically what it is currently" and "fully fixed for every pulse, don't
  care about variations in amp" -- so this is a hardcoded, static template
  taken directly from one real captured pulse (antenna 1, pa_enable=1,
  first 20 range bins, block format), applied identically to every pulse
  whenever pa_enable=1, overwriting the clutter in those bins. No look-to-
  look or antenna-to-antenna variation.
"""

import argparse
import socket
import struct
import time

import numpy as np

# ---------------------------------------------------------------------------
# Confirmed constants (from real capture / user-specified radar config)
# ---------------------------------------------------------------------------
PRF_HZ = 25000
INTEGRATION_FACTOR = 16
TIMESTEP_US = INTEGRATION_FACTOR / PRF_HZ * 1e6  # = 640.0 exactly
FS_LOOK_HZ = 1e6 / TIMESTEP_US                   # per-look pulse rate, 1562.5 Hz

LOOK_SIZE = 64            # packets per antenna dwell / per look
NUM_RANGE_BINS = 1024

ANTENNA_OFFSETS_DEG = {0: 180.0, 1: 90.0, 2: 0.0, 3: -90.0}
ANTENNA_ORDER = [0, 1, 2, 3]

ROTATION_RATE_DEG_PER_S = 5.0
ENCODER_UPDATE_PERIOD_PACKETS = 8  # confirmed from real capture (measured, not guessed)
ENCODER_STEP_PER_UPDATE_DEG = (ROTATION_RATE_DEG_PER_S
                                * (TIMESTEP_US / 1e6) * ENCODER_UPDATE_PERIOD_PACKETS)

WIND_DIR_DEG = 45.0
BRAGG_SPEED_MPS = 3.0
CARRIER_HZ = 9.4e9
WAVELENGTH_M = 3e8 / CARRIER_HZ
DOPPLER_SCALE_HZ_PER_MPS = 2.0 / WAVELENGTH_M

# --- GUESS/DEFAULT: sea clutter model tuning, not measured from data ---
NOISE_FLOOR_POWER = 2300.0     # E[|x|^2] per range bin, matches measured ~34 std
COHERENCE_TIME_S = 0.005       # 5ms default coherence time -- TUNABLE
AR_POLE_MAG = np.exp(-(TIMESTEP_US * 1e-6) / COHERENCE_TIME_S)

# PA-disable test window
PA_DISABLE_ANGLE_LO = 90.0
PA_DISABLE_ANGLE_HI = 180.0

# --- Close-in return: hardcoded static template from real capture ---
# antenna=1, pa_enable=1, first record, block-format bins 0-19 (real, imag)
CLOSE_IN_REAL = np.array([
    7855, 47717, 380411, 201498, -85826, 7214, -32523, 6642, -22759, -6264,
    -4459, 1304, 4834, 3268, 1559, -295, 5539, 1715, -2571, -987,
], dtype=np.int64)
CLOSE_IN_IMAG = np.array([
    -18995, 41775, 30058, -152464, -261, -34741, -10862, 5991, -13637, 11184,
    486, 4899, -2555, 2344, -1608, 89, -2756, -4950, 2184, 243,
], dtype=np.int64)
CLOSE_IN_LEN = len(CLOSE_IN_REAL)
CLOSE_IN_COMPLEX = CLOSE_IN_REAL.astype(np.float64) + 1j * CLOSE_IN_IMAG.astype(np.float64)

HEADER_STRUCT = struct.Struct('<QQHHId')
assert HEADER_STRUCT.size == 32
INT32_MAX = 2_147_483_647
INT32_MIN = -2_147_483_648


def pa_enable_for_angle(anglea_deg: float) -> int:
    a = anglea_deg % 360.0
    if PA_DISABLE_ANGLE_LO <= a <= PA_DISABLE_ANGLE_HI:
        return 0
    return 1


def generate_look(anglea_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Generate one (NUM_RANGE_BINS, LOOK_SIZE) complex clutter array for a
    single look, with a fresh random draw and Doppler line set by anglea."""
    v_radial = BRAGG_SPEED_MPS * np.cos(np.radians(anglea_deg - WIND_DIR_DEG))
    f_d = DOPPLER_SCALE_HZ_PER_MPS * v_radial
    phase_step = np.exp(1j * 2 * np.pi * f_d / FS_LOOK_HZ)

    pole = AR_POLE_MAG
    sigma_w2 = NOISE_FLOOR_POWER * (1 - pole ** 2)
    w_std = np.sqrt(sigma_w2 / 2)
    init_std = np.sqrt(NOISE_FLOOR_POWER / 2)

    x = np.empty((NUM_RANGE_BINS, LOOK_SIZE), dtype=np.complex128)
    x[:, 0] = (rng.normal(0, init_std, NUM_RANGE_BINS)
               + 1j * rng.normal(0, init_std, NUM_RANGE_BINS))
    for n in range(1, LOOK_SIZE):
        w = (rng.normal(0, w_std, NUM_RANGE_BINS)
             + 1j * rng.normal(0, w_std, NUM_RANGE_BINS))
        x[:, n] = pole * phase_step * x[:, n - 1] + w
    return x


def pulse_to_int32_samples(pulse: np.ndarray, pa_enable: int) -> np.ndarray:
    """Convert one pulse's complex range profile into the 2048x int32
    block-format sample array (real half, imag half), applying the static
    close-in return when pa_enable is set."""
    pulse = pulse.copy()
    if pa_enable:
        pulse[:CLOSE_IN_LEN] = CLOSE_IN_COMPLEX

    real = np.clip(np.round(pulse.real), INT32_MIN, INT32_MAX).astype('<i4')
    imag = np.clip(np.round(pulse.imag), INT32_MIN, INT32_MAX).astype('<i4')
    return np.concatenate([real, imag])


def stream(dest_ip: str, dest_port: int, seed: int, start_encoder_deg: float,
           progress_every: int = 1000):
    rng = np.random.default_rng(seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (dest_ip, dest_port)

    interval_s = TIMESTEP_US / 1e6
    counter = 0
    timestamp_us = int(time.time() * 1e6)  # arbitrary fixed starting reference
    encoder_deg = start_encoder_deg
    antenna_slot = 0  # index into ANTENNA_ORDER

    print(f"streaming synthetic radar data -> {dest_ip}:{dest_port}, "
          f"interval={TIMESTEP_US:.1f}us, wavelength={WAVELENGTH_M*100:.2f}cm, "
          f"AR pole={AR_POLE_MAG:.4f} (coherence={COHERENCE_TIME_S*1000:.1f}ms)")

    next_time = time.perf_counter()
    packet_total = 0

    try:
        while True:
            antenna_index = ANTENNA_ORDER[antenna_slot % len(ANTENNA_ORDER)]
            antenna_slot += 1

            anglea = (encoder_deg + ANTENNA_OFFSETS_DEG[antenna_index]) % 360.0
            pa_enable = pa_enable_for_angle(anglea)

            look = generate_look(anglea, rng)

            for n in range(LOOK_SIZE):
                samples = pulse_to_int32_samples(look[:, n], pa_enable)
                header = HEADER_STRUCT.pack(
                    counter, timestamp_us, antenna_index, pa_enable, 0,
                    encoder_deg,
                )
                payload = header + samples.tobytes()

                now = time.perf_counter()
                if next_time > now:
                    time.sleep(next_time - now)
                sock.sendto(payload, addr)
                next_time += interval_s

                counter = (counter + INTEGRATION_FACTOR) % 65536
                timestamp_us += int(TIMESTEP_US)
                packet_total += 1
                if packet_total % ENCODER_UPDATE_PERIOD_PACKETS == 0:
                    encoder_deg += ENCODER_STEP_PER_UPDATE_DEG
                    if encoder_deg > 180.0:
                        encoder_deg -= 360.0
                    elif encoder_deg < -180.0:
                        encoder_deg += 360.0

                if progress_every and packet_total % progress_every == 0:
                    print(f"  sent {packet_total} packets "
                          f"(antenna={antenna_index} pa_enable={pa_enable} "
                          f"encoder={encoder_deg:.2f} anglea={anglea:.2f})")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest-ip", default="192.168.88.50")
    ap.add_argument("--dest-port", type=int, default=34952)
    ap.add_argument("--seed", type=int, default=None,
                     help="RNG seed for reproducibility (default: random)")
    ap.add_argument("--start-encoder-deg", type=float, default=-86.85,
                     help="initial encoder angle, matches real capture start")
    args = ap.parse_args()

    stream(args.dest_ip, args.dest_port, args.seed, args.start_encoder_deg)


if __name__ == "__main__":
    main()
