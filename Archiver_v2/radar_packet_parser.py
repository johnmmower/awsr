#!/usr/bin/env python3
"""
Receiver-side parser for the synthetic radar UDP stream produced by
generate_synthetic_radar.py.

Packet format (matches HEADER_STRUCT in the sender):
    counter          : uint64 LE
    timestamp_us     : uint64 LE
    antenna_index    : uint16 LE
    pa_enable        : uint16 LE
    reserved         : uint32 LE (always 0)
    encoder_position : float64 LE, degrees
    samples          : 2048 x int32 LE
                        samples[:1024]  = real part (range bins 0..1023)
                        samples[1024:]  = imag part (range bins 0..1023)
"""

import socket
import struct
from dataclasses import dataclass

import numpy as np

HEADER_STRUCT = struct.Struct('<QQHHId')
HEADER_SIZE = HEADER_STRUCT.size  # 32 bytes
NUM_RANGE_BINS = 1024
SAMPLES_COUNT = 2 * NUM_RANGE_BINS  # 2048 int32 values
SAMPLES_STRUCT = struct.Struct(f'<{SAMPLES_COUNT}i')
PAYLOAD_SIZE = SAMPLES_STRUCT.size  # 8192 bytes
PACKET_SIZE = HEADER_SIZE + PAYLOAD_SIZE

# Antenna look-angle geometry -- copied directly from generate_synthetic_radar.py
# (ANTENNA_OFFSETS_DEG), confirmed with the user for that sender script.
# anglea = (encoder_position_deg + antenna_offset) % 360
ANTENNA_OFFSETS_DEG = {0: 180.0, 1: 90.0, 2: 0.0, 3: -90.0}


@dataclass
class RadarPacket:
    counter: int
    timestamp_us: int
    antenna_index: int
    pa_enable: int
    reserved: int
    encoder_position_deg: float
    range_profile: np.ndarray  # complex128, shape (NUM_RANGE_BINS,)


@dataclass
class RadarLook:
    """A completed, coherent 64-pulse look: one antenna, contiguous counters.

    look_angle_deg : the antenna's actual pointing angle for this look,
                      computed as (encoder_position_deg + antenna_offset) % 360
                      using ANTENNA_OFFSETS_DEG, matching generate_synthetic_radar.py.
                      GUESS/DESIGN CHOICE: uses the encoder value from the
                      FIRST packet in the look, matching the sender's own
                      behavior (it computes anglea once per look, before any
                      encoder updates that occur during that look).
    timestamp_us   : timestamp_us of the first packet in the look (start-of-look
                      time). DESIGN CHOICE: not an average or per-pulse array --
                      just the look's start time.
    complex_data   : complex128 array, shape (NUM_RANGE_BINS, block_size),
                      one column per pulse in arrival order (column 0 = first
                      pulse of the look).
    """
    look_angle_deg: float
    timestamp_us: int
    complex_data: np.ndarray


class RadarPacketParser:
    """Parses raw UDP payloads into RadarPacket objects, and can optionally
    listen on a UDP socket and yield packets as they arrive."""

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: if True, raise ValueError on malformed packets (wrong
                    size, non-zero reserved field unexpectedly, etc).
                    If False, best-effort parse and skip validation.
        """
        self.strict = strict

    def parse(self, data: bytes) -> RadarPacket:
        """Parse a single raw UDP payload into a RadarPacket."""
        if self.strict and len(data) != PACKET_SIZE:
            raise ValueError(
                f"unexpected packet size: got {len(data)} bytes, "
                f"expected {PACKET_SIZE} (header={HEADER_SIZE} + "
                f"payload={PAYLOAD_SIZE})"
            )

        header_bytes = data[:HEADER_SIZE]
        sample_bytes = data[HEADER_SIZE:HEADER_SIZE + PAYLOAD_SIZE]

        (counter, timestamp_us, antenna_index, pa_enable,
         reserved, encoder_position_deg) = HEADER_STRUCT.unpack(header_bytes)

        samples = np.frombuffer(sample_bytes, dtype='<i4')
        if len(samples) != SAMPLES_COUNT:
            if self.strict:
                raise ValueError(
                    f"unexpected sample count: got {len(samples)}, "
                    f"expected {SAMPLES_COUNT}"
                )
            samples = np.pad(samples, (0, SAMPLES_COUNT - len(samples)))

        real = samples[:NUM_RANGE_BINS].astype(np.float64)
        imag = samples[NUM_RANGE_BINS:].astype(np.float64)
        range_profile = real + 1j * imag

        return RadarPacket(
            counter=counter,
            timestamp_us=timestamp_us,
            antenna_index=antenna_index,
            pa_enable=pa_enable,
            reserved=reserved,
            encoder_position_deg=encoder_position_deg,
            range_profile=range_profile,
        )

    def listen(self, bind_ip: str = "0.0.0.0", bind_port: int = 34952,
               recv_bufsize: int = 65536):
        """Bind a UDP socket and yield parsed RadarPacket objects forever.

        Usage:
            parser = RadarPacketParser()
            for pkt in parser.listen(bind_port=34952):
                print(pkt.counter, pkt.antenna_index, pkt.pa_enable)
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((bind_ip, bind_port))
        try:
            while True:
                data, _addr = sock.recvfrom(recv_bufsize)
                try:
                    yield self.parse(data)
                except ValueError as e:
                    print(f"[RadarPacketParser] dropping malformed packet: {e}")
        finally:
            sock.close()


class RadarBlockParser(RadarPacketParser):
    """Extends RadarPacketParser to group incoming packets into fixed-size
    blocks of BLOCK_SIZE consecutive packets (default 64, matching one
    antenna "look"), based on the `counter` field AND `antenna_index`.

    A block can start at any counter value -- it does NOT wait for a
    look-boundary-aligned counter. It accumulates packets as long as BOTH:
      1. counter increments by exactly COUNTER_STEP (16) from the previous
         packet in the block, with correct wraparound at 65536, AND
      2. antenna_index is identical to the first packet in the block.

    This matters for coherent processing (e.g. Doppler/clutter estimation
    across a look) -- all 64 pulses in a block must come from the same
    antenna, or the block is meaningless for that purpose.

    If either condition fails (a counter gap, OR an antenna_index change
    mid-block), the in-progress block is dropped entirely (not yielded)
    and a new block starts accumulating from that packet.

    Output is a RadarLook: look_angle_deg, timestamp_us, and complex_data
    (the stacked range profiles) -- not the raw RadarPacket list.

    Usage:
        parser = RadarBlockParser()
        for look in parser.listen_blocks(bind_port=34952):
            look.look_angle_deg   # float, degrees
            look.timestamp_us     # int, start-of-look time
            look.complex_data     # complex128, shape (1024, 64)
    """

    BLOCK_SIZE = 64
    COUNTER_STEP = 16
    COUNTER_WRAP = 65536

    def __init__(self, strict: bool = True, block_size: int = None):
        super().__init__(strict=strict)
        self.block_size = block_size or self.BLOCK_SIZE
        self._buffer = []
        self._expected_counter = None
        self._block_antenna = None
        self._dropped_blocks = 0
        self._dropped_packets = 0

    def _expected_next(self, counter: int) -> int:
        return (counter + self.COUNTER_STEP) % self.COUNTER_WRAP

    def _start_new_block(self, pkt: RadarPacket):
        self._buffer = [pkt]
        self._expected_counter = self._expected_next(pkt.counter)
        self._block_antenna = pkt.antenna_index

    def _build_look(self, block) -> RadarLook:
        first = block[0]
        if first.antenna_index not in ANTENNA_OFFSETS_DEG:
            raise ValueError(
                f"unknown antenna_index {first.antenna_index}, expected one "
                f"of {sorted(ANTENNA_OFFSETS_DEG)}"
            )
        offset = ANTENNA_OFFSETS_DEG[first.antenna_index]
        look_angle_deg = (first.encoder_position_deg + offset) % 360.0
        complex_data = np.stack([p.range_profile for p in block], axis=1)
        return RadarLook(
            look_angle_deg=look_angle_deg,
            timestamp_us=first.timestamp_us,
            complex_data=complex_data,
        )

    def feed(self, pkt: RadarPacket):
        """Feed one parsed RadarPacket in. Returns a completed RadarLook if
        this packet completes one, else None."""
        if not self._buffer:
            self._start_new_block(pkt)
            return None

        counter_ok = (pkt.counter == self._expected_counter)
        antenna_ok = (pkt.antenna_index == self._block_antenna)

        if not (counter_ok and antenna_ok):
            # Gap or antenna change detected -- drop the in-progress
            # (incomplete/incoherent) block and restart at this packet.
            self._dropped_packets += len(self._buffer)
            self._dropped_blocks += 1
            self._start_new_block(pkt)
            return None

        self._buffer.append(pkt)
        self._expected_counter = self._expected_next(pkt.counter)

        if len(self._buffer) == self.block_size:
            block = self._buffer
            self._buffer = []
            self._expected_counter = None
            self._block_antenna = None
            return self._build_look(block)

        return None

    def listen_blocks(self, bind_ip: str = "0.0.0.0", bind_port: int = 34952,
                       recv_bufsize: int = 65536):
        """Bind a UDP socket and yield complete, contiguous, single-antenna
        looks as RadarLook objects. Incomplete or mixed-antenna blocks are
        silently discarded -- only full, coherent looks are yielded."""
        for pkt in self.listen(bind_ip, bind_port, recv_bufsize):
            look = self.feed(pkt)
            if look is not None:
                yield look


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--bind-port", type=int, default=34952)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--blocks", action="store_true",
                     help="group packets into contiguous 64-packet blocks "
                          "and drop incomplete ones (RadarBlockParser)")
    args = ap.parse_args()

    if args.blocks:
        block_parser = RadarBlockParser()
        look_count = 0
        for look in block_parser.listen_blocks(args.bind_ip, args.bind_port):
            look_count += 1
            if look_count % max(1, args.print_every // 64) == 0:
                mean_power = np.mean(np.abs(look.complex_data) ** 2)
                print(f"look#{look_count} angle={look.look_angle_deg:.2f}deg "
                      f"t_us={look.timestamp_us} "
                      f"shape={look.complex_data.shape} "
                      f"mean|x|^2={mean_power:.1f} "
                      f"dropped_blocks_so_far={block_parser._dropped_blocks}")
    else:
        parser = RadarPacketParser()
        count = 0
        for pkt in parser.listen(args.bind_ip, args.bind_port):
            count += 1
            if count % args.print_every == 0:
                print(f"pkt#{count} counter={pkt.counter} "
                      f"antenna={pkt.antenna_index} pa_enable={pkt.pa_enable} "
                      f"encoder={pkt.encoder_position_deg:.2f} "
                      f"mean|x|^2={np.mean(np.abs(pkt.range_profile)**2):.1f}")
