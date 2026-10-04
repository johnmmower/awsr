#!/usr/bin/env python3
"""
Log live AIS from aisstream.io inside the AWSR field of view.

- Subscribes with a bounding box around awsr_fov.geojson (tiny area -> tiny traffic,
  fine for the cell link).
- Keeps position reports only if inside the FOV polygon; keeps all static/voyage
  messages (names, dimensions) for vessels in the box.
- Writes one JSON line per message to <outdir>/ais_YYYYMMDD.jsonl (UTC day),
  stamped with the NUC receive time so it shares a clock with the radar archive.
- Reconnects forever with exponential backoff + jitter.

Setup:
  pip install websockets
  put the key (from https://aisstream.io/account) in aisstream.io.key, chmod 600
  python ais_logger.py --keyfile aisstream.io.key --fov awsr_fov.geojson --outdir ~/ais_log
  (falls back to $AISSTREAM_API_KEY if the key file is missing)
"""
import argparse, asyncio, json, os, random, sys, time, datetime as dt
import websockets

URL = "wss://stream.aisstream.io/v0/stream"
POS_TYPES = ["PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"]
STATIC_TYPES = ["ShipStaticData", "StaticDataReport"]

def load_ring(fn):
    ring = json.load(open(fn))["features"][0]["geometry"]["coordinates"][0]  # [lon, lat]
    lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
    pad = 0.002  # ~200 m margin on the subscription box only
    bbox = [[max(lats) + pad, min(lons) - pad], [min(lats) - pad, max(lons) + pad]]
    return ring, bbox

def inside(lon, lat, ring):
    """Ray-casting point-in-polygon."""
    c = False
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                c = not c
    return c

class DailyWriter:
    def __init__(self, outdir):
        self.outdir = os.path.expanduser(outdir); os.makedirs(self.outdir, exist_ok=True)
        self.day, self.fh = None, None
    def write(self, rec):
        day = dt.datetime.utcnow().strftime("%Y%m%d")
        if day != self.day:
            if self.fh: self.fh.close()
            self.fh = open(os.path.join(self.outdir, f"ais_{day}.jsonl"), "a", buffering=1)
            self.day = day
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

def read_key(fn):
    fn = os.path.expanduser(fn)
    if os.path.isfile(fn):
        key = open(fn).read().strip()
        if key:
            return key
        sys.exit(f"key file {fn} is empty")
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    if key:
        return key
    sys.exit(f"no key: {fn} not found and AISSTREAM_API_KEY not set")

async def run(args):
    key = read_key(args.keyfile)
    ring, bbox = load_ring(args.fov)
    sub = {"APIKey": key, "BoundingBoxes": [bbox], "FilterMessageTypes": POS_TYPES + STATIC_TYPES}
    w = DailyWriter(args.outdir)
    backoff = 1
    while True:
        try:
            # websockets negotiates permessage-deflate by default; aisstream bandwidth-limits
            # uncompressed connections from Sept 2026.
            async with websockets.connect(URL, compression="deflate", ping_interval=20, ping_timeout=20,
                                          max_size=2**22) as ws:
                await ws.send(json.dumps(sub))
                print(f"{dt.datetime.utcnow():%F %T} connected, bbox={bbox}", flush=True)
                backoff = 1
                async for raw in ws:
                    t_rx = time.time()
                    msg = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    mt = msg.get("MessageType")
                    if mt == "SubscriptionConfirmation":
                        print("subscription confirmed:", msg.get("Message"), flush=True); continue
                    if mt in POS_TYPES:
                        body = msg["Message"][mt]
                        lat, lon = body.get("Latitude"), body.get("Longitude")
                        if lat is None or lon is None or not inside(lon, lat, ring):
                            continue
                    elif mt not in STATIC_TYPES:
                        continue
                    w.write({"t_rx": t_rx, **msg})
        except Exception as e:
            wait = min(backoff, 300) * (0.5 + random.random())
            print(f"{dt.datetime.utcnow():%F %T} disconnected ({e!r}); retry in {wait:.0f}s", flush=True)
            await asyncio.sleep(wait); backoff *= 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", default="awsr_fov.geojson")
    ap.add_argument("--outdir", default="~/ais_log")
    ap.add_argument("--keyfile", default="aisstream.io.key",
                    help="file containing the aisstream.io API key (relative paths resolve from cwd)")
    asyncio.run(run(ap.parse_args()))
