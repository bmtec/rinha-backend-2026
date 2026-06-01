#!/usr/bin/env python3
"""Replay the official labeled test-data through a running API instance and
report detection accuracy using the same TP/TN/FP/FN/error definitions as the
k6 scorer in test.js.

Usage: accuracy.py <base_url> [max_entries]
  e.g. accuracy.py http://127.0.0.1:9998
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from threading import local

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9998"
host, port = BASE.replace("http://", "").split(":")
port = int(port)

with open("test/test-data.json") as f:
    data = json.load(f)
entries = data["entries"]
if len(sys.argv) > 2:
    entries = entries[: int(sys.argv[2])]

_tl = local()


def conn():
    c = getattr(_tl, "c", None)
    if c is None:
        c = HTTPConnection(host, port, timeout=10)
        _tl.c = c
    return c


def one(entry):
    body = json.dumps(entry["request"])
    expected = entry["expected_approved"]
    c = conn()
    try:
        c.request("POST", "/fraud-score", body,
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        if r.status != 200:
            return ("err", None)
        approved = json.loads(raw)["approved"]
    except Exception:
        try:
            _tl.c = None
        except Exception:
            pass
        return ("err", None)
    # expected True == legit, False == fraud
    if expected == approved:
        return ("tn" if approved else "tp", None)
    return ("fn" if approved else "fp", None)


t0 = time.time()
tally = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "err": 0}
with ThreadPoolExecutor(max_workers=24) as ex:
    for kind, _ in ex.map(one, entries):
        tally[kind] += 1
dt = time.time() - t0

n = sum(tally.values())
tp, tn, fp, fn, errs = tally["tp"], tally["tn"], tally["fp"], tally["fn"], tally["err"]
E = fp * 1 + fn * 3 + errs * 5
failures = fp + fn + errs
epsilon = E / n if n else 0
acc = (tp + tn) / n if n else 0

print(f"\n  entries        : {n}")
print(f"  elapsed        : {dt:.1f}s ({n/dt:.0f} req/s)")
print(f"  TP (deny fraud): {tp}")
print(f"  TN (ok legit)  : {tn}")
print(f"  FP (deny legit): {fp}   weight 1")
print(f"  FN (miss fraud): {fn}   weight 3")
print(f"  errors (non200): {errs}   weight 5")
print(f"  failures       : {failures}  ({100*failures/n:.3f}%)")
print(f"  weighted E     : {E}")
print(f"  epsilon (E/N)  : {epsilon:.5f}")
print(f"  accuracy       : {100*acc:.3f}%")
