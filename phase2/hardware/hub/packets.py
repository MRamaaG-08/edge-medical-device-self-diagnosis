"""Signed packets between the ESP32 and the Pi.

Each line the ESP32 sends looks like:   {json payload} <64 hex characters>
The hex string is HMAC-SHA256(payload, shared key). A packet whose tag does not
match, or whose sequence number is not larger than the last one, is rejected.
"""
import hashlib
import hmac
import json

KEY_FILE = "secrets/node_key.txt"     # never committed: secrets/ is in .gitignore
#                                       make one with:  python -m phase2.hardware.hub.make_key


def load_key(path=KEY_FILE):
    return open(path).read().strip().encode()


def sign(payload: str, key: bytes) -> str:
    return payload + " " + hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


class Verifier:
    def __init__(self, key: bytes):
        self.key = key
        self.last_seq = -1
        self.rejected = 0

    def check(self, line: str):
        """Return the packet as a dict, or None if it is tampered or replayed."""
        line = line.strip()
        if " " not in line:
            self.rejected += 1
            return None
        payload, tag = line.rsplit(" ", 1)
        good = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, tag):
            self.rejected += 1
            return None
        pkt = json.loads(payload)
        if pkt["seq"] <= self.last_seq:          # replayed or out of order
            self.rejected += 1
            return None
        self.last_seq = pkt["seq"]
        return pkt
