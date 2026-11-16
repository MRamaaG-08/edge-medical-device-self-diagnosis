"""Encrypted session storage on the Pi (AES-GCM). Raw signals never sit on disk in plain text."""
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_FILE = Path("secrets/storage_key.bin")


def storage_key():
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(exist_ok=True)
        KEY_FILE.write_bytes(AESGCM.generate_key(bit_length=256))
    return KEY_FILE.read_bytes()


def save_encrypted(records, path):
    aes, nonce = AESGCM(storage_key()), os.urandom(12)
    data = json.dumps(records).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(nonce + aes.encrypt(nonce, data, None))


def load_encrypted(path):
    blob = Path(path).read_bytes()
    return json.loads(AESGCM(storage_key()).decrypt(blob[:12], blob[12:], None))


if __name__ == "__main__":
    # python -m phase2.hardware.hub.store data/sessions/S01.jsonl   -> S01.jsonl.enc, then you may delete the plain file
    import sys
    src = sys.argv[1]
    rows = [json.loads(x) for x in open(src) if x.strip()]
    save_encrypted(rows, src + ".enc")
    print(f"encrypted {len(rows)} windows into {src}.enc (key in secrets/storage_key.bin)")
