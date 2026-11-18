"""Make the secret key the ESP32 and the Pi share for signing packets.

    python -m phase2.hardware.hub.make_key

Writes secrets/node_key.txt (64 hex characters) for the hub, and the same key as
phase2/hardware/firmware/sensor_node/node_key.h for the ESP32 sketch. Both are in
.gitignore: the key must never reach GitHub or a chat. Copy secrets/node_key.txt to
the Pi with scp.
"""
import os
import secrets

os.makedirs("secrets", exist_ok=True)
path = os.path.join("secrets", "node_key.txt")
if os.path.exists(path):
    print(f"{path} already exists; not overwriting. Delete it first if you really want a new key.")
else:
    open(path, "w").write(secrets.token_hex(32))
    print(f"wrote {path}")
key = open(path).read().strip()
hdr = os.path.join("phase2", "hardware", "firmware", "sensor_node", "node_key.h")
if os.path.isdir(os.path.dirname(hdr)):
    open(hdr, "w").write("// made by make_key.py -- never commit this file\n"
                         f'const char NODE_KEY[] = "{key}";\n')
    print(f"wrote {hdr} (the same key, for the ESP32 sketch)")
