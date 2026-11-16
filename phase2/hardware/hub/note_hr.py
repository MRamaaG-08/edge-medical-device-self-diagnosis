"""Type the reference heart rate during a recording (second terminal window).

If anyone at home has a fingertip pulse oximeter, clip it on the OTHER hand and type
its reading every 30 seconds. No oximeter: count the wrist pulse for 15 s and
multiply by 4 (rougher, say so in the session log).

    python -m phase2.hardware.hub.note_hr --session S01

Each number you type is saved with the time to data/sessions/S01_refhr.csv.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True)
ap.add_argument("--source", default="oximeter", choices=["oximeter", "manual_count"])
a = ap.parse_args()
os.makedirs("data/sessions", exist_ok=True)
path = os.path.join("data/sessions", f"{a.session}_refhr.csv")
new = not os.path.exists(path)
with open(path, "a") as fh:
    if new:
        fh.write("wall_time,hr_bpm,source\n")
    print("Type a heart rate and press Enter. Type q to stop.")
    while True:
        s = input("HR> ").strip()
        if s.lower() == "q":
            break
        try:
            fh.write(f"{time.time():.2f},{float(s):.1f},{a.source}\n"); fh.flush()
        except ValueError:
            print("numbers only, please")
print(f"saved {path}")
