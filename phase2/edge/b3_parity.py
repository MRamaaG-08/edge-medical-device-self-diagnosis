"""B3. Compile the C code on your laptop and prove it matches Python.

Needs gcc (on Windows: MSYS2 UCRT64, see the handbook). No board needed.

    python -m phase2.edge.b3_parity

Writes results/phase2/b3_parity.txt (the full printout, for your handover)
"""
import os
import shutil
import subprocess
import sys

import config
from phase2.common import ensure_dir

HERE = os.path.dirname(os.path.abspath(__file__))
gcc = shutil.which("gcc")
if not gcc:
    sys.exit("gcc not found. Install MSYS2 (handbook section 'Install a C compiler') and open a NEW terminal.")
exe = os.path.join(HERE, "host_test", "parity.exe" if os.name == "nt" else "parity")
srcs = [os.path.join(HERE, "host_test", "parity.c")] + [
    os.path.join(HERE, "c", f) for f in ("edge_frontend.c", "edge_features.c", "edge_model.c")]
cmd = [gcc, "-O2", "-I" + os.path.join(HERE, "c")] + srcs + ["-lm", "-o", exe]
print(f"compiling the C code with {gcc} ...")
subprocess.run(cmd, check=True)
r = subprocess.run([exe], cwd=config.ROOT, capture_output=True, text=True)
print(r.stdout)
out = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "b3_parity.txt")
open(out, "w").write(r.stdout)
print(f"saved {out}")
sys.exit(r.returncode)
