# notebooks/_bootstrap.py  (create once and run in every nb)
import sys, pathlib, importlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# optional quality-of-life
importlib.reload(importlib)  # enables autoreload magic later