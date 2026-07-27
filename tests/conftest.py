import os
import pathlib
import sys

# Make the shared test builders importable as a top-level module.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# The on-disk record-and-replay caches (runcache.py) must never cross test
# boundaries: a source one test "fetched" replayed into another test would bypass the
# exact refusal paths the tests exist to exercise. Tests that test the caches
# themselves re-enable them against a tmp_path via monkeypatch.
os.environ.setdefault("SWORLDMODEL_CACHE", "off")
