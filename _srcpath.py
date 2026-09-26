"""Let scripts in this checkout import ``fronttrack`` without installing it.

If ``fronttrack`` is already importable (for example after
``python -m pip install -e .``), this does nothing. Otherwise it prepends this
checkout's ``src/`` directory. ``src/`` contains only the ``fronttrack``
package, so no generic module names (``flux``, ``solver``, ...) leak onto the
import path. Import it first, before any ``fronttrack`` import.
"""
import importlib.util
from pathlib import Path
import sys

if importlib.util.find_spec("fronttrack") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
