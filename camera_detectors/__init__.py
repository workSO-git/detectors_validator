import sys
from pathlib import Path

# Ensure camera_detectors directory is in sys.path so internal imports like
# `import analyze_photos_v3` work cleanly.
pkg_dir = str(Path(__file__).parent)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
