#!/usr/bin/env python3
"""Compatibility shim for the legacy root-level ColorCast launcher.

This script is preserved only for backward compatibility. It launches the
graphical interface via ``colorcast.__main__:gui_main``. New code should use
``python -m colorcast`` or the ``colorcast-gui`` console script instead.

This shim may be removed in a future release.
"""

import warnings

warnings.warn(
    "Running the root colorcast.py script directly is deprecated. "
    "Use 'python -m colorcast' or the 'colorcast-gui' console script instead.",
    DeprecationWarning,
    stacklevel=2,
)

from colorcast.__main__ import gui_main

if __name__ == "__main__":
    gui_main()
