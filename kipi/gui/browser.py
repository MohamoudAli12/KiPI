"""Open a local file (the user guide, an HTML report) in the default web browser."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices

log = logging.getLogger(__name__)

# Set by or for KiPI's own Python and Qt; a browser started from KiPI must not inherit
# them.
_OWN_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "VIRTUAL_ENV", "QT_API",
            "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH")


def _xdg_open(path: Path) -> bool:
    """Linux: run ``xdg-open`` ourselves. Qt hands the file to the desktop portal over
    D-Bus instead and reports success at once; where the portal runs without the
    display (a window manager started with startx, say) the browser then fails to start
    and nothing opens. ``xdg-open`` runs with KiPI's own display. True if it started."""
    exe = shutil.which("xdg-open")
    if exe is None:
        return False
    env = {k: v for k, v in os.environ.items() if k not in _OWN_ENV}
    try:
        subprocess.Popen([exe, str(path)], env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
    except OSError as e:
        log.warning("xdg-open could not start for %s: %s", path, e)
        return False
    return True


def open_in_browser(path) -> bool:
    """Ask the desktop to open ``path``; fall back to Python's webbrowser module where Qt
    cannot (some Linux desktops without a handler). True if a browser was asked to open
    it."""
    path = Path(path).resolve()
    if sys.platform.startswith("linux") and _xdg_open(path):
        return True
    if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        return True
    try:
        import webbrowser
        if webbrowser.open(path.as_uri()):
            return True
    except Exception as e:  # noqa: BLE001 - any browser failure is only logged
        log.warning("Could not open %s: %s", path, e)
    return False
