"""How the setup window looks: dark or light, as KiPI's own windows are, and how much to
enlarge it on a high-resolution display. Kept free of Tk so it can be tested anywhere.

Dark or light follows KiPI's Dark mode button (``appearance/theme`` in KiPI's settings,
which Qt keeps in a file, the registry or a plist), dark if never chosen, as in KiPI.

The scale comes from, first found:

* ``KIPI_UI_SCALE`` (for example ``2``), to set it by hand;
* the desktop's own scale: ``Xft.dpi``, ``QT_FONT_DPI``, ``QT_SCALE_FACTOR``,
  ``GDK_SCALE`` × ``GDK_DPI_SCALE`` (Linux);
* the DPI Tk was told by the system (Windows, once the process is DPI aware);
* else, where X says 96 DPI as it does when nobody set it, the screen's height against a
  1080-line screen: twice as large on a 4K screen (not on Windows, where 96 DPI is the
  100% scaling chosen in its settings).

macOS draws Tk in points already, so there it stays 1.
"""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

DARK, LIGHT = "dark", "light"
DEFAULT = DARK                  # as KiPI's kipi.gui.theme
SCALE_ENV = "KIPI_UI_SCALE"
MAX_SCALE = 4.0

#: The colours of KiPI's dark palette (kipi.gui.theme.dark_palette), and the light ones
#: Tk's default theme does not set.
COLORS = {
    DARK: {"window": "#2b2b2b", "base": "#1e1e1e", "text": "#e0e0e0", "dim": "#9a9a9a",
           "button": "#353535", "active": "#424242", "border": "#555555",
           "accent": "#2f7fd0", "accent_text": "#ffffff", "disabled": "#8a8a8a"},
    LIGHT: {"dim": "gray25"},
}


# --- dark or light ----------------------------------------------------------------------
def _theme_value(value) -> Optional[str]:
    value = str(value or "").strip().strip('"').lower()
    return value if value in (DARK, LIGHT) else None


def saved_theme(environ: Optional[Mapping[str, str]] = None,
                platform: Optional[str] = None, home: Optional[Path] = None) -> str:
    """The theme KiPI was last left in, else :data:`DEFAULT`."""
    env = os.environ if environ is None else environ
    platform = platform or sys.platform
    home = home or Path.home()
    try:
        if platform == "win32":
            value = _windows_setting()
        elif platform == "darwin":
            value = _mac_setting(home)
        else:
            config = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
            value = _ini_setting(config / "KiPI" / "KiPI.conf")
    except Exception:  # noqa: BLE001 - a setting we cannot read is a setting not made
        value = None
    return _theme_value(value) or DEFAULT


def _ini_setting(path: Path) -> Optional[str]:
    """QSettings' INI file on Linux: ``[appearance]`` ``theme=dark``."""
    if not path.is_file():
        return None
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.read(path, encoding="utf-8")
    return ini.get("appearance", "theme", fallback=None)


def _windows_setting() -> Optional[str]:
    """QSettings on Windows: ``HKCU\\Software\\KiPI\\KiPI\\appearance``, value ``theme``."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\KiPI\KiPI\appearance") as key:
        return winreg.QueryValueEx(key, "theme")[0]


def _mac_setting(home: Path) -> Optional[str]:
    """QSettings on macOS: ``appearance.theme`` in ``com.kipi.KiPI.plist``."""
    import plistlib
    path = home / "Library" / "Preferences" / "com.kipi.KiPI.plist"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return plistlib.load(f).get("appearance.theme")


# --- how large ----------------------------------------------------------------------------
def _number(text) -> Optional[float]:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _xft_dpi() -> Optional[float]:
    """``Xft.dpi`` from the X resources, which KDE, Xfce and many ``.Xresources`` set."""
    try:
        out = subprocess.run(["xrdb", "-query"], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "xft.dpi":
            return _number(value)
    return None


def _desktop_scales(env: Mapping[str, str], xft_dpi: Optional[float]) -> Iterable[float]:
    """The scales the desktop asks other toolkits for, in the order trusted."""
    if xft_dpi:
        yield xft_dpi / 96
    dpi = _number(env.get("QT_FONT_DPI"))
    if dpi:
        yield dpi / 96
    for name in ("QT_SCALE_FACTOR", "GDK_SCALE"):
        value = _number(env.get(name))
        if value and value != 1:
            if name == "GDK_SCALE":
                value *= _number(env.get("GDK_DPI_SCALE")) or 1
            yield value


def ui_scale(tk_dpi: float, screen_height: int, environ: Optional[Mapping[str, str]] = None,
             platform: Optional[str] = None,
             xft_dpi: Optional[Callable[[], Optional[float]]] = None) -> float:
    """How many times larger than at 96 DPI to draw the window. ``tk_dpi`` is Tk's pixels
    per inch (``winfo_fpixels('1i')``)."""
    env = os.environ if environ is None else environ
    platform = platform or sys.platform

    def clamp(value: float) -> float:
        return round(min(max(value, 1.0), MAX_SCALE), 2)

    forced = _number(env.get(SCALE_ENV))
    if forced:
        return clamp(forced)
    if platform == "darwin":
        return 1.0
    windows = platform in ("win32", "cygwin")
    if not windows:
        for value in _desktop_scales(env, (xft_dpi or _xft_dpi)()):
            return clamp(value)
    if tk_dpi and abs(tk_dpi - 96) > 1:
        return clamp(tk_dpi / 96)
    # 96 DPI is what X says when nobody told it otherwise: go by the screen instead, in
    # quarter steps (1440 lines: 1.25, 2160: 2). On Windows it is the scaling chosen.
    if not windows and screen_height >= 1400:
        return clamp(round(screen_height / 1080 * 4) / 4)
    return 1.0
