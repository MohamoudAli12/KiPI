"""Dark or light widgets. Dark unless the user picks light with the toolbar's Dark mode
button; the choice is kept in KiPI's QSettings.

KiPI draws with Qt's Fusion style, which takes its colours from the application palette:
dark mode sets one here, and light mode puts back the palette Qt started with (the
system's). The plots and the 3D view are dark in either mode, and the power tree keeps its
light canvas; everything that reads the palette (tooltip drawings, the stackup view, the
spinner) follows the switch.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QSettings
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

DARK, LIGHT = "dark", "light"
DEFAULT = DARK
_KEY = "appearance/theme"

_startup: Optional[QPalette] = None       # the palette Qt started with, for light mode


def _settings() -> QSettings:
    return QSettings("KiPI", "KiPI")


def saved() -> str:
    """The theme chosen last, else the default."""
    value = str(_settings().value(_KEY, DEFAULT) or DEFAULT).lower()
    return value if value in (DARK, LIGHT) else DEFAULT


def dark_palette() -> QPalette:
    """A dark Fusion palette, close to the plots' #1e1e1e background."""
    window, base, alt = QColor("#2b2b2b"), QColor("#1e1e1e"), QColor("#262626")
    text, dim, accent = QColor("#e0e0e0"), QColor("#8a8a8a"), QColor("#2f7fd0")
    pal = QPalette()
    for role, color in (
            (QPalette.Window, window), (QPalette.WindowText, text),
            (QPalette.Base, base), (QPalette.AlternateBase, alt),
            (QPalette.ToolTipBase, QColor("#3a3a3a")), (QPalette.ToolTipText, text),
            (QPalette.PlaceholderText, dim), (QPalette.Text, text),
            (QPalette.Button, QColor("#353535")), (QPalette.ButtonText, text),
            (QPalette.BrightText, QColor("#ff6b6b")),
            (QPalette.Light, QColor("#4a4a4a")), (QPalette.Midlight, QColor("#3e3e3e")),
            (QPalette.Mid, QColor("#555555")), (QPalette.Dark, QColor("#1a1a1a")),
            (QPalette.Shadow, QColor("#000000")),
            (QPalette.Highlight, accent), (QPalette.HighlightedText, QColor("#ffffff")),
            (QPalette.Link, QColor("#6ab0f3")), (QPalette.LinkVisited, QColor("#b18cf0"))):
        pal.setColor(role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, dim)
    pal.setColor(QPalette.Disabled, QPalette.Base, window)
    pal.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#4a4a4a"))
    return pal


def apply(name: Optional[str] = None, app: Optional[QApplication] = None) -> str:
    """Show ``name`` (default: the saved theme) on ``app``; returns the theme shown."""
    global _startup
    app = app or QApplication.instance()
    name = name or saved()
    if _startup is None:
        _startup = QPalette(app.palette())
    app.setPalette(dark_palette() if name == DARK else _startup)
    # A widget with its own style sheet (the bold Solve buttons, the log's monospace
    # font) took its colours when it was first shown and ignores a new palette; setting
    # the sheet again makes it take them afresh.
    for w in app.allWidgets():
        sheet = w.styleSheet()
        if sheet:
            w.setStyleSheet(sheet)
    return name


def choose(name: str) -> None:
    """Show ``name`` and keep it for the next start."""
    _settings().setValue(_KEY, apply(name))
