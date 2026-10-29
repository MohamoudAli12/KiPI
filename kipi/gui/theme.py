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

from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import QApplication, QProxyStyle, QStyle

DARK, LIGHT = "dark", "light"
DEFAULT = DARK
_KEY = "appearance/theme"

_startup: Optional[QPalette] = None       # the palette Qt started with, for light mode
_style: Optional["_IndicatorStyle"] = None   # kept: Qt does not own a Python style

# Each mode's Solve button: green, so the action the setup leads to stands out (grey while
# it cannot be pressed).
SOLVE_BUTTON_STYLE = (
    "QPushButton { font-weight: bold; padding: 6px 14px; color: white; "
    "background: #2e8b3e; border: 1px solid #1f6b2c; border-radius: 4px; }"
    "QPushButton:hover { background: #37a34a; }"
    "QPushButton:pressed { background: #23702f; }"
    "QPushButton:disabled { color: palette(mid); background: palette(button); "
    "border: 1px solid palette(mid); }")


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


class _IndicatorStyle(QProxyStyle):
    """Fusion with a light outline round check boxes and radio buttons in dark mode:
    Fusion outlines them in a shade of the window colour, next to invisible on a dark
    window."""

    dark = False
    _BOXES = (QStyle.PE_IndicatorCheckBox, QStyle.PE_IndicatorItemViewItemCheck)

    def drawPrimitive(self, element, option, painter, widget=None):
        super().drawPrimitive(element, option, painter, widget)
        radio = element == QStyle.PE_IndicatorRadioButton
        if not self.dark or not (radio or element in self._BOXES):
            return
        r = option.rect.adjusted(1, 1, -2, -2) if radio else option.rect.adjusted(0, 0, -1, -1)
        if r.width() < 4 or r.height() < 4:
            return                      # a hidden indicator (the mode switch's)
        on = bool(option.state & QStyle.State_Enabled)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, radio)
        painter.setPen(QPen(QColor("#f0f0f0" if on else "#808080"), 1))
        painter.setBrush(Qt.NoBrush)
        if radio:
            painter.drawEllipse(r)
        else:
            painter.drawRoundedRect(r, 2, 2)
        painter.restore()


def _indicator_style(app: QApplication) -> "_IndicatorStyle":
    global _style
    if not isinstance(app.style(), _IndicatorStyle):
        _style = _IndicatorStyle(app.style().name())
        app.setStyle(_style)
    return app.style()


def apply(name: Optional[str] = None, app: Optional[QApplication] = None) -> str:
    """Show ``name`` (default: the saved theme) on ``app``; returns the theme shown."""
    global _startup
    app = app or QApplication.instance()
    name = name or saved()
    if _startup is None:
        _startup = QPalette(app.palette())
    _indicator_style(app).dark = name == DARK
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
