"""Dark mode: the default, kept in KiPI's settings, and switched back to the startup
palette for light."""

import os

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtGui import QPalette  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from kipi.gui import theme  # noqa: E402


class _Store:
    def __init__(self, value=None):
        self.data = {} if value is None else {theme._KEY: value}

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):
        self.data[key] = value


@pytest.fixture
def app():
    app = QApplication.instance() or QApplication([])
    before = QPalette(app.palette())
    yield app
    app.setPalette(before)
    theme._startup = None


@pytest.fixture
def store(monkeypatch):
    s = _Store()
    monkeypatch.setattr(theme, "_settings", lambda: s)
    return s


def test_dark_is_the_default(store):
    assert theme.saved() == theme.DARK


@pytest.mark.parametrize("value, want", [("light", "light"), ("LIGHT", "light"),
                                         ("dark", "dark"), ("purple", "dark"), ("", "dark")])
def test_saved_choice(monkeypatch, value, want):
    monkeypatch.setattr(theme, "_settings", lambda: _Store(value))
    assert theme.saved() == want


def test_dark_palette_is_dark():
    pal = theme.dark_palette()
    assert pal.window().color().lightness() < 80
    assert pal.base().color().lightness() < pal.window().color().lightness()
    assert pal.windowText().color().lightness() > 200


def test_light_puts_back_the_startup_palette(app, store):
    start = app.palette().window().color()
    assert theme.apply(theme.DARK, app) == theme.DARK
    assert app.palette().window().color() == theme.dark_palette().window().color()
    theme.apply(theme.LIGHT, app)
    assert app.palette().window().color() == start


def test_choose_shows_and_keeps(app, store):
    theme.choose(theme.LIGHT)
    assert store.data[theme._KEY] == theme.LIGHT
    theme.choose(theme.DARK)
    assert store.data[theme._KEY] == theme.DARK
    assert app.palette().window().color() == theme.dark_palette().window().color()


def test_widgets_with_their_own_style_sheet_follow_the_switch(app, store):
    """The bold Solve all button and the log view have style sheets of their own; they
    kept the colours of the theme they were first shown in."""
    from qtpy.QtWidgets import QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

    def middle(w):
        img = w.grab().toImage()
        return img.pixelColor(img.width() // 2, img.height() - 4).lightness()

    theme.apply(theme.LIGHT, app)
    window = QWidget()                       # in a window, as in KiPI's panels
    button = QPushButton("Solve all", window)
    button.setStyleSheet("font-weight: bold; padding: 6px;")
    log = QPlainTextEdit(window)
    log.setStyleSheet("font-family: monospace;")
    layout = QVBoxLayout(window)
    layout.addWidget(button)
    layout.addWidget(log)
    window.resize(300, 300)
    window.show()
    app.processEvents()
    light = middle(button), middle(log)
    theme.apply(theme.DARK, app)
    app.processEvents()
    dark = middle(button), middle(log)
    assert dark[0] < light[0] - 50 and dark[1] < light[1] - 50
    theme.apply(theme.LIGHT, app)
    app.processEvents()
    assert (middle(button), middle(log)) == light
    window.close()
