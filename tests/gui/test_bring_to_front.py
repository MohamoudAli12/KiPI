"""KiPI's window comes to the front when it opens (#112)."""

import os
import sys

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QMainWindow  # noqa: E402

from kipi.gui import single_instance as si  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_bring_to_front_shows_and_restores(app):
    win = QMainWindow()
    win.showMinimized()
    si.bring_to_front(win)
    assert win.isVisible() and not win.isMinimized()
    win.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows foreground rules")
def test_win32_foreground_does_not_raise(app):
    win = QMainWindow()
    win.show()
    si._win32_foreground(int(win.winId()))
    win.close()

