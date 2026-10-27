# Portions Copyright (c) 2026 Tomachie LLC.
"""The main window must fit a common screen when maximized. Its minimum size is the sum of
its docks' minimums; once that is taller or wider than the screen, Windows places the
window partly off it and the bottom (the log) is cut off."""

import os

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyvistaqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication  # noqa: E402

# A 1080p screen less the taskbar and title bar; 1366 is the narrowest common laptop.
MAX_HEIGHT = 1000
MAX_WIDTH = 1366


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, no_3d):
    from kipi.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    app.processEvents()
    yield w
    w.close()


def test_the_main_window_fits_a_1080p_screen(window):
    size = window.minimumSizeHint()
    assert size.height() <= MAX_HEIGHT, f"minimum height {size.height()} px"
    assert size.width() <= MAX_WIDTH, f"minimum width {size.width()} px"


def test_the_window_opens_inside_the_screens_work_area(window):
    """Frame and title bar included, the window opens within the area above the taskbar,
    on any screen: a 2560 x 1600 laptop at 175 % has only 1463 x 874 logical pixels."""
    area, frame = window.screen().availableGeometry(), window.frameGeometry()
    assert frame.top() >= area.top() and frame.left() >= area.left()
    assert frame.bottom() <= area.bottom()                     # clear of the taskbar
    if window.minimumSizeHint().width() <= area.width():      # room for it across
        assert frame.right() <= area.right()


def test_the_side_docks_scroll_rather_than_set_the_window_height(window):
    # The setup and board docks scroll, so the window's minimum stays well under a
    # small laptop screen's 874 px.
    assert window.minimumSizeHint().height() <= 600


def _board_split(window):
    from qtpy.QtWidgets import QDockWidget, QSplitter
    return window.findChild(QDockWidget, "board_dock").widget().widget().findChild(QSplitter)


def test_the_board_dock_runs_the_full_height(window):
    # Under a full-width log it had no height to spare, so its sections could not be
    # resized: every one sat at its minimum.
    from qtpy.QtCore import Qt
    assert window.corner(Qt.BottomLeftCorner) == Qt.LeftDockWidgetArea


def test_board_sections_resize_without_snapping_shut(window):
    split = _board_split(window)
    assert not split.childrenCollapsible()
    stretch = [split.widget(i).sizePolicy().verticalStretch() for i in range(split.count())]
    assert stretch[0] == max(stretch)                    # the net list grows first


def test_board_lists_stay_tall_enough_to_drag_their_scroll_bars(window):
    # Squeezed to two rows, a list's scroll bar was all handle and could not be dragged.
    rows = window.net_list.fontMetrics().lineSpacing()
    assert window.net_list.minimumHeight() >= 8 * rows
    assert window.layer_list.minimumHeight() >= 4 * rows
    assert window.info.minimumHeight() >= 4 * rows
