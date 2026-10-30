"""Selected rows of a list tick and untick together."""

import os

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QPoint, Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication, QListWidget, QListWidgetItem  # noqa: E402

from kipi.gui import multi_tick  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lst(app):
    w = QListWidget()
    w.setSelectionMode(QListWidget.ExtendedSelection)
    for name in ("A", "B", "C", "D"):
        it = QListWidgetItem(name)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Unchecked)
        w.addItem(it)
    w.resize(200, 200)
    w.show()
    QTest.qWaitForWindowExposed(w)
    calls = []
    w.mt = multi_tick.install(w, lambda: calls.append(1))
    w.calls = calls
    yield w
    w.close()


def _ticked(w):
    return [w.item(i).text() for i in range(w.count())
            if w.item(i).checkState() == Qt.Checked]


def _select(w, *rows):
    w.clearSelection()
    for r in rows:
        w.item(r).setSelected(True)


def _box(w, row) -> QPoint:
    r = w.visualItemRect(w.item(row))
    return QPoint(r.left() + 6, r.center().y())


def test_tick_and_untick_selected(lst):
    _select(lst, 0, 2)
    assert lst.mt.tick_selected(True) == 2
    assert _ticked(lst) == ["A", "C"]
    assert lst.mt.tick_selected(False) == 2
    assert _ticked(lst) == []
    assert len(lst.calls) == 2


def test_toggle_ticks_unless_all_ticked(lst):
    lst.item(0).setCheckState(Qt.Checked)
    _select(lst, 0, 1)
    lst.mt.tick_selected(None)                  # one unticked: tick both
    assert _ticked(lst) == ["A", "B"]
    lst.mt.tick_selected(None)                  # all ticked: untick both
    assert _ticked(lst) == []


def test_space_ticks_the_selection(lst):
    _select(lst, 1, 3)
    lst.setFocus()
    QTest.keyClick(lst, Qt.Key_Space)
    assert _ticked(lst) == ["B", "D"]


def test_click_on_a_selected_tick_box_sets_them_all(lst):
    _select(lst, 0, 1, 2)
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, _box(lst, 1))
    assert _ticked(lst) == ["A", "B", "C"]
    assert [lst.item(i).isSelected() for i in range(4)] == [True, True, True, False]
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, _box(lst, 0))
    assert _ticked(lst) == []


def test_one_selected_row_ticks_as_usual(lst):
    _select(lst, 0, 1)
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, _box(lst, 3))
    assert _ticked(lst) == ["D"]                # an unselected row: only it
