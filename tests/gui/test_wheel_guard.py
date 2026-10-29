"""The mouse wheel scrolls a panel instead of changing the field under the pointer."""

import os

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QPoint, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QWheelEvent  # noqa: E402
from qtpy.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QScrollArea,  # noqa: E402
                            QSlider, QVBoxLayout, QWidget)

from kipi.gui.wheel_guard import install  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    install(a)
    return a


def _wheel(w, dy=-120):
    pos = QPointF(w.rect().center())
    ev = QWheelEvent(pos, QPointF(w.mapToGlobal(pos.toPoint())), QPoint(), QPoint(0, dy),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)


def test_wheel_scrolls_the_panel_not_the_fields(app):
    inner = QWidget()
    lay = QVBoxLayout(inner)
    spin, combo, slider = QDoubleSpinBox(), QComboBox(), QSlider(Qt.Horizontal)
    spin.setValue(5.0)
    combo.addItems(["a", "b", "c"])
    slider.setRange(0, 100)
    slider.setValue(50)
    for w in (spin, combo, slider):
        lay.addWidget(w)
    lay.addSpacing(3000)
    area = QScrollArea()
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    app.processEvents()
    bar = area.verticalScrollBar()
    for w in (spin, combo, slider):
        w.setFocus()                    # even the field being edited
        _wheel(w)
    assert (spin.value(), combo.currentIndex(), slider.value()) == (5.0, 0, 50)
    assert bar.value() > 0              # the panel scrolled instead
    before = bar.value()
    _wheel(bar)                         # scroll bars still take the wheel
    assert bar.value() > before
    area.close()
