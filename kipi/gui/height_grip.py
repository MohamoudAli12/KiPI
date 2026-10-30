"""A bar under a view to drag it taller than the panel it sits in.

A splitter shares out the height its panel has, so in a dock no section grows past the
dock's own height. Dragged down, :class:`HeightGrip` raises its view's minimum height
instead: the panel grows with it, and the scroll area round the panel scrolls. Dragged
up, it gives the height back, down to the view's own minimum. A double-click goes back
to that.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QPoint, QRect, Qt, Signal
from qtpy.QtGui import QPainter, QPalette
from qtpy.QtWidgets import QSizePolicy, QWidget


class HeightGrip(QWidget):
    height_changed = Signal(int)            # the view's new minimum height, or 0: reset

    def __init__(self, view: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view = view
        self.floor = view.minimumHeight()   # the view's own minimum: never below it
        self._press: Optional[QPoint] = None
        self._start = 0
        self.setFixedHeight(9)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to make the list taller (the panel scrolls); double-click "
                        "to shrink it back")

    def set_height(self, h: int):
        """The view's minimum height: ``h``, or its own minimum when that is more."""
        h = max(int(h), self.floor)
        if h != self.view.minimumHeight():
            self.view.setMinimumHeight(h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = self._global(event)
            # From the height it has now, which the splitter may have made more than its
            # minimum: the first pixel dragged moves it.
            self._start = self.view.height()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press is not None:
            self.set_height(self._start + self._global(event).y() - self._press.y())
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._press is not None:
            self._press = None
            self.height_changed.emit(self.view.minimumHeight())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.set_height(self.floor)
        self.height_changed.emit(0)
        event.accept()

    @staticmethod
    def _global(event) -> QPoint:
        p = event.globalPosition() if hasattr(event, "globalPosition") else event.globalPos()
        return p.toPoint() if hasattr(p, "toPoint") else p

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setPen(self.palette().color(QPalette.Mid))
        r: QRect = self.rect()
        cx, cy = r.center().x(), r.center().y()
        for dy in (-2, 0, 2):                           # three short lines: a handle
            p.drawLine(cx - 14, cy + dy, cx + 14, cy + dy)
