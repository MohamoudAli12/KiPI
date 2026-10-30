"""Tick or untick many rows of a list at once.

A list of tick boxes with Ctrl and Shift selection still ticks one row per click. With
:func:`install` on it, the selected rows tick together: clicking the tick box of a
selected row sets every selected row like it, Space ticks them all (or unticks them when
all are ticked), and the right-click menu has Tick selected and Untick selected.
"""

from __future__ import annotations

from typing import Callable, Optional

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QListWidget, QMenu, QStyle


def _checkable(it) -> bool:
    return bool(it.flags() & Qt.ItemIsUserCheckable) and bool(it.flags() & Qt.ItemIsEnabled)


class MultiTick(QObject):
    def __init__(self, lst: QListWidget, changed: Optional[Callable[[], None]] = None,
                 menu_extra: Optional[Callable[[QMenu, list], None]] = None):
        super().__init__(lst)
        self.lst = lst
        self.changed = changed
        self.menu_extra = menu_extra
        self._ate_press = False
        lst.viewport().installEventFilter(self)
        space = QShortcut(QKeySequence(Qt.Key_Space), lst)
        space.setContext(Qt.WidgetShortcut)
        space.activated.connect(lambda: self.tick_selected(None))
        lst.setContextMenuPolicy(Qt.CustomContextMenu)
        lst.customContextMenuRequested.connect(self._menu)

    def selected(self):
        return [it for it in self.lst.selectedItems() if _checkable(it)]

    def tick_selected(self, on: Optional[bool]) -> int:
        """Tick (True) or untick (False) the selected rows; None ticks them, or unticks
        them when every one is ticked. Returns how many rows changed."""
        items = self.selected()
        if not items:
            return 0
        if on is None:
            on = not all(it.checkState() == Qt.Checked for it in items)
        state = Qt.Checked if on else Qt.Unchecked
        todo = [it for it in items if it.checkState() != state]
        if not todo:
            return 0
        self.lst.blockSignals(True)
        for it in todo:
            it.setCheckState(state)
        self.lst.blockSignals(False)
        if self.changed is not None:
            self.changed()
        return len(todo)

    def _in_tick_box(self, it, x: int) -> bool:
        """Is ``x`` (viewport) over the row's tick box, at the row's left?"""
        st = self.lst.style()
        box = (st.pixelMetric(QStyle.PM_IndicatorWidth, None, self.lst)
               + 3 * st.pixelMetric(QStyle.PM_FocusFrameHMargin, None, self.lst) + 4)
        return x - self.lst.visualItemRect(it).left() <= box

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseButtonRelease and self._ate_press:
            self._ate_press = False
            return True
        if t not in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            return False
        if event.button() != Qt.LeftButton or event.modifiers() != Qt.NoModifier:
            return False
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        it = self.lst.itemAt(pos)
        # Only a click on the tick box of one of several selected rows; anything else is
        # Qt's own (it selects the row, ticks the one row).
        if it is None or not it.isSelected() or len(self.selected()) < 2 \
                or not _checkable(it) or not self._in_tick_box(it, pos.x()):
            return False
        self.tick_selected(it.checkState() != Qt.Checked)
        self._ate_press = True
        return True

    def _menu(self, pos):
        items = self.selected()
        menu = QMenu(self.lst)
        n = len(items)
        a_on = menu.addAction(f"Tick selected ({n})" if n > 1 else "Tick selected")
        a_off = menu.addAction(f"Untick selected ({n})" if n > 1 else "Untick selected")
        a_on.setEnabled(any(it.checkState() != Qt.Checked for it in items))
        a_off.setEnabled(any(it.checkState() == Qt.Checked for it in items))
        a_on.triggered.connect(lambda: self.tick_selected(True))
        a_off.triggered.connect(lambda: self.tick_selected(False))
        if self.menu_extra is not None:
            self.menu_extra(menu, items)
        menu.exec(self.lst.viewport().mapToGlobal(pos))


def install(lst: QListWidget, changed: Optional[Callable[[], None]] = None,
            menu_extra: Optional[Callable[[QMenu, list], None]] = None) -> MultiTick:
    """Let ``lst``'s selected rows be ticked together; ``changed`` is called after the
    ticks of several rows change at once (their itemChanged signals are held back).
    ``menu_extra(menu, selected rows)`` adds the list's own entries to the right-click
    menu."""
    return MultiTick(lst, changed, menu_extra)
