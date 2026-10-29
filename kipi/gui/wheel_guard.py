"""The mouse wheel scrolls panels, not the fields under the pointer.

Qt turns a wheel turned over a spin box, a combo box or a slider into a new value, so
scrolling down a setup panel changes whatever field passes under the pointer. Installed
on the application, :class:`WheelGuard` hands those wheel events to the field's parent
instead: the panel scrolls and the value is left alone. Typing, the arrow keys and the
spin buttons still change it.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import (QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox,
                            QScrollBar)


def _guarded(obj) -> bool:
    if isinstance(obj, (QAbstractSpinBox, QComboBox)):
        return True
    return isinstance(obj, QAbstractSlider) and not isinstance(obj, QScrollBar)


class WheelGuard(QObject):
    def eventFilter(self, obj, event):
        if event.type() != QEvent.Wheel or not _guarded(obj):
            return False
        # On up to the scroll area: Qt passes a sent wheel event to no parent itself.
        w = obj.parentWidget()
        while w is not None:
            if not _guarded(w):
                event.ignore()
                QApplication.sendEvent(w, event)
                if event.isAccepted():
                    break
            if w.isWindow():
                break
            w = w.parentWidget()
        return True


def install(app) -> WheelGuard:
    """Guard every field of ``app`` (once; the guard is kept on the app)."""
    guard = app.findChild(WheelGuard)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
    return guard
