"""Report several channels: tick the solved channels to put in one report, which has a
picker at its top to show one of them or all."""

from __future__ import annotations

from typing import List, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
                            QListWidgetItem, QPushButton, QVBoxLayout)


class ReportChannelsDialog(QDialog):
    """``results`` (ChannelResults, in the order the report takes them) listed with their
    result, every one ticked; :meth:`chosen` gives the ticked ones."""

    def __init__(self, results: Sequence[object], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report several channels")
        self._results = list(results)
        lay = QVBoxLayout(self)
        intro = QLabel("Tick the solved channels to put in the report. It is one page with "
                       "a summary of them all and a picker at its top to show one channel "
                       "or every one.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.lst = QListWidget()
        for r in self._results:
            it = QListWidgetItem(f"{r.channel.name}  ({r.status})")
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked)
            self.lst.addItem(it)
        self.lst.itemChanged.connect(lambda _it: self._update())
        lay.addWidget(self.lst, 1)
        row = QHBoxLayout()
        for label, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
            b = QPushButton(label)
            b.clicked.connect(lambda _c=False, s=state: self._tick_all(s))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Report")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(420, 360)
        self._update()

    def _tick_all(self, state):
        self.lst.blockSignals(True)
        for i in range(self.lst.count()):
            self.lst.item(i).setCheckState(state)
        self.lst.blockSignals(False)
        self._update()

    def _update(self):
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.chosen()))

    def chosen(self) -> List[object]:
        return [r for i, r in enumerate(self._results)
                if self.lst.item(i).checkState() == Qt.Checked]
