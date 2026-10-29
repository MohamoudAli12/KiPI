"""Add a bus: pick the transmitting and the receiving part, tick the nets between them
(grouped by name: a DDR byte lane, an ADC's data lanes, a SerDes' lanes), and get a
channel for each with both ends and its neighbours as aggressors set."""

from __future__ import annotations

from typing import List, Optional, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                            QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                            QTableWidget, QTableWidgetItem, QVBoxLayout)

from kipi.core.model.board import BoardSnapshot
from kipi.si.bus import BusNet, _Index, bus_nets, connected_parts, make_channels
from kipi.si.protocols import PROTOCOLS

COLUMNS = ("Net", "Group", "Through series parts")


class BusDialog(QDialog):
    def __init__(self, snap: BoardSnapshot, taken: Sequence[str] = (), parent=None,
                 tx: str = "", rx: str = "", not_fitted: Sequence[str] = ()):
        super().__init__(parent)
        self.setWindowTitle("Add a bus")
        self.snap = snap
        self.taken = set(taken)
        self.bus: List[BusNet] = []
        self.channels: list = []
        self._ix = _Index(snap, not_fitted)        # the build's DNP parts join nothing
        lay = QVBoxLayout(self)
        intro = QLabel("Pick the part that transmits and the one that receives: every "
                       "signal net between them (through series resistors and AC-coupling "
                       "capacitors too) is listed, pairs found by name. Tick the ones to "
                       "make channels of.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        self.cmb_tx = QComboBox()
        self.cmb_tx.setToolTip("The transmitting part (the driver of each channel): parts "
                               "with the most signal nets first")
        self.cmb_rx = QComboBox()
        self.cmb_rx.setToolTip("The receiving part: the parts joined to the transmitter, "
                               "with how many nets join them")
        swap = QPushButton("Swap")
        swap.setToolTip("Swap transmitter and receiver (for the other direction of a "
                        "bidirectional bus such as DDR DQ: reads)")
        swap.clicked.connect(self._swap)
        row = QHBoxLayout()
        row.addWidget(self.cmb_rx, 1)
        row.addWidget(swap)
        f.addRow("Transmitter", self.cmb_tx)
        f.addRow("Receiver", row)
        self.cmb_group = QComboBox()
        self.cmb_group.setToolTip("Tick only one group (a byte lane, the clocks, the "
                                  "address)")
        self.cmb_group.activated.connect(self._tick_group)
        f.addRow("Tick group", self.cmb_group)
        self.cmb_proto = QComboBox()
        self.cmb_proto.addItem("Guess from each net's name", "")
        for key, p in PROTOCOLS.items():
            self.cmb_proto.addItem(p.label, key)
        self.cmb_proto.setToolTip("The interface of every channel made (one per group "
                                  "is easier: add the data lanes, then the clocks)")
        f.addRow("Interface", self.cmb_proto)
        self.spin_agg = QSpinBox()
        self.spin_agg.setRange(0, 4)
        self.spin_agg.setValue(1)
        self.spin_agg.setSuffix(" each side")
        self.spin_agg.setToolTip("Aggressors of each channel: the nets this many places "
                                 "either side of it in its group (DQ3: DQ2 and DQ4)")
        f.addRow("Neighbours as aggressors", self.spin_agg)
        lay.addLayout(f)
        self.tbl = QTableWidget(0, len(COLUMNS))
        self.tbl.setHorizontalHeaderLabels(COLUMNS)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.itemChanged.connect(lambda _it: self._count())
        lay.addWidget(self.tbl, 1)
        row = QHBoxLayout()
        for text, on in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, on=on: self._tick(lambda _b: on))
            row.addWidget(b)
        row.addStretch(1)
        self.lbl = QLabel()
        row.addWidget(self.lbl)
        lay.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Add channels")
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(640, 560)

        self._fill_tx()
        self.cmb_tx.currentIndexChanged.connect(lambda _i: self._fill_rx())
        self.cmb_rx.currentIndexChanged.connect(lambda _i: self._fill_bus())
        if tx:
            self.cmb_tx.setCurrentIndex(max(self.cmb_tx.findData(tx), 0))
        self._fill_rx()
        if rx:
            self.cmb_rx.setCurrentIndex(max(self.cmb_rx.findData(rx), 0))

    # ------------------------------------------------------------------ filling
    def _fill_tx(self):
        ix = self._ix
        refs = sorted(ix.pads_of, key=lambda r: (-len(ix.nets_of(r)), r))
        self.cmb_tx.blockSignals(True)
        self.cmb_tx.clear()
        for r in refs:
            n = len(ix.nets_of(r))
            if n and r not in ix.series_refs:
                self.cmb_tx.addItem(f"{r}   ({n} signal nets)", r)
        self.cmb_tx.blockSignals(False)

    def _fill_rx(self):
        tx = self.cmb_tx.currentData()
        self.cmb_rx.blockSignals(True)
        self.cmb_rx.clear()
        for r, n in (connected_parts(self.snap, tx, self._ix) if tx else []):
            self.cmb_rx.addItem(f"{r}   ({n} net{'s' if n != 1 else ''})", r)
        self.cmb_rx.blockSignals(False)
        self._fill_bus()

    def _fill_bus(self):
        tx, rx = self.cmb_tx.currentData(), self.cmb_rx.currentData()
        self.bus = bus_nets(self.snap, tx, rx, self._ix) if tx and rx else []
        self.tbl.blockSignals(True)            # half-filled rows: not counted yet
        self.tbl.setRowCount(0)
        for b in self.bus:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            it = QTableWidgetItem(b.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            done = b.name in self.taken
            it.setCheckState(Qt.Unchecked if done else Qt.Checked)
            if done:
                it.setToolTip("Already a channel")
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.tbl.setItem(r, 0, it)
            grp = b.group + ("" if b.index is None else f" [{b.index}]")
            self.tbl.setItem(r, 1, QTableWidgetItem(grp + ("  (pair)" if b.partner else "")))
            self.tbl.setItem(r, 2, QTableWidgetItem(", ".join(n.rsplit("/", 1)[-1]
                                                              for n in b.series)))
        self.tbl.blockSignals(False)
        self.tbl.resizeColumnsToContents()
        groups = sorted({b.group for b in self.bus}, key=str.lower)
        self.cmb_group.blockSignals(True)
        self.cmb_group.clear()
        self.cmb_group.addItem("(all)", None)
        for g in groups:
            self.cmb_group.addItem(g, g)
        self.cmb_group.blockSignals(False)
        self._count()

    def _tick(self, pred):
        self.tbl.blockSignals(True)
        for r, b in enumerate(self.bus):
            it = self.tbl.item(r, 0)
            if it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(Qt.Checked if pred(b) else Qt.Unchecked)
        self.tbl.blockSignals(False)
        self._count()

    def _tick_group(self, _i):
        g = self.cmb_group.currentData()
        self._tick(lambda b: g is None or b.group == g)

    def _swap(self):
        tx, rx = self.cmb_tx.currentData(), self.cmb_rx.currentData()
        if not (tx and rx):
            return
        self.cmb_tx.setCurrentIndex(max(self.cmb_tx.findData(rx), 0))
        self.cmb_rx.setCurrentIndex(max(self.cmb_rx.findData(tx), 0))

    def chosen(self) -> List[BusNet]:
        return [b for r, b in enumerate(self.bus)
                if self.tbl.item(r, 0).checkState() == Qt.Checked
                and self.tbl.item(r, 0).flags() & Qt.ItemIsEnabled]

    def _count(self):
        n = len(self.chosen())
        self.lbl.setText(f"{n} channel{'s' if n != 1 else ''} to add")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(n > 0)

    def _accept(self):
        chosen = self.chosen()
        if not chosen:
            return
        self.channels = make_channels(self.snap, self.bus, self.cmb_tx.currentData(),
                                      self.cmb_rx.currentData(),
                                      self.cmb_proto.currentData() or None,
                                      self.spin_agg.value(), sorted(self.taken))
        keep = {b.name for b in chosen}
        self.channels = [c for c in self.channels if c.name in keep]
        self.accept()


def add_bus(snap: BoardSnapshot, taken: Sequence[str], parent=None,
            tx: str = "", rx: str = "", not_fitted: Sequence[str] = ()) -> Optional[list]:
    """Run the dialog: the channels to add, or None. ``not_fitted``: the parts the
    build leaves off."""
    d = BusDialog(snap, taken, parent, tx, rx, not_fitted)
    return d.channels if d.exec() == QDialog.Accepted and d.channels else None
