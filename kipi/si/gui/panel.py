"""The Signal Integrity mode's setup panel.

* **Quick Scan**  every signal net (or the ticked ones) cut into sections: impedance,
  crosstalk, return path, stubs and pair skew, each net PASS / WARN / FAIL. A net or pair
  becomes a channel in one click.
* **Channels**  each channel's nets, interface and buffers (generic or IBIS), and any
  value of the interface's to override.
* **Results**  the solved channel's checks and notes, its report, its results file and its
  Touchstone file.
* **Settings**  the cross-sections, the frequency sweep and the scan's limits.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                            QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                            QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                            QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from kipi.core.model.board import BoardSnapshot
from kipi.dc.gui.pi_panel import ItemList, _Editor
from kipi.dc.pi.model import PISetup
from kipi.gui.report_option import ReportOption
from kipi.gui.theme import SOLVE_BUTTON_STYLE
from kipi.si.model import (CORNERS, ROUGHNESS_MODELS, Buffer, Channel, match_generic,
                           new_channel)
from kipi.si.protocols import PROTOCOLS

STATUS_COLORS = {"PASS": "#60d060", "WARN": "#e0c040", "FAIL": "#ff6060"}
FROM_PROTOCOL = "interface's"


def _spin(lo, hi, decimals, suffix, step=None, special: Optional[str] = None,
          tip: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    if step:
        s.setSingleStep(step)
    if special:
        s.setSpecialValueText(special)     # shown at the minimum: "use the interface's"
    s.setToolTip(tip)
    s.setKeyboardTracking(False)
    return s


def _status_item(status: str) -> QTableWidgetItem:
    it = QTableWidgetItem(status)
    it.setForeground(QColor(STATUS_COLORS.get(status, "#ccc")))
    it.setTextAlignment(Qt.AlignCenter)
    return it


# ---------------------------------------------------------------------------
# Channel editor
# ---------------------------------------------------------------------------

class BufferBox(QGroupBox):
    """A channel end's buffer: generic (supply, output resistance, edge, capacitance) or a
    model in an IBIS file."""
    changed = Signal()

    def __init__(self, title: str, driver: bool, parent=None):
        super().__init__(title, parent)
        self.driver = driver
        self.obj: Optional[Buffer] = None
        self._loading = False
        f = QFormLayout(self)
        self.cmb_kind = QComboBox()
        self.cmb_kind.addItem("Generic", "generic")
        self.cmb_kind.addItem("IBIS model", "ibis")
        self.cmb_kind.setToolTip("A generic CMOS buffer from the values below, or a "
                                 "manufacturer's IBIS model")
        f.addRow("Model", self.cmb_kind)
        self.spin_vcc = _spin(0.1, 20, 3, " V", 0.1, tip="The buffer's supply")
        self.spin_r = _spin(0.1, 1e4, 1, " Ω", 1, tip="The driver's output resistance")
        self.spin_edge = _spin(1, 1e5, 0, " ps", 10, tip="The driver's 0–100 % edge")
        self.spin_c = _spin(0, 100, 2, " pF", 0.1, tip="The pad's (die's) capacitance")
        self.generic_rows = [self.spin_vcc, self.spin_r, self.spin_edge, self.spin_c]
        f.addRow("Supply", self.spin_vcc)
        if driver:
            f.addRow("Output R", self.spin_r)
            f.addRow("Edge (0–100 %)", self.spin_edge)
        f.addRow("Pad C", self.spin_c)
        self.ed_file = QLineEdit()
        self.ed_file.setPlaceholderText("an .ibs file")
        b = QPushButton("…")
        b.setMaximumWidth(30)
        b.setToolTip("Pick the IBIS file")
        b.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.ed_file, 1)
        row.addWidget(b)
        self.file_row = QWidget()
        self.file_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("IBIS file", self.file_row)
        self.cmb_model = QComboBox()
        self.cmb_model.setToolTip("The [Model] in the file")
        f.addRow("IBIS model", self.cmb_model)
        self.ed_pin = QLineEdit()
        self.ed_pin.setPlaceholderText("package's own R/L/C")
        self.ed_pin.setToolTip("The component pin, for its own package R, L and C; blank for "
                               "the [Package]'s")
        f.addRow("Pin", self.ed_pin)
        self.cmb_corner = QComboBox()
        for key, text in CORNERS:
            self.cmb_corner.addItem(text, key)
        f.addRow("Corner", self.cmb_corner)
        self.ibis_rows = [self.file_row, self.cmb_model, self.ed_pin, self.cmb_corner]
        self.lbl_ibis = QLabel()
        self.lbl_ibis.setWordWrap(True)
        self.lbl_ibis.setStyleSheet("color: #999;")
        f.addRow(self.lbl_ibis)
        self.form = f
        for w in (self.spin_vcc, self.spin_r, self.spin_edge, self.spin_c):
            w.valueChanged.connect(self._store)
        for w in (self.cmb_kind, self.cmb_model, self.cmb_corner):
            w.currentIndexChanged.connect(self._store)
        self.ed_file.editingFinished.connect(self._file_changed)
        self.ed_pin.editingFinished.connect(self._store)

    def load(self, b: Buffer):
        self._loading = True
        self.obj = b
        self.cmb_kind.setCurrentIndex(max(self.cmb_kind.findData(b.kind), 0))
        self.spin_vcc.setValue(b.vcc_v)
        self.spin_r.setValue(b.r_out_ohm)
        self.spin_edge.setValue(b.edge_s * 1e12)
        self.spin_c.setValue(b.c_pf)
        self.ed_file.setText(b.ibis_file)
        self._fill_models(b.ibis_model)
        self.ed_pin.setText(b.ibis_pin)
        self.cmb_corner.setCurrentIndex(max(self.cmb_corner.findData(b.corner), 0))
        self._loading = False
        self._show_kind()

    def _show_kind(self):
        ibis = self.cmb_kind.currentData() == "ibis"
        for w in self.ibis_rows:
            self.form.setRowVisible(w, ibis) if hasattr(self.form, "setRowVisible") \
                else w.setVisible(ibis)
        for w in self.generic_rows:
            if self.form.labelForField(w) is not None:
                self.form.setRowVisible(w, not ibis) if hasattr(self.form, "setRowVisible") \
                    else w.setVisible(not ibis)
        self.lbl_ibis.setVisible(ibis)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "IBIS file", str(Path(self.ed_file.text()
                                                                          ).parent),
                                              "IBIS models (*.ibs *.IBS);;All files (*)")
        if path:
            self.ed_file.setText(path)
            self._file_changed()

    def _file_changed(self):
        self._fill_models(self.cmb_model.currentData() or "")
        self._store()

    def _fill_models(self, want: str):
        loading, self._loading = self._loading, True
        self.cmb_model.clear()
        path = self.ed_file.text().strip()
        self.lbl_ibis.setText("")
        if path:
            from kipi.si import ibis
            try:
                f = ibis.read(path)
            except Exception as e:  # noqa: BLE001 - shown, not raised: the user picks again
                self.lbl_ibis.setText(f'<span style="color:#ff7070">{html.escape(str(e))}'
                                      '</span>')
            else:
                for name, m in f.models.items():
                    kind = getattr(m, "model_type", "")
                    self.cmb_model.addItem(f"{name} ({kind})" if kind else name, name)
                self.lbl_ibis.setText(html.escape(
                    f"{f.component or Path(path).name}"
                    + (f" by {f.manufacturer}" if f.manufacturer else "")
                    + f": {len(f.models)} model(s), {len(f.pins)} pin(s)"))
        i = self.cmb_model.findData(want)
        self.cmb_model.setCurrentIndex(i if i >= 0 else 0)
        self._loading = loading

    def _store(self, *_):
        self._show_kind()
        if self._loading or self.obj is None:
            return
        b = self.obj
        b.kind = self.cmb_kind.currentData()
        b.vcc_v = self.spin_vcc.value()
        b.r_out_ohm = self.spin_r.value()
        b.edge_s = self.spin_edge.value() * 1e-12
        b.c_pf = self.spin_c.value()
        b.ibis_file = self.ed_file.text().strip()
        b.ibis_model = self.cmb_model.currentData() or ""
        b.ibis_pin = self.ed_pin.text().strip()
        b.corner = self.cmb_corner.currentData()
        self.changed.emit()


class ChannelEditor(_Editor):
    """A channel: its nets, interface, ends, overrides and buffers."""

    # (attribute, label, minimum = "interface's", maximum, decimals, suffix, scale, tip)
    OVERRIDES = (
        ("rate_bps", "Data rate", 0, 1e3, 4, " Gb/s", 1e9, "The line rate"),
        ("swing_v", "Swing", 0, 10, 3, " V", 1.0,
         "The transmitter's swing (a pair's: peak to peak differential)"),
        ("rise_s", "Edge (20–80 %)", 0, 1e5, 1, " ps", 1e-12, "The transmitter's edge"),
        ("r_source_ohm", "Source R", 0, 1e4, 1, " Ω", 1.0, "Each leg's source resistance"),
        ("r_term_ohm", "Termination R", 0, 1e7, 1, " Ω", 1.0,
         "Each leg's termination at the receiver, to ground or Vtt"),
        ("il_max_db", "IL limit", -100, 0, 1, " dB", 1.0,
         "The insertion loss allowed at Nyquist"),
        ("rl_max_db", "RL limit", -100, 0, 1, " dB", 1.0,
         "The return loss allowed up to Nyquist"),
        ("eye_h_v", "Eye mask height", 0, 10, 3, " V", 1.0, "The receiver's eye mask"),
        ("eye_w_ui", "Eye mask width", 0, 1, 2, " UI", 1.0, "The receiver's eye mask"),
    )

    def __init__(self, panel: "SIPanel", parent=None):
        super().__init__(panel, parent)
        self._ends_for = None
        f = self.form
        self.ed_name = QLineEdit()
        f.addRow("Name", self.ed_name)
        self.cmb_net = QComboBox()
        self.cmb_net.setEditable(True)
        self.cmb_net.setToolTip("The channel's net: a pair's positive leg")
        f.addRow("Net", self.cmb_net)
        self.cmb_partner = QComboBox()
        self.cmb_partner.setEditable(True)
        self.cmb_partner.setToolTip("A pair's negative leg; none for a single-ended channel")
        f.addRow("Pair with", self.cmb_partner)
        self.cmb_proto = QComboBox()
        for key, p in PROTOCOLS.items():
            self.cmb_proto.addItem(p.label, key)
            self.cmb_proto.setItemData(self.cmb_proto.count() - 1, p.note or p.label,
                                       Qt.ToolTipRole)
        self.cmb_proto.setToolTip("The interface: its rate, swing, edges, terminations, "
                                  "equaliser and limits (each can be overridden below)")
        f.addRow("Interface", self.cmb_proto)
        # The ends: the parts on the channel's nets to pick from (or type a reference);
        # "guessed" leaves it to KiPI (the two parts farthest apart, an IC driving).
        self.cmb_drv = QComboBox()
        self.cmb_drv.setEditable(True)
        self.cmb_drv.setToolTip("The transmitting part: the parts on the channel's nets "
                                "(through series parts), or type a reference. Guessed: one "
                                "end of the net, an IC where there is one")
        self.cmb_rcv = QComboBox()
        self.cmb_rcv.setEditable(True)
        self.cmb_rcv.setToolTip("The receiving part; guessed: the part farthest from the "
                                "driver")
        b_swap = QPushButton("Swap")
        b_swap.setToolTip("Swap the transmitter and the receiver (a bidirectional net's "
                          "other direction)")
        b_swap.clicked.connect(self._swap_ends)
        f.addRow("Transmitter", self.cmb_drv)
        row = QHBoxLayout()
        row.addWidget(self.cmb_rcv, 1)
        row.addWidget(b_swap)
        f.addRow("Receiver", row)
        self.chk_follow = QCheckBox("Through series parts")
        self.chk_follow.setToolTip("Follow the net through series resistors and AC-coupling "
                                   "capacitors to the nets beyond them")
        f.addRow("", self.chk_follow)
        self.ed_nofit = QLineEdit()
        self.ed_nofit.setPlaceholderText("none")
        self.ed_nofit.setToolTip("Parts on the board but not fitted in the build to solve "
                                 "(an assembly variant's), comma separated: the channel "
                                 "does not go through them, and their empty pads stay as "
                                 "stubs")
        f.addRow("Not fitted", self.ed_nofit)
        self.lbl_dnp = QLabel()                 # the board's own, from Settings' build
        self.lbl_dnp.setWordWrap(True)
        self.lbl_dnp.setStyleSheet("color: #999;")
        f.addRow("", self.lbl_dnp)
        self.ed_agg = QLineEdit()
        self.ed_agg.setPlaceholderText("none")
        self.ed_agg.setToolTip("Nets whose crosstalk into the channel to include, comma "
                               "separated (the Quick Scan names each net's worst)")
        f.addRow("Aggressors", self.ed_agg)
        self.lbl_proto = QLabel()
        self.lbl_proto.setWordWrap(True)
        self.lbl_proto.setStyleSheet("color: #999;")
        f.addRow(self.lbl_proto)
        self.spins: Dict[str, QDoubleSpinBox] = {}
        for attr, label, lo, hi, dec, suffix, _scale, tip in self.OVERRIDES:
            s = _spin(lo, hi, dec, suffix, special=FROM_PROTOCOL,
                      tip=tip + f"; at the minimum, the {FROM_PROTOCOL} value")
            self.spins[attr] = s
            f.addRow(label, s)
        self.spin_series = _spin(0, 1e4, 1, " Ω", 1, tip="A series resistor at the driver "
                                                          "that is not on the board")
        f.addRow("Series R at driver", self.spin_series)
        self.cmb_prbs = QComboBox()
        for n in (7, 9, 11, 15):
            self.cmb_prbs.addItem(f"PRBS{n}", n)
        self.cmb_prbs.setToolTip("The pattern of the PRBS waveform eye")
        f.addRow("Pattern", self.cmb_prbs)
        self.chk_eq = QCheckBox("Find the best equaliser")
        self.chk_eq.setToolTip("Try the interface's transmitter presets and receiver CTLE "
                               "gains (and use its DFE) for the most open eye")
        f.addRow("", self.chk_eq)
        self.drv = BufferBox("Driver buffer", True)
        self.rcv = BufferBox("Receiver buffer", False)
        b_match = QPushButton("Match generic buffers to the interface")
        b_match.setToolTip("Set the generic driver's supply, output resistance and edge, "
                           "and both ends' pad capacitance, from the interface")
        b_match.clicked.connect(self._match)
        f.addRow(b_match)
        f.addRow(self.drv)
        f.addRow(self.rcv)
        # Last, under the settings it uses: set them, then solve.
        self.btn_solve = QPushButton("Solve")
        self.btn_solve.setStyleSheet(SOLVE_BUTTON_STYLE)
        self.btn_solve.setToolTip("Extract the channel and work out its S-parameters, TDR, "
                                  "eye, waveforms and checks")
        f.addRow(self.btn_solve)
        self.drv.changed.connect(self._store)
        self.rcv.changed.connect(self._store)
        self._watch(self.ed_name, self.ed_agg, self.ed_nofit, self.spin_series,
                    self.cmb_prbs, self.cmb_proto, *self.spins.values())
        for c in (self.cmb_drv, self.cmb_rcv):
            c.activated.connect(lambda _i: self._store())
            c.lineEdit().editingFinished.connect(self._store)
        for c in (self.cmb_net, self.cmb_partner):
            c.currentIndexChanged.connect(self._store)
            c.lineEdit().editingFinished.connect(self._store)
        self.chk_follow.toggled.connect(self._store)
        self.chk_eq.toggled.connect(self._store)

    def refresh_nets(self, nets):
        loading, self._loading = self._loading, True
        try:
            for c, blank in ((self.cmb_net, False), (self.cmb_partner, True)):
                keep = c.currentText()
                c.clear()
                if blank:
                    c.addItem("", "")
                for n in nets:
                    c.addItem(n, n)
                c.setCurrentText(keep)
        finally:
            self._loading = loading

    def _load(self, ch: Channel):
        self.ed_name.setText(ch.name)
        self.cmb_net.setCurrentText(ch.net)
        self.cmb_partner.setCurrentText(ch.partner)
        self.cmb_proto.setCurrentIndex(max(self.cmb_proto.findData(ch.protocol), 0))
        self._fill_ends(ch)
        self._describe_dnp(ch)
        self.chk_follow.setChecked(ch.follow_series)
        self.ed_agg.setText(", ".join(ch.aggressors))
        self.ed_nofit.setText(", ".join(ch.not_fitted))
        for attr, _l, lo, _h, _d, _s, scale, _t in self.OVERRIDES:
            v = getattr(ch, attr)
            self.spins[attr].setValue(lo if v is None else v / scale)
        self.spin_series.setValue(ch.series_r_ohm)
        self.cmb_prbs.setCurrentIndex(max(self.cmb_prbs.findData(ch.prbs), 0))
        self.chk_eq.setChecked(ch.equalize)
        self.drv.load(ch.driver)
        self.rcv.load(ch.receiver)
        self._describe_proto(ch)

    def _save(self, ch: Channel):
        old_name = ch.name
        ch.net = self.cmb_net.currentText().strip()
        ch.partner = self.cmb_partner.currentText().strip()
        ch.name = self.ed_name.text().strip() or (f"{ch.net}/{ch.partner}" if ch.partner
                                                  else ch.net)
        ch.protocol = self.cmb_proto.currentData() or "custom"
        ch.driver_ref = self._end_ref(self.cmb_drv)
        ch.receiver_ref = self._end_ref(self.cmb_rcv)
        if (ch.net, ch.partner, ch.driver_ref, ch.receiver_ref) != self._ends_for:
            self._fill_ends(ch)                 # new nets' parts; the guess re-made
        ch.follow_series = self.chk_follow.isChecked()
        ch.aggressors = [a.strip() for a in self.ed_agg.text().split(",") if a.strip()]
        nofit = [r.strip() for r in self.ed_nofit.text().replace(";", ",").split(",")
                 if r.strip()]
        if nofit != ch.not_fitted:
            ch.not_fitted = nofit
            self._fill_ends(ch)                 # the parts beyond them come and go
        self._describe_dnp(ch)
        for attr, _l, lo, _h, _d, _s, scale, _t in self.OVERRIDES:
            v = self.spins[attr].value()
            setattr(ch, attr, None if v <= lo else v * scale)
        ch.series_r_ohm = self.spin_series.value()
        ch.prbs = int(self.cmb_prbs.currentData() or 7)
        ch.equalize = self.chk_eq.isChecked()
        self._describe_proto(ch)
        if ch.name != old_name:
            self.ed_name.setText(ch.name)

    GUESSED = "guessed"

    @staticmethod
    def _end_ref(c: QComboBox) -> str:
        """A picker's part reference ("" when guessed)."""
        data = c.currentData()
        text = c.currentText().strip()
        if data is not None and c.itemText(c.currentIndex()) == text:
            return data
        return "" if text.lower().startswith(ChannelEditor.GUESSED) else text.split()[0] \
            if text else ""

    def _fill_ends(self, ch: Channel):
        """The ends' pickers: "guessed (U3)", then the parts on the channel's nets."""
        self._ends_for = (ch.net, ch.partner, ch.driver_ref, ch.receiver_ref)
        snap = self.panel.snapshot
        parts, guess = [], ("", "")
        if snap is not None and ch.net:
            from kipi.si.bus import parts_on
            from kipi.si.extract import follow_series
            from kipi.si.model import not_fitted
            skip = not_fitted(snap, self.panel.setup.si, ch)
            nets = [n for x in (ch.net, ch.partner) if x
                    for n in (follow_series(snap, x, not_fitted=skip)
                              if ch.follow_series else [x])]
            try:
                parts = parts_on(snap, nets)
                from kipi.si.solver import ends
                probe = Channel(ch.name, ch.net, ch.partner, not_fitted=skip)
                probe.driver_ref, probe.receiver_ref = ch.driver_ref, ch.receiver_ref
                probe.follow_series = ch.follow_series
                guess = ends(snap, probe, nets)       # what a blank end resolves to
            except Exception:  # noqa: BLE001 (a net with no parts: nothing to offer)
                pass
        loading, self._loading = self._loading, True
        try:
            for c, ref, g in ((self.cmb_drv, ch.driver_ref, guess[0]),
                              (self.cmb_rcv, ch.receiver_ref, guess[1])):
                c.clear()
                c.addItem(f"{self.GUESSED} ({g})" if g else self.GUESSED, "")
                for r, n in parts:
                    c.addItem(f"{r}   ({n} pad{'s' if n != 1 else ''})", r)
                i = c.findData(ref) if ref else 0
                if i >= 0:
                    c.setCurrentIndex(i)
                else:
                    c.setEditText(ref)
        finally:
            self._loading = loading

    def _describe_dnp(self, ch: Channel):
        """The parts on the channel's nets the board's build leaves off (KiCad's DNP or
        the variant's), which apply with the ones typed above."""
        snap, si = self.panel.snapshot, self.panel.setup.si
        if snap is None:
            self.lbl_dnp.setText("")
            return
        from kipi.si.extract import follow_series
        from kipi.si.model import not_fitted
        variant = si.variant if si.variant in snap.variants else ""
        dnp = set(snap.dnp_refs(variant)) - set(ch.not_fitted)
        skip = not_fitted(snap, si, ch)
        nets = [n for x in (ch.net, ch.partner) if x and x in snap.net_names()
                for n in (follow_series(snap, x, not_fitted=skip) if ch.follow_series
                          else [x])]
        here = sorted({p.footprint for n in nets for p in snap.items_on_net(n)[2]} & dnp)
        build = f"variant {variant}" if variant else "KiCad's DNP"
        self.lbl_dnp.setText(f"Also not fitted ({build}): {', '.join(here)}" if here else "")

    def _swap_ends(self):
        if self.obj is None:
            return
        ch = self.obj
        ch.driver_ref, ch.receiver_ref = self._end_ref(self.cmb_rcv), self._end_ref(self.cmb_drv)
        if not (ch.driver_ref and ch.receiver_ref):
            # A guessed end swapped: name both, the guess's parts the other way round.
            g = [self.cmb_drv.itemText(0), self.cmb_rcv.itemText(0)]
            refs = [x[x.find("(") + 1:-1] if "(" in x else "" for x in g]
            ch.driver_ref = ch.driver_ref or refs[1]
            ch.receiver_ref = ch.receiver_ref or refs[0]
        self._fill_ends(ch)
        self._store()

    def _describe_proto(self, ch: Channel):
        p = ch.proto()
        bits = [f"{p.rate_bps / 1e9:g} Gb/s", f"{'Zdiff' if p.diff else 'Z0'} "
                f"{p.z_target_ohm:g} Ω", f"swing {p.swing_v:g} V",
                f"edge {p.rise_s * 1e12:g} ps", f"Rs {p.r_source_ohm:g} Ω",
                f"Rt {p.r_term_ohm:g} Ω" if p.r_term_ohm < 1e5 else "unterminated"]
        if p.il_max_db is not None:
            bits.append(f"IL ≥ {p.il_max_db:g} dB")
        if p.eye_h_v:
            bits.append(f"mask {p.eye_h_v * 1e3:g} mV × {p.eye_w_ui:g} UI")
        if len(p.ffe_presets) > 1 or p.ctle_range_db != (0.0, 0.0) or p.dfe_taps:
            bits.append(f"equaliser: {len(p.ffe_presets)} FFE presets, CTLE to "
                        f"{min(p.ctle_range_db):g} dB, {p.dfe_taps} DFE taps")
        self.lbl_proto.setText(f"{p.label}: " + ", ".join(bits)
                               + (f". {p.note}" if p.note else ""))

    def _match(self):
        if self.obj is None:
            return
        match_generic(self.obj)
        self.load(self.obj)
        self.changed.emit()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

SCAN_COLUMNS = ("Net", "Status", "Z0 (Ω)", "Zdiff (Ω)", "Length (mm)", "Worst NEXT",
                "Unreferenced (mm)", "Stub (mm)", "Skew (mm)")


class ChannelList(ItemList):
    """The Channels list, each channel with a tick box: Solve solves the ticked ones (the
    selected one when none is ticked)."""
    ticks_changed = Signal()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.status: Dict[str, str] = {}          # channel name -> its last solve's status
        self.lst.setToolTip("Select a channel to edit it below; tick channels to solve them "
                            "all with Solve")
        self.lst.itemChanged.connect(self._ticked)
        row = QHBoxLayout()
        self.chk_all = QCheckBox("Tick all")
        self.chk_all.setToolTip("Tick or untick every channel")
        self.chk_all.setTristate(False)
        self.chk_all.clicked.connect(self._tick_all)
        row.addWidget(self.chk_all)
        row.addStretch(1)
        self.layout().insertLayout(0, row)

    def reload(self):
        super().reload()
        self._decorate()

    def _decorate(self):
        """Tick boxes and each channel's last status on the list's rows."""
        self.lst.blockSignals(True)
        for i, ch in enumerate(self.items()):
            it = self.lst.item(i)
            if it is None:
                continue
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if ch.ticked else Qt.Unchecked)
            st = self.status.get(ch.name)
            text = self.describe(ch) + (f"  —  {st}" if st else "")
            if it.text() != text:
                it.setText(text)
        self.lst.blockSignals(False)
        self._sync_all()

    def _edited(self):
        super()._edited()
        self._decorate()

    def _add(self):
        """Add: make_new gives a list, a channel for each net ticked in Board."""
        new = self.make_new() or []
        if not new:
            return
        self.items().extend(new)
        self.reload()
        self.select_name(new[0].name)
        self.changed.emit()

    def set_status(self, name: str, status: Optional[str]):
        if status is None:
            self.status.pop(name, None)
        else:
            self.status[name] = status
        self._decorate()

    def ticked(self) -> List[Channel]:
        return [c for c in self.items() if c.ticked]

    def _ticked(self, item):
        row = self.lst.row(item)
        items = self.items()
        if not 0 <= row < len(items):
            return
        on = item.checkState() == Qt.Checked
        if items[row].ticked != on:
            items[row].ticked = on
            self._sync_all()
            self.changed.emit()
            self.ticks_changed.emit()

    def _tick_all(self, on: bool):
        for c in self.items():
            c.ticked = bool(on)
        self._decorate()
        self.changed.emit()
        self.ticks_changed.emit()

    def _sync_all(self):
        items = self.items()
        self.chk_all.blockSignals(True)
        self.chk_all.setChecked(bool(items) and all(c.ticked for c in items))
        self.chk_all.setEnabled(bool(items))
        self.chk_all.blockSignals(False)


def _skipped_html(nets) -> str:
    if not nets:
        return ""
    names = ", ".join(html.escape(n.rsplit("/", 1)[-1]) for n in nets[:12])
    more = f" and {len(nets) - 12} more" if len(nets) > 12 else ""
    return (f'<br><span style="color:#999">{len(nets)} slow control net(s) left out '
            f"({names}{more}): tick <b>Slow nets</b> to scan them too.</span>")


class SIPanel(QWidget):
    solve_requested = Signal(object, object)       # PISetup, channel names (a list)
    scan_requested = Signal(object, object)        # PISetup, nets (None: every signal net)
    result_selected = Signal(object)               # ChannelResult shown; None clears
    results_changed = Signal()                     # a channel solved, failed or dropped
    overlay_changed = Signal()                     # the 3D impedance overlay changed
    focus_changed = Signal()                       # focus_nets() changed
    setup_changed = Signal()
    results_saved = Signal(str)
    open_results_requested = Signal()              # reopen a saved .kipi-si-results
    report_requested = Signal(object, str)         # the shown ChannelResult, "html"/"pdf"
    reports_requested = Signal(object, str)        # ChannelResults for one report, format
    touchstone_requested = Signal(object)          # the shown ChannelResult
    spice_requested = Signal(object)               # the shown ChannelResult
    part_selected = Signal(str, str, bool)         # for the main window; none here

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.snapshot: Optional[BoardSnapshot] = None
        self.setup = PISetup()
        self._nets: List[str] = []
        self.result = None
        self._results: Dict[str, object] = {}
        self._failed: Dict[str, str] = {}          # channel name -> why its solve failed
        self._busy = False
        self.scan_result = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        top.addWidget(QLabel("Channel:"))
        self.cmb_channel = QComboBox()
        self.cmb_channel.setToolTip("The channel to solve and to show the results of")
        self.cmb_channel.currentIndexChanged.connect(self._channel_changed)
        top.addWidget(self.cmb_channel, 1)
        lay.addLayout(top)
        self.lbl_errors = QLabel()
        self.lbl_errors.setWordWrap(True)
        self.lbl_errors.setStyleSheet("color: #ff9060;")
        lay.addWidget(self.lbl_errors)

        self.tabs = QTabWidget(self)
        lay.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_scan(), "Quick Scan")
        editor = ChannelEditor(self)
        # Solve sits at the foot of the channel's settings; the picker above and the
        # Channels list name the same channel, so it (and Ctrl+Return) solves that one.
        self.btn_solve = editor.btn_solve
        self.btn_solve.clicked.connect(self._solve)
        self.ch_list = ChannelList(lambda: self.setup.si_channels, editor,
                                   self._new_channels, self._describe_channel, noun="channel")
        self.ch_list.buttons.itemAt(0).widget().setToolTip(
            "Add a channel for each signal net ticked in Board's Nets list (a pair's two "
            "legs as one channel) that has none yet")
        self.ch_list.ticks_changed.connect(self._on_changed)
        self.ch_list.selected.connect(self._list_selected)
        self.ch_list.changed.connect(self._on_changed)
        b_bus = QPushButton("Add bus…")
        b_bus.setToolTip("Channels for every net between a transmitting and a receiving "
                         "part (a DDR byte lane, an ADC's lanes, SerDes lanes), with both "
                         "ends and each one's neighbours as aggressors set")
        b_bus.clicked.connect(self._add_bus)
        self.ch_list.buttons.insertWidget(1, b_bus)
        self.tabs.addTab(self.ch_list, "Channels")
        self.tabs.addTab(self._build_results(), "Results")
        self.tabs.addTab(self._build_settings(), "Settings")
        self.tabs.currentChanged.connect(lambda _i: self.overlay_changed.emit())
        self.tabs.currentChanged.connect(lambda _i: self.focus_changed.emit())
        self.ch_list.selected.connect(lambda *_a: self.focus_changed.emit())
        self.ch_list.changed.connect(self.focus_changed.emit)    # its ends or nets edited
        self.tbl_scan.itemSelectionChanged.connect(self.focus_changed.emit)
        self.result_selected.connect(lambda _r: self.focus_changed.emit())
        self.setEnabled(False)

    # ------------------------------------------------------------------ building
    def _build_scan(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItem("Every signal net", "all")
        self.cmb_scope.addItem("The nets ticked in Board", "scope")
        self.cmb_scope.setToolTip("Which nets the Quick Scan cuts")
        self.cmb_scope.currentIndexChanged.connect(self._scope_changed)
        row.addWidget(self.cmb_scope, 1)
        self.chk_slow = QCheckBox("Slow nets")
        self.chk_slow.setToolTip("Also scan the nets named as slow controls (enables, resets, "
                                 "power-good, LEDs, interrupts, I²C, JTAG, straps): their "
                                 "impedance seldom matters, and leaving them out is faster")
        self.chk_slow.toggled.connect(lambda on: self._setting("scan_slow", bool(on)))
        row.addWidget(self.chk_slow)
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setToolTip("Cut every chosen net into sections and check its impedance, "
                                 "crosstalk, return path, stubs and pair skew. Cuts alike "
                                 "are solved once, on every CPU core")
        self.btn_scan.clicked.connect(self._scan)
        row.addWidget(self.btn_scan)
        v.addLayout(row)
        self.tbl_scan = QTableWidget(0, len(SCAN_COLUMNS))
        self.tbl_scan.setHorizontalHeaderLabels(SCAN_COLUMNS)
        self.tbl_scan.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_scan.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_scan.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_scan.verticalHeader().setVisible(False)
        self.tbl_scan.setSortingEnabled(True)
        self.tbl_scan.itemSelectionChanged.connect(self._scan_selected)
        v.addWidget(self.tbl_scan, 2)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_to_channel = QPushButton("Make a channel")
        self.btn_to_channel.setToolTip("Add the selected net (and its pair) as a channel, "
                                       "its worst neighbour as an aggressor")
        self.btn_to_channel.clicked.connect(self._scan_to_channel)
        self.btn_to_channel.setEnabled(False)
        row.addWidget(self.btn_to_channel)
        v.addLayout(row)
        self.txt_scan = QTextBrowser()
        self.txt_scan.setOpenLinks(False)
        v.addWidget(self.txt_scan, 1)
        return w

    def _build_results(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        self.lbl_status = QLabel("Not solved")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)
        self.tbl_checks = QTableWidget(0, 3)
        self.tbl_checks.setHorizontalHeaderLabels(("Check", "Status", "Result"))
        self.tbl_checks.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_checks.verticalHeader().setVisible(False)
        self.tbl_checks.horizontalHeader().setStretchLastSection(True)
        self.tbl_checks.setWordWrap(True)
        v.addWidget(self.tbl_checks, 2)
        self.txt_notes = QTextBrowser()
        v.addWidget(self.txt_notes, 1)
        row = QHBoxLayout()
        self.report_option = ReportOption("Signal Integrity", self._report, self._report_fmt)
        row.addWidget(self.report_option)
        self.btn_report = QPushButton("Open report")
        self.btn_report.setVisible(False)
        self.btn_report.clicked.connect(self._open_report)
        row.addWidget(self.btn_report)
        self.btn_reports = QPushButton("Report several…")
        self.btn_reports.setToolTip("One report of the solved channels you tick, with a "
                                    "picker at its top to show one channel or all")
        self.btn_reports.clicked.connect(self._report_several)
        row.addWidget(self.btn_reports)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.btn_touchstone = QPushButton("Export Touchstone…")
        self.btn_touchstone.setToolTip("Write the channel's S-parameters at its die ports "
                                       "(.s2p / .s4p, 50 Ω) for another simulator")
        self.btn_touchstone.clicked.connect(lambda: self.touchstone_requested.emit(self.result))
        self.btn_save = QPushButton("Save results…")
        self.btn_save.setToolTip("Save every solved channel and the Quick Scan, to reopen "
                                 "without solving")
        self.btn_save.clicked.connect(self._save_results)
        self.btn_open = QPushButton("Open results…")
        self.btn_open.setToolTip("Reopen saved Signal Integrity results (.kipi-si-results) "
                                 "for this board, without solving again")
        self.btn_open.clicked.connect(self.open_results_requested)
        self.btn_spice = QPushButton("Export SPICE…")
        self.btn_spice.setToolTip("Fit the channel's S-parameters with a passive rational "
                                  "model and write it as a SPICE subcircuit (.lib)")
        self.btn_spice.clicked.connect(lambda: self.spice_requested.emit(self.result))
        row.addWidget(self.btn_touchstone)
        row.addWidget(self.btn_spice)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        v.addLayout(row)
        self._show_buttons()
        return w

    def _build_settings(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.set_spins = {}

        def spin(attr, label, lo, hi, dec, suffix, scale=1.0, tip="", special=None):
            sp = _spin(lo, hi, dec, suffix, special=special, tip=tip)
            sp.valueChanged.connect(lambda v, a=attr, k=scale, sp=sp, lo=lo:
                                    self._setting(a, None if (special and v <= lo) else v * k))
            self.set_spins[attr] = (sp, scale, special, lo)
            f.addRow(label, sp)

        f.addRow(QLabel("<b>Assembly</b>"))
        self.cmb_variant = QComboBox()
        self.cmb_variant.setToolTip("The build solved: the parts it leaves off (KiCad's DNP, "
                                    "or a KiCad 10 assembly variant's) are not gone "
                                    "through by any channel, with each channel's own "
                                    "Not fitted parts")
        self.cmb_variant.currentIndexChanged.connect(self._variant_changed)
        f.addRow("Build", self.cmb_variant)
        f.addRow(QLabel("<b>Cross-sections</b>"))
        spin("segment_mm", "Section length", 0.1, 50, 2, " mm",
             tip="A channel's traces are cut into sections at most this long")
        spin("window_mm", "Neighbour window", 0.2, 20, 2, " mm",
             tip="Copper this far either side of a trace is in its cross-section")
        spin("etch_mm", "Etch (each side)", 0, 0.2, 3, " mm",
             tip="A trace's top is this much narrower each side than its bottom")
        self.cmb_rough = QComboBox()
        for key, text in ROUGHNESS_MODELS:
            self.cmb_rough.addItem(text, key)
        self.cmb_rough.currentIndexChanged.connect(
            lambda _i: self._setting("roughness", self.cmb_rough.currentData()))
        f.addRow("Copper roughness", self.cmb_rough)
        spin("rq_um", "RMS roughness", 0, 20, 2, " µm", tip="Hammerstad and Groisse")
        spin("huray_radius_um", "Nodule radius", 0.01, 10, 2, " µm", tip="Huray")
        spin("huray_sr", "Surface ratio", 1, 20, 2, "", tip="Huray: the nodules' area over "
                                                            "the flat area")
        self.chk_vias = QCheckBox("Model vias (barrel, pads, antipads and return path)")
        self.chk_vias.toggled.connect(lambda on: self._setting("via_models", on))
        f.addRow(self.chk_vias)
        f.addRow(QLabel("<b>Frequency sweep</b>"))
        spin("f_max_hz", "Highest frequency", 0, 200, 2, " GHz", 1e9, special="from the rate",
             tip="The sweep's top; at the minimum, from the rate and edge")
        spin("df_hz", "Step", 0.1, 1000, 1, " MHz", 1e6, tip="The sweep's step (its inverse "
                                                              "is the time window)")
        f.addRow(QLabel("<b>Quick Scan</b>"))
        spin("z_target_ohm", "Z0 target", 10, 200, 1, " Ω",
             tip="Single-ended nets' target impedance")
        spin("z_tol_pct", "Tolerance", 1, 50, 1, " %")
        spin("diff_target_ohm", "Zdiff target", 0, 300, 1, " Ω", special="by interface name",
             tip="Pairs' target; at the minimum, from the name (USB 90 Ω, PCIe 85 Ω, else "
                 "100 Ω)")
        spin("kb_limit", "NEXT limit", 0.001, 0.5, 3, "",
             tip="The backward coupling coefficient above which a neighbour warns")
        spin("scan_segment_mm", "Scan section", 0.2, 50, 2, " mm")
        spin("scan_f_max_hz", "Stub check to", 0.1, 100, 2, " GHz", 1e9,
             tip="A via stub resonating below this fails")
        spin("skew_limit_mm", "Pair skew limit", 0, 50, 2, " mm")
        spin("stitch_radius_mm", "Return via radius", 0, 20, 2, " mm",
             tip="A ground via this near a layer change carries its return current")
        f.addRow(QLabel("<b>3D view</b>"))
        self.chk_scan_3d = QCheckBox("Quick Scan impedance in 3D")
        self.chk_scan_3d.setToolTip("Colour the scanned traces in the 3D view by their "
                                    "impedance's distance from the target")
        self.chk_scan_3d.toggled.connect(lambda on: self._setting_3d("scan_3d", on))
        f.addRow(self.chk_scan_3d)
        self.chk_res_3d = QCheckBox("Solved channel's impedance in 3D")
        self.chk_res_3d.setToolTip("Colour the shown result's line sections in the 3D view by "
                                   "their impedance's distance from the target")
        self.chk_res_3d.toggled.connect(lambda on: self._setting_3d("result_3d", on))
        f.addRow(self.chk_res_3d)
        self.chk_end_labels = QCheckBox("TX and RX labels in 3D")
        self.chk_end_labels.setToolTip("Mark the selected channel's Transmitter and Receiver "
                                       "in the 3D view, with their parts' names")
        self.chk_end_labels.toggled.connect(lambda on: self._setting_3d("end_labels_3d", on))
        f.addRow(self.chk_end_labels)
        return w

    # ------------------------------------------------------------------ data
    def nets(self) -> List[str]:
        return list(self._nets)

    def set_board(self, snap: BoardSnapshot, setup: PISetup):
        self.snapshot = snap
        self.setup = setup
        self.clear_results()
        self.scan_result = None
        self._fill_scan()
        self._load_settings()
        self.setEnabled(True)
        self.reload()

    def set_nets(self, nets: List[str]):
        """The nets ticked in the Board panel: the Quick Scan's scope, when chosen."""
        self._nets = list(nets)

    def _signal_nets(self) -> List[str]:
        if self.snapshot is None:
            return []
        from kipi.si.scan import signal_nets
        return signal_nets(self.snapshot)

    def reload(self):
        self.ch_list.editor.refresh_nets(self._signal_nets())
        self.ch_list.reload()
        self._fill_channels()
        self._on_changed()

    def _load_settings(self):
        s = self.setup.si
        for attr, (sp, scale, _special, lo) in self.set_spins.items():
            v = getattr(s, attr)
            sp.blockSignals(True)
            sp.setValue(lo if v is None else v / scale)
            sp.blockSignals(False)
        self.cmb_rough.blockSignals(True)
        self.cmb_rough.setCurrentIndex(max(self.cmb_rough.findData(s.roughness), 0))
        self.cmb_rough.blockSignals(False)
        self.chk_vias.blockSignals(True)
        self.chk_vias.setChecked(s.via_models)
        self.chk_vias.blockSignals(False)
        self._fill_variants()
        self.cmb_scope.blockSignals(True)
        self.cmb_scope.setCurrentIndex(max(self.cmb_scope.findData(s.scan_nets), 0))
        self.cmb_scope.blockSignals(False)
        self.chk_slow.blockSignals(True)
        self.chk_slow.setChecked(s.scan_slow)
        self.chk_slow.setEnabled(s.scan_nets == "all")
        self.chk_slow.blockSignals(False)
        self.report_option.set_value(s.report_format)
        for chk, on in ((self.chk_scan_3d, s.scan_3d), (self.chk_res_3d, s.result_3d),
                        (self.chk_end_labels, s.end_labels_3d)):
            chk.blockSignals(True)
            chk.setChecked(on)
            chk.blockSignals(False)

    def _setting(self, attr, value):
        setattr(self.setup.si, attr, value)
        self.setup_changed.emit()

    def _setting_3d(self, attr, on: bool):
        """A 3D view setting: stored, and the view redrawn."""
        self._setting(attr, bool(on))
        self.overlay_changed.emit()
        self.focus_changed.emit()

    def _fill_variants(self):
        """The builds: the board's own DNP, then each KiCad 10 assembly variant."""
        c, snap = self.cmb_variant, self.snapshot
        variants = getattr(snap, "variants", {}) if snap is not None else {}
        c.blockSignals(True)
        c.clear()
        n = len(snap.dnp_refs()) if snap is not None else 0
        c.addItem(f"As on the board (KiCad's DNP: {n} part{'s' if n != 1 else ''})", "")
        for name, desc in variants.items():
            k = len(snap.dnp_refs(name))
            c.addItem(f"Variant {name}" + (f": {desc}" if desc else "")
                      + f" ({k} not fitted)", name)
        want = self.setup.si.variant
        i = c.findData(want)
        if i < 0 and want:                      # a variant this board does not have
            c.addItem(f"Variant {want} (not on this board: its own DNP used)", want)
            i = c.count() - 1
        c.setCurrentIndex(max(i, 0))
        c.setEnabled(c.count() > 1)
        c.blockSignals(False)

    def _variant_changed(self, _i):
        self._setting("variant", self.cmb_variant.currentData() or "")
        ed = self.ch_list.editor
        if ed.obj is not None:                  # its ends and DNP note follow the build
            ed.load(ed.obj)

    def _scope_changed(self, _i):
        self._setting("scan_nets", self.cmb_scope.currentData())
        self.chk_slow.setEnabled(self.cmb_scope.currentData() == "all")    # ticked: all scanned

    def _report_fmt(self, fmt: str):
        self.setup.si.report_format = fmt

    def _fill_channels(self):
        want = self.cmb_channel.currentData()
        self.cmb_channel.blockSignals(True)
        self.cmb_channel.clear()
        for ch in self.setup.si_channels:
            self.cmb_channel.addItem(ch.name, ch.name)
        i = self.cmb_channel.findData(want)
        self.cmb_channel.setCurrentIndex(i if i >= 0 else (0 if self.setup.si_channels else -1))
        self.cmb_channel.blockSignals(False)
        names = {c.name for c in self.setup.si_channels}
        gone_results = set(self._results) - names
        for gone in gone_results:
            del self._results[gone]
        if gone_results:
            self.results_changed.emit()
        for gone in set(self._failed) - names:
            del self._failed[gone]
        if hasattr(self, "ch_list"):
            for gone in set(self.ch_list.status) - names:
                del self.ch_list.status[gone]
        self._show_channel_result()

    def _channel_changed(self, _i):
        self._on_changed()
        self._show_channel_result()
        name = self.cmb_channel.currentData()
        cur = self.ch_list.current() if hasattr(self, "ch_list") else False
        if name and cur is not False and (cur is None or cur.name != name):
            self.ch_list.select_name(name)

    def _list_selected(self, ch, _by_user=False):
        """The channel picked in the list is the one the picker shows and Solve solves."""
        if ch is not None and ch.name != self.cmb_channel.currentData():
            i = self.cmb_channel.findData(ch.name)
            if i >= 0:
                self.cmb_channel.setCurrentIndex(i)

    def _channel(self) -> Optional[Channel]:
        name = self.cmb_channel.currentData()
        return next((c for c in self.setup.si_channels if c.name == name), None)

    def _on_changed(self):
        self.setup_changed.emit()
        if [self.cmb_channel.itemData(i) for i in range(self.cmb_channel.count())] != \
                [c.name for c in self.setup.si_channels]:
            self._fill_channels()
        self.lbl_errors.setText("\n".join(f"• {e}" for e in self._errors()))
        if not self._busy:
            self.btn_solve.setText(self._solve_label())
            self.btn_solve.setEnabled(not self._errors())

    def to_solve(self) -> List[Channel]:
        """The channels Solve solves: the ticked ones, else the one selected."""
        ticked = [c for c in self.setup.si_channels if c.ticked]
        if ticked:
            return ticked
        ch = self._channel()
        return [ch] if ch is not None else []

    def _solve_label(self) -> str:
        n = sum(c.ticked for c in self.setup.si_channels)
        return f"Solve {n} ticked" if n else "Solve"

    def _errors(self) -> List[str]:
        if self.snapshot is None:
            return ["Load a board"]
        chans = self.to_solve()
        if not chans:
            return ["Add a channel (Channels tab), or make one from the Quick Scan"]
        names = set(self.snapshot.net_names())
        out = []
        for ch in chans:
            which = f"{ch.name}: " if len(chans) > 1 else ""
            for n in (ch.net, ch.partner):
                if n and n not in names:
                    out.append(f"{which}the board has no net {n}" if which else
                               f"The board has no net {n}")
            for b in (ch.driver, ch.receiver):
                if b.kind == "ibis" and not Path(b.ibis_file).is_file():
                    out.append(f"{which}IBIS file not found: "
                               f"{b.ibis_file or '(none chosen)'}")
        return out

    def _solve(self):
        names = [c.name for c in self.to_solve()]
        if names:
            self.solve_requested.emit(self.setup, names)

    def _scan(self):
        nets = self._nets if self.setup.si.scan_nets == "scope" else None
        self.scan_requested.emit(self.setup, nets)

    def set_busy(self, busy: bool, queued: bool = False):
        self._busy = busy
        self.btn_solve.setText(("Queued…" if queued else "Solving…") if busy
                               else self._solve_label())
        self.btn_solve.setEnabled(not busy and not self._errors())
        self.btn_scan.setEnabled(not busy)
        self.btn_open.setEnabled(not busy)

    # ------------------------------------------------------------------ new items
    def _new_channels(self) -> List[Channel]:
        """A channel for each signal net ticked in Board (a pair's two legs one channel,
        its positive leg the net) that no channel has yet."""
        if self.snapshot is None:
            return []
        from kipi.si.pairs import manual_of, pair_map, positive_first
        signal = self._signal_nets()
        ticked = [n for n in signal if n in set(self._nets)]
        if not ticked:
            self.lbl_errors.setText("• Tick the nets to add in Board's Nets list, then Add")
            return []
        manual = manual_of(self.setup.si)
        pairs = pair_map(signal, manual)
        taken = {n for c in self.setup.si_channels for n in (c.net, c.partner) if n}
        out, seen = [], set(taken)
        for n in ticked:
            if n in seen:
                continue
            partner = pairs.get(n, "")
            net, partner = positive_first(n, partner, manual) if partner else (n, "")
            seen |= {net, partner}
            out.append(self._unique(new_channel(net, partner), [c.name for c in out]))
        if not out:
            self.lbl_errors.setText("• Every net ticked in Board has a channel already")
        return out

    def _unique(self, ch: Channel, more: Sequence[str] = ()) -> Channel:
        names = {c.name for c in self.setup.si_channels} | set(more)
        base, k = ch.name, 2
        while ch.name in names:
            ch.name = f"{base} ({k})"
            k += 1
        return ch

    @staticmethod
    def _describe_channel(ch: Channel) -> str:
        return f"{ch.name}: {ch.proto().label}"

    # ------------------------------------------------------------------ Quick Scan
    def show_scan(self, scan):
        self.scan_result = scan
        self._fill_scan()
        self.overlay_changed.emit()

    def _fill_scan(self):
        t = self.tbl_scan
        t.setSortingEnabled(False)
        t.setRowCount(0)
        scan = self.scan_result
        self.txt_scan.clear()
        if scan is None:
            self.txt_scan.setHtml('<span style="color:#999">Scan the board to check every '
                                  'signal net\'s impedance, crosstalk, return path, stubs '
                                  'and pair skew.</span>')
            self.btn_to_channel.setEnabled(False)
            t.setSortingEnabled(True)
            return
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        for ns in sorted(scan.nets, key=lambda n: (order[n.status], n.net)):
            r = t.rowCount()
            t.insertRow(r)

            def num(x, fmt="{:.1f}"):
                it = QTableWidgetItem("" if x is None else fmt.format(x))
                it.setData(Qt.UserRole, -1e30 if x is None else float(x))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                return it

            name = QTableWidgetItem(ns.net + (f" / {ns.partner}" if ns.partner else ""))
            name.setData(Qt.UserRole, ns.net)
            t.setItem(r, 0, name)
            t.setItem(r, 1, _status_item(ns.status))
            z = QTableWidgetItem(f"{ns.z_min:.1f}–{ns.z_max:.1f}" if ns.z_max else "")
            t.setItem(r, 2, z)
            t.setItem(r, 3, QTableWidgetItem(
                f"{ns.zdiff_min:.1f}–{ns.zdiff_max:.1f}" if ns.zdiff_min is not None else ""))
            t.setItem(r, 4, num(ns.length_mm))
            t.setItem(r, 5, QTableWidgetItem(
                f"{ns.worst_kb:.3f} {ns.worst_aggressor}" if ns.worst_aggressor else ""))
            t.setItem(r, 6, num(ns.unreferenced_mm or None))
            t.setItem(r, 7, num(ns.max_stub_mm or None, "{:.2f}"))
            t.setItem(r, 8, num(ns.skew_mm, "{:.2f}"))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()
        n = len(scan.nets)
        fails = sum(ns.status == "FAIL" for ns in scan.nets)
        warns = sum(ns.status == "WARN" for ns in scan.nets)
        self.txt_scan.setHtml(f"{n} net(s) in {scan.seconds:.1f} s: "
                              f'<span style="color:{STATUS_COLORS["FAIL"]}">{fails} FAIL</span>'
                              f', <span style="color:{STATUS_COLORS["WARN"]}">{warns} WARN'
                              "</span>. Select a net for its findings."
                              + _skipped_html(getattr(scan, "skipped_slow", [])))

    def _scan_net(self):
        rows = self.tbl_scan.selectionModel().selectedRows() if self.tbl_scan.selectionModel() \
            else []
        if not rows or self.scan_result is None:
            return None
        net = self.tbl_scan.item(rows[0].row(), 0).data(Qt.UserRole)
        return self.scan_result.by_net().get(net)

    def selected_scan_nets(self) -> Optional[List[str]]:
        ns = self._scan_net()
        if ns is None:
            return None
        return [ns.net] + ([ns.partner] if ns.partner else [])

    def _scan_selected(self):
        ns = self._scan_net()
        self.btn_to_channel.setEnabled(ns is not None)
        if ns is None:
            return
        lines = [f"<b>{html.escape(ns.net)}</b>"
                 + (f" / {html.escape(ns.partner)}" if ns.partner else "")
                 + f": {ns.length_mm:.1f} mm, {ns.delay_s * 1e12:.0f} ps, {ns.vias} via(s), "
                   f"target {ns.target:g} Ω"]
        for fd in ns.findings:
            col = STATUS_COLORS.get(fd.status, "#ccc")
            where = f" at ({fd.where[0]:.2f}, {fd.where[1]:.2f})" if fd.where else ""
            lines.append(f'<span style="color:{col}">{fd.status}</span> {fd.check}: '
                         f"{html.escape(fd.text)}{where}")
        self.txt_scan.setHtml("<br>".join(lines))
        self.overlay_changed.emit()

    def _scan_to_channel(self):
        ns = self._scan_net()
        if ns is None:
            return
        net, partner = ns.net, ns.partner
        if partner and partner < net and not net.endswith(("P", "+", "_p")):
            net, partner = partner, net
        existing = next((c for c in self.setup.si_channels if c.net in (net, partner)), None)
        if existing is None:
            ch = self._unique(new_channel(net, partner))
            if ns.worst_aggressor and ns.worst_kb >= 0.5 * self.setup.si.kb_limit:
                ch.aggressors = [ns.worst_aggressor]
            self.ch_list.add(ch)
            existing = ch
        self._fill_channels()
        self.cmb_channel.setCurrentIndex(self.cmb_channel.findData(existing.name))
        self.tabs.setCurrentWidget(self.ch_list)
        self.ch_list.select_name(existing.name)

    def _add_bus(self):
        if self.snapshot is None:
            return
        from kipi.si.gui.bus_dialog import add_bus
        from kipi.si.pairs import manual_of
        cur = self.ch_list.current()
        from kipi.si.model import Channel, not_fitted
        chans = add_bus(self.snapshot, [c.name for c in self.setup.si_channels], self,
                        getattr(cur, "driver_ref", ""), getattr(cur, "receiver_ref", ""),
                        not_fitted(self.snapshot, self.setup.si, Channel("", "")),
                        manual_of(self.setup.si))
        if not chans:
            return
        self.setup.si_channels.extend(chans)
        self.ch_list.reload()
        self.ch_list.changed.emit()
        self._fill_channels()
        self.tabs.setCurrentWidget(self.ch_list)
        self.ch_list.select_name(chans[0].name)

    # ------------------------------------------------------------------ results
    def clear_results(self):
        self._results.clear()
        self._failed.clear()
        self.ch_list.status.clear()
        self.ch_list.reload()
        self._clear_shown()
        self.results_changed.emit()

    def _clear_shown(self):
        self.result = None
        self.tbl_checks.setRowCount(0)
        self.txt_notes.clear()
        self.lbl_status.setText("Not solved")
        self.btn_report.setVisible(False)
        self._show_buttons()

    def _show_buttons(self):
        have = self.result is not None
        self.report_option.set_available(have)
        self.btn_reports.setEnabled(len(self._results) >= 2)
        self.btn_touchstone.setEnabled(have)
        self.btn_spice.setEnabled(have)
        self.btn_save.setEnabled(bool(self._results) or self.scan_result is not None)

    def show_error(self, message: str, name: Optional[str] = None):
        """Why a solve failed; with ``name``, kept for that channel and shown when it is
        the one picked."""
        if name is not None:
            self._failed[name] = message
            self._results.pop(name, None)
            self.ch_list.set_status(name, "ERROR")
            self.results_changed.emit()
            if name != (self.cmb_channel.currentData() or ""):
                return
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.tbl_checks.parentWidget()))
        self.txt_notes.setHtml(f'<span style="color:#ff7070"><b>Cannot solve:</b><br>'
                               f'{html.escape(message).replace(chr(10), "<br>")}</span>')

    def show_result(self, result, front: bool = True):
        """A solved channel; ``front`` shows the Results tab if it is the one picked."""
        self._results[result.channel.name] = result
        self._failed.pop(result.channel.name, None)
        self.ch_list.set_status(result.channel.name, result.status)
        self.results_changed.emit()
        if result.channel.name == (self.cmb_channel.currentData() or ""):
            self._display(result)
            if front:
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.tbl_checks.parentWidget()))
        self._show_buttons()

    def _show_channel_result(self):
        name = self.cmb_channel.currentData() or ""
        r = self._results.get(name)
        if r is not None:
            self._display(r)
        elif name in self._failed:
            self.show_error(self._failed[name])
        elif self.result is not None:
            self._clear_shown()
            self.result_selected.emit(None)
            self.overlay_changed.emit()

    def _display(self, r):
        self.result = r
        col = STATUS_COLORS.get(r.status, "#ccc")
        self.lbl_status.setText(
            f'<b>{html.escape(r.channel.name)}</b>: <span style="color:{col}"><b>{r.status}'
            f"</b></span> — {r.length_mm:.1f} mm, {r.delay_s * 1e12:.0f} ps, "
            f"{len(r.port_names)} ports, to {r.freqs[-1] / 1e9:.3g} GHz in {r.seconds:.1f} s")
        t = self.tbl_checks
        t.setRowCount(0)
        for c in r.checks:
            k = t.rowCount()
            t.insertRow(k)
            t.setItem(k, 0, QTableWidgetItem(c.check))
            t.setItem(k, 1, _status_item(c.status))
            t.setItem(k, 2, QTableWidgetItem(c.text))
        t.resizeColumnsToContents()
        t.resizeRowsToContents()
        notes = [f"Nets: {', '.join(r.nets)}", "Ports: " + "; ".join(r.port_names)]
        notes += r.notes
        self.txt_notes.setHtml("<br>".join(html.escape(n) for n in notes))
        self.btn_report.setVisible(bool(getattr(r, "report_path", "")))
        self._show_buttons()
        self.result_selected.emit(r)
        self.overlay_changed.emit()

    def set_report(self, result, path: str):
        result.report_path = path
        if result is self.result:
            self.btn_report.setVisible(True)

    def _report(self, fmt: str):
        if self.result is not None:
            self.report_requested.emit(self.result, fmt)

    def solved_results(self) -> List[object]:
        """The solved channels' results, in the setup's order of the channels."""
        return [self._results[c.name] for c in self.setup.si_channels
                if c.name in self._results]

    def _report_several(self):
        from kipi.si.gui.report_dialog import ReportChannelsDialog
        dlg = ReportChannelsDialog(self.solved_results(), self)
        if dlg.exec() == ReportChannelsDialog.Accepted and dlg.chosen():
            self.reports_requested.emit(dlg.chosen(), self.report_option.fmt())

    def _open_report(self):
        path = getattr(self.result, "report_path", "")
        if path:
            from kipi.gui.browser import open_in_browser
            open_in_browser(path)

    def results(self) -> Dict[str, object]:
        return dict(self._results)

    def load_results(self, results: Dict[str, object], scan=None) -> List[str]:
        """Results opened from a file, in place of the ones held; channels not in the
        setup are left out and their names returned."""
        names = {c.name for c in self.setup.si_channels}
        self._results = {k: r for k, r in results.items() if k in names}
        self._failed.clear()
        self.ch_list.status = {k: r.status for k, r in self._results.items()}
        self.ch_list.reload()
        if scan is not None:
            self.show_scan(scan)
        self._show_channel_result()
        self._show_buttons()
        self.results_changed.emit()
        return sorted(set(results) - names)

    def _save_results(self):
        if self.snapshot is None or not (self._results or self.scan_result is not None):
            return
        from kipi.core.naming import stamped
        from kipi.si.report import OPEN_FILTER, RESULTS_SUFFIX, save_results
        start = self.snapshot.project_dir
        stem = f"{stamped(Path(self.snapshot.board_name).stem or 'kipi')}{RESULTS_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(self, "Save Signal Integrity results",
                                              f"{start}/{stem}" if start else stem,
                                              OPEN_FILTER)
        if not path:
            return
        try:
            save_results(path, self._results, self.scan_result, self.snapshot.board_name,
                         self.setup)
        except OSError as e:
            QMessageBox.warning(self, "KiPI", f"Saving failed: {e}")
            return
        self.results_saved.emit(path)

    # ------------------------------------------------------------------ the 3D view
    def overlay(self):
        """The impedance overlay for the 3D view: the shown result's on the Results tab,
        else the Quick Scan's (the selected net's, or every net's)."""
        from kipi.si.viz import channel_overlay, scan_overlay
        on_results = self.tabs.currentWidget() is self.tbl_checks.parentWidget()
        if self.result is not None and (on_results or self.scan_result is None) \
                and self.setup.si.result_3d:
            return channel_overlay(self.snapshot, self.result)
        if self.scan_result is not None and self.setup.si.scan_3d:
            return scan_overlay(self.snapshot, self.scan_result, self.selected_scan_nets())
        return None

    def _focus_channel(self) -> Optional[Channel]:
        """The channel the 3D view picks out: the one being edited, or the shown result's."""
        tab = self.tabs.currentWidget()
        if tab is self.ch_list:
            return self.ch_list.current()
        if tab is self.tbl_checks.parentWidget() and self.result is not None:
            return self.result.channel
        return None

    def end_labels(self) -> list:
        """The picked-out channel's TX and RX, to mark in 3D (:func:`kipi.si.viz.end_labels`)."""
        from kipi.si.viz import end_labels
        if not self.setup.si.end_labels_3d:
            return []
        return end_labels(self.snapshot, self.setup.si, self._focus_channel())

    def focus_nets(self) -> List[str]:
        """The nets to pick out in 3D: the Quick Scan's selected net, the channel being
        edited, or the shown result's channel, as the tab open; with its aggressors."""
        if self.tabs.currentWidget() is self.tbl_scan.parentWidget():
            return self.selected_scan_nets() or []
        ch = self._focus_channel()
        if ch is None or not ch.net:
            return []
        nets = [n for n in (ch.net, ch.partner) if n]
        if ch.follow_series and self.snapshot is not None:
            from kipi.si.extract import follow_series
            from kipi.si.model import not_fitted
            skip = not_fitted(self.snapshot, self.setup.si, ch)
            nets = [x for n in nets for x in follow_series(self.snapshot, n, not_fitted=skip)]
        return nets + [a for a in ch.aggressors if a not in nets]

    # PartSelection's interface, for the main window: SI lists no parts.
    def _part_selection(self, by_user: bool):
        self.part_selected.emit("", "", by_user)

    def select_part(self, kind: str, name: str) -> bool:
        return False
