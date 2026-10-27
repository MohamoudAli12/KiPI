"""The Auto-fill rail dialog: pick a rail and the part that drives it, and KiPI fills its
power tree from the board, the parts library and the parts' PI_ fields."""

from __future__ import annotations

from typing import Optional, Tuple

from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout,
)

from kipi.core.model.board import BoardSnapshot
from kipi.core.model.nets import classify_nets, sorted_nets
from kipi.dc.pi.importers.rail_autofill import (
    DEFAULT_MAX_LINK_OHM, FIELDS, LIB_NAME, PartsLibrary, library_paths, source_candidates,
)
from kipi.gui.unit_spin import UnitSpinBox


class RailAutofillDialog(QDialog):
    """:meth:`choice` is then (rail, source part, include ground, max link ohms)."""

    def __init__(self, snap: BoardSnapshot, library: PartsLibrary,
                 rail: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-fill rail")
        self.snap, self.library = snap, library
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.rail = QComboBox(self)
        kinds = classify_nets(snap)
        for n in sorted_nets(snap):
            if kinds.get(n) == "power":
                self.rail.addItem(n, n)
        from kipi.dc.gui.pi_panel import make_searchable
        make_searchable(self.rail)
        self.source = QComboBox(self)
        make_searchable(self.source)
        self.ground = QCheckBox("Include the ground pins (simulates the ground net too: slower)",
                                self)
        self.max_ohm = UnitSpinBox()
        self.max_ohm.setShownDecimals(3)
        self.max_ohm.setRange(0.0, 1e6)
        self.max_ohm.setSingleStep(0.1)
        self.max_ohm.setSuffix(" Ω")
        self.max_ohm.setValue(DEFAULT_MAX_LINK_OHM)
        form.addRow("Rail", self.rail)
        form.addRow("Driven by", self.source)
        form.addRow("Follow resistors up to", self.max_ohm)
        form.addRow("", self.ground)
        lay.addLayout(form)
        found = ", ".join(str(p) for p in library.files) or "none found"
        where = " or ".join(str(p) for p in reversed(library_paths(snap.project_dir)))
        self.note = QLabel(
            f"Parts library ({len(library.parts)} parts): {found}.<br>"
            f"KiPI reads {LIB_NAME} from {where}.<br>"
            "Each part's PI_ fields from the schematic override the library.", self)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #999;")
        lay.addWidget(self.note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.rail.setToolTip("The net the source drives. Series parts out of it (beads, "
                             "inductors, small resistors) are followed into their nets.")
        self.source.setToolTip("The regulator or connector that drives the rail: an ideal "
                               "source on its pads on the rail")
        self.max_ohm.setToolTip("Resistors up to this value are parts in series (sense "
                                "resistors, 0 Ω links) and are followed; bigger ones (pull-ups, "
                                "dividers) are skipped")
        self.ground.setToolTip("Give every load and the source its ground pins, and tick "
                               "the ground net. On a large board the ground plane makes the "
                               "solve much slower.")
        self.setToolTip("<p>Fields read from each part:</p><ul>" + "".join(
            f"<li><b>{k}</b>: {v}</li>" for k, v in FIELDS.items()) + "</ul>")
        self.rail.currentIndexChanged.connect(self._fill_sources)
        if rail and self.rail.findData(rail) >= 0:
            self.rail.setCurrentIndex(self.rail.findData(rail))
        self._fill_sources()

    def _fill_sources(self):
        rail = self.rail.currentData()
        self.source.clear()
        if not rail:
            return
        values = {f.reference: f.value for f in self.snap.footprints}
        for ref in source_candidates(self.snap, rail, self.library):
            self.source.addItem(f"{ref}  {values.get(ref, '')}".strip(), ref)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.source.count() > 0)

    def choice(self) -> Tuple[str, str, bool, float]:
        return (self.rail.currentData(), self.source.currentData(), self.ground.isChecked(),
                float(self.max_ohm.value()))
