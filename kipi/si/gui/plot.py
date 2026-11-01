"""A Signal Integrity channel's plots: the impedance profile (TDR), the S-parameters,
the eye, the waveforms at the dies and the pulse response; and the solved channels side
by side."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from kipi.ac.gui.plot import COLORS, _eng_hz
from kipi.si.siplots import Ink, draw_compare

EMPTY = "Solve a channel in Signal Integrity mode to see its plots here."

VIEWS = (("eye", "Eye diagram"), ("tdr", "Impedance (TDR)"), ("sparams", "S-parameters"),
         ("waves", "Waveforms at the dies"), ("pulse", "Pulse response"),
         ("compare", "Solved channels compared"))
BG = "#1e1e1e"


class SIPlot(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.result = None
        #: The solved channels' results, in the setup's order, for the comparison.
        self.results_source: Callable[[], List[object]] = lambda: []
        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.figure.patch.set_facecolor(BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.cmb_view = QComboBox(self)
        for key, text in VIEWS:
            self.cmb_view.addItem(text, key)
        self.cmb_view.setToolTip("The plot to show")
        self.cmb_view.currentIndexChanged.connect(lambda _i: self._draw())
        self.chk_eq = QCheckBox("Equalised", self)
        self.chk_eq.setChecked(True)
        self.chk_eq.setToolTip("The eye after the equaliser KiPI chose (the transmitter's "
                               "FFE, the receiver's CTLE and DFE); untick for the raw eye")
        self.chk_eq.toggled.connect(lambda _on: self._draw())
        self.chk_wave = QCheckBox("PRBS waveform", self)
        self.chk_wave.setToolTip("The eye of a simulated PRBS pattern (no jitter or noise) in "
                                 "place of the statistical eye at the target BER")
        self.chk_wave.toggled.connect(lambda _on: self._draw())
        self.lbl = QLabel(EMPTY, self)
        self.lbl.setStyleSheet("color: #999;")
        self.lbl.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.cmb_view, 1)
        top.addWidget(self.chk_eq)
        top.addWidget(self.chk_wave)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.lbl)
        self.clear()

    # ------------------------------------------------------------------ helpers
    def _style(self, ax=None):
        for a in ([ax] if ax is not None else self.figure.axes):
            a.set_facecolor(BG)
            for s in a.spines.values():
                s.set_color("#888")
            a.tick_params(colors="#ccc", which="both")
            a.xaxis.label.set_color("#ccc")
            a.yaxis.label.set_color("#ccc")
            a.title.set_color("#ddd")
            a.grid(True, color="#444", lw=0.5)

    def _legend(self, ax):
        if ax.get_legend_handles_labels()[0]:
            leg = ax.legend(loc="best", fontsize=8, facecolor="#2a2a2a", edgecolor="#555")
            for t in leg.get_texts():
                t.set_color("#ddd")

    def clear(self):
        self.result = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._style()
        self.lbl.setText(EMPTY)
        self.canvas.draw_idle()

    def set_view(self, key: str):
        i = self.cmb_view.findData(key)
        if i >= 0:
            self.cmb_view.setCurrentIndex(i)

    def set_result(self, result):
        self.result = result
        self._draw()

    def refresh_compare(self):
        """The solved channels changed: redraw the comparison if it is shown."""
        if self.cmb_view.currentData() == "compare":
            self._draw()

    def save_png(self, path: str):
        self.figure.savefig(path, dpi=150, facecolor=self.figure.get_facecolor())

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        r = self.result
        view = self.cmb_view.currentData()
        self.chk_eq.setVisible(view == "eye")
        self.chk_wave.setVisible(view == "eye")
        if r is None:
            return
        self.figure.clear()
        self.figure.set_layout_engine("tight")
        self.ax = self.figure.add_subplot(111)
        self._styled = False
        text = getattr(self, f"_draw_{view}")(r) or ""
        if not self._styled:                 # the eye styles its own axes
            self._style()
        if view == "compare":                # every channel, not the one picked
            self.lbl.setText(text)
        else:
            self.lbl.setText(f"<b>{r.channel.name}</b> — {text}" if text else r.channel.name)
        self.canvas.draw_idle()

    def _draw_eye(self, r) -> str:
        from kipi.si.eyeplot import DARK, draw_eye
        e = (r.eye_wave if self.chk_wave.isChecked() else
             (r.eye if self.chk_eq.isChecked() else r.eye_raw))
        if e is None:
            self.ax.set_title("No eye: the channel's pulse response could not be built")
            return ""
        ch = r.channel
        draw_eye(self.figure, e, (ch.value("eye_h_v"), ch.value("eye_w_ui")), DARK)
        self._styled = True
        eq = e.eq.describe() if e.eq is not None else "none"
        out = f"eye {e.height_v * 1e3:.1f} mV × {e.width_ui:.2f} UI"
        if e.kind == "statistical":
            out += f" at BER {e.ber:g}"
        if eq != "none" and self.chk_eq.isChecked():
            out += f"; {eq}"
        if r.com_db is not None:
            out += f"; COM {r.com_db:.2f} dB"
        return out

    def _draw_tdr(self, r) -> str:
        ax = self.ax
        if r.tdr_t is None:
            return ""
        t = r.tdr_t * 1e12
        ax.plot(t, r.tdr_z, color=COLORS[0], lw=1.4,
                label="Zdiff (Sdd11)" if r.channel.partner else "Z (S11)")
        target = r.channel.proto().z_target_ohm
        ax.axhline(target, color="#888", ls="--", lw=1, label=f"Target {target:g} Ω")
        tol = target * r.z_tol_pct / 100
        ax.axhspan(target - tol, target + tol, color="#60d060", alpha=0.08)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Impedance (Ω)")
        sel = r.tdr_t < 2.5 * r.delay_s + 4 * r.channel.value("rise_s")
        if sel.any():
            ax.set_xlim(0, t[sel][-1])
            z = r.tdr_z[sel]
            ax.set_ylim(min(z.min(), target - 2 * tol) - 5, max(z.max(), target + 2 * tol) + 5)
        self._legend(ax)
        note = getattr(r, "tdr_note", "")
        return (f"TDR with a {r.channel.value('rise_s') * 1e12:.0f} ps edge, referred to "
                f"{r.z_ref:g} Ω" + (f". {note}" if note else ""))

    def _draw_compare(self, _r) -> str:
        results = self.results_source()
        if not results:
            self.ax.set_title("Solve channels to compare them here")
            return ""
        self.figure.clear()
        draw_compare(self.figure, results, Ink(COLORS, "#ccc", "#888"), self._style)
        self._styled = True
        passed = sum(x.status == "PASS" for x in results)
        return (f"{len(results)} solved channels, {passed} passing; each bar coloured by "
                "its check, the tick its limit")

    def _draw_sparams(self, r) -> str:
        ax = self.ax
        f = r.freqs
        pair = bool(r.channel.partner)
        ax.plot(f, r.il_db, color=COLORS[0], lw=1.4, label="Sdd21 insertion loss" if pair
                else "S21 insertion loss")
        ax.plot(f, r.rl_db, color=COLORS[1], lw=1.2, label="Sdd11 return loss" if pair
                else "S11 return loss")
        if r.conv_db is not None:
            ax.plot(f, r.conv_db, color=COLORS[2], lw=1, label="Scd21 mode conversion")
        for k, (a, (nxt, fxt)) in enumerate(r.xtalk_db.items()):
            col = COLORS[(3 + k) % len(COLORS)]
            ax.plot(f, nxt, color=col, lw=1, label=f"NEXT from {a}")
            ax.plot(f, fxt, color=col, lw=1, ls="--", label=f"FEXT from {a}")
        ny = r.nyquist_hz
        ax.axvline(ny, color="#888", ls=":", lw=1)
        ax.text(ny, 0, " Nyquist", color="#aaa", fontsize=8, va="top")
        ch = r.channel
        if ch.value("il_max_db") is not None:
            ax.plot([ny], [ch.value("il_max_db")], marker="v", color=COLORS[0], ls="none")
        if ch.value("rl_max_db") is not None:
            ax.hlines(ch.value("rl_max_db"), 0, ny, colors=COLORS[1], linestyles=":", lw=1)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(_eng_hz))
        ax.set_xlim(0, f[-1])
        ax.set_ylim(max(-80, ax.get_ylim()[0]), 3)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("dB")
        self._legend(ax)
        k = int(np.argmin(abs(f - ny)))
        return f"{r.il_db[k]:.2f} dB insertion loss at Nyquist ({_eng_hz(ny)})"

    def _draw_waves(self, r) -> str:
        ax = self.ax
        if r.waves is None:
            ax.set_title("No transient: see the notes")
            return ""
        t = r.waves.t * 1e9
        keys = (["TX diff", "RX diff"] if "RX diff" in r.waves.v else list(r.waves.v))
        for k, name in enumerate(keys):
            ax.plot(t, r.waves.v[name], color=COLORS[k % len(COLORS)], lw=1.2, label=name)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Voltage (V)")
        self._legend(ax)
        kinds = {r.channel.driver.kind, r.channel.receiver.kind}
        return f"with {'IBIS' if 'ibis' in kinds else 'generic'} buffers"

    def _draw_pulse(self, r) -> str:
        ax = self.ax
        p = r.pulse
        if p is None:
            return ""
        t = p.t / p.ui_s
        ax.plot(t, p.v * r.channel.value("swing_v") * 1e3, color=COLORS[0], lw=1.4,
                label="Single-bit response")
        main = int(np.argmax(p.v))
        cur = np.arange(main % p.spui, len(p.v), p.spui)
        ax.plot(t[cur], p.v[cur] * r.channel.value("swing_v") * 1e3, "o", color=COLORS[1],
                ms=3, label="Cursors")
        ax.set_xlim(max(0, t[main] - 4), t[main] + 16)
        ax.set_xlabel("Time (UI)")
        ax.set_ylabel("Voltage at the receiver (mV)")
        self._legend(ax)
        return "a single bit through the channel, before equalisation"
