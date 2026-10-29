"""A Signal Integrity channel's report: its checks, the extracted line sections, the
plots (eye, TDR, S-parameters, waveforms, pulse response), the Quick Scan's nets and the
channel's S-parameters as a Touchstone file to download (in a PDF, attached)."""

from __future__ import annotations

import base64
import datetime as _dt
import re
import tempfile
from pathlib import Path
from typing import List, Optional

from kipi import __version__
from kipi.core.naming import timestamp
from kipi.dc.report.html import CSS, _e, _png, _table, _verdict, finish, notes_section
from kipi.noise.html_report import _img

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf")
SCAN_ROWS_MAX = 200


def report_name(board_stem: str, channel: str, when: Optional[_dt.datetime] = None,
                fmt: str = "html") -> str:
    """'<board>_<channel>_si_report_<time stamp>.html' (or .pdf)."""
    safe = re.sub(r"[^\w.+\-]+", "_", channel).strip("_") or "channel"
    return f"{board_stem}_{safe}_si_report_{timestamp(when)}.{fmt}"


def _figure(h: float = 3.2):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, h))
    ax = fig.add_subplot(111)
    ax.grid(True, lw=0.4, color="#ccc")
    return fig, ax


def _status(st: str) -> str:
    if st == "WARN":
        return '<span class="unjudged">WARN</span>'
    return _verdict(st == "PASS")


def _eye_png(r, e, title: str) -> str:
    from matplotlib.figure import Figure
    from kipi.si.eyeplot import LIGHT, draw_eye
    fig = Figure(figsize=(9, 4.6 if e.kind == "statistical" else 3.8))
    ch = r.channel
    draw_eye(fig, e, (ch.value("eye_h_v"), ch.value("eye_w_ui")), LIGHT, title=title)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.92, bottom=0.11)
    return _png(fig)


def _tdr_png(r) -> str:
    fig, ax = _figure()
    t = r.tdr_t * 1e12
    ax.plot(t, r.tdr_z, color=COLORS[0], lw=1.2)
    target = r.channel.proto().z_target_ohm
    ax.axhline(target, color="#888", ls="--", lw=0.8)
    tol = target * r.z_tol_pct / 100
    ax.axhspan(target - tol, target + tol, color="#60d060", alpha=0.08)
    sel = r.tdr_t < 2.5 * r.delay_s + 4 * r.channel.value("rise_s")
    if sel.any():
        ax.set_xlim(0, t[sel][-1])
    ax.set_xlabel("ps")
    ax.set_ylabel("Ω")
    ax.set_title(f"Impedance (TDR, {r.channel.value('rise_s') * 1e12:.0f} ps edge, "
                 f"{r.z_ref:g} Ω reference)", fontsize=10)
    fig.tight_layout()
    return _png(fig)


def _sparams_png(r) -> str:
    fig, ax = _figure()
    f = r.freqs / 1e9
    ax.plot(f, r.il_db, color=COLORS[0], lw=1.2, label="insertion loss")
    ax.plot(f, r.rl_db, color=COLORS[1], lw=1.0, label="return loss")
    if r.conv_db is not None:
        ax.plot(f, r.conv_db, color=COLORS[2], lw=0.9, label="Scd21")
    for k, (a, (nxt, fxt)) in enumerate(r.xtalk_db.items()):
        c = COLORS[(3 + k) % len(COLORS)]
        ax.plot(f, nxt, color=c, lw=0.8, label=f"NEXT {a}")
        ax.plot(f, fxt, color=c, lw=0.8, ls="--", label=f"FEXT {a}")
    ax.axvline(r.nyquist_hz / 1e9, color="#888", ls=":", lw=0.8)
    ax.set_ylim(max(-80, ax.get_ylim()[0]), 3)
    ax.set_xlabel("GHz")
    ax.set_ylabel("dB")
    ax.set_title("S-parameters" + (" (differential)" if r.channel.partner else ""), fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return _png(fig)


def _waves_png(r) -> str:
    fig, ax = _figure()
    keys = ["TX diff", "RX diff"] if "RX diff" in r.waves.v else list(r.waves.v)
    for k, name in enumerate(keys):
        ax.plot(r.waves.t * 1e9, r.waves.v[name], color=COLORS[k % len(COLORS)], lw=0.9,
                label=name)
    ax.set_xlabel("ns")
    ax.set_ylabel("V")
    ax.set_title("Waveforms at the dies", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _png(fig)


def _touchstone(r) -> tuple:
    from kipi.si.report import export_touchstone, touchstone_name
    name = touchstone_name("channel", r.channel.name, len(r.port_names))
    with tempfile.TemporaryDirectory() as tmp:
        out = export_touchstone(Path(tmp) / name, r)
        return out.name, out.read_bytes()


def write_si_report(path, result, snap, setup, scan=None, progress=None) -> Path:
    """Write the report of ``result`` (a :class:`~kipi.si.solver.ChannelResult`): a web
    page, or a PDF when ``path`` ends in .pdf."""
    path = Path(path)
    pdf = path.suffix.lower() == ".pdf"
    r = result
    ch = r.channel
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts: List[str] = [
        f"<h1>KiPI Signal Integrity report: {_e(ch.name)}</h1>"
        f"<div class='meta'>{_e(snap.board_name)} · {now} · KiPI {__version__}</div>"]
    if not pdf:
        parts.append("<nav><a href='#checks'>Checks</a> <a href='#channel'>Channel</a> "
                     "<a href='#plots'>Plots</a> <a href='#sections'>Line sections</a>"
                     + (" <a href='#scan'>Quick Scan</a>" if scan is not None else "")
                     + " <a href='#notes'>Notes</a> <a href='#user-notes'>User notes</a></nav>")
    st = r.status
    parts.append(f"<p>Overall: {_status(st)} ({sum(c.status == 'PASS' for c in r.checks)} of "
                 f"{len(r.checks)} checks pass).</p>")
    parts.append("<h2 id='checks'>Checks</h2>")
    parts.append(_table(["Check", "Result", "Detail"],
                        [[_e(c.check), _status(c.status), _e(c.text)] for c in r.checks]))

    # ---------------- channel
    p = ch.proto()
    parts.append("<h2 id='channel'>Channel</h2>")
    rows = [("Nets", ", ".join(r.nets)), ("Interface", p.label),
            ("Ports", "; ".join(r.port_names)),
            ("Data rate", f"{ch.value('rate_bps') / 1e9:.4g} Gb/s"),
            ("Swing", f"{ch.value('swing_v'):.3g} V"),
            ("Edge (20–80 %)", f"{ch.value('rise_s') * 1e12:.4g} ps"),
            ("Source / termination", f"{ch.value('r_source_ohm'):g} Ω / "
                                     f"{ch.value('r_term_ohm'):g} Ω"),
            ("Length / delay", f"{r.length_mm:.2f} mm / {r.delay_s * 1e12:.1f} ps"),
            ("Driver", _buffer(ch.driver)), ("Receiver", _buffer(ch.receiver)),
            ("Sweep", f"{len(r.freqs)} points to {r.freqs[-1] / 1e9:.4g} GHz"),
            ("Solved in", f"{r.seconds:.1f} s")]
    if r.eye is not None:
        rows.append(("Equaliser", r.eye.eq.describe()))
    if r.com_db is not None:
        rows.append(("COM", f"{r.com_db:.2f} dB"))
    parts.append(_table(["", ""], [[_e(a), _e(b)] for a, b in rows]))

    # ---------------- plots
    parts.append("<h2 id='plots'>Plots</h2>")
    if r.eye is not None:
        e = r.eye
        parts.append(_img(pdf, _eye_png(r, e, f"Statistical eye at BER {e.ber:g}: "
                                              f"{e.height_v * 1e3:.1f} mV × {e.width_ui:.2f} UI"
                                              + (f", {e.eq.describe()}"
                                                 if e.eq.describe() != "none" else ""))))
    if r.eye_wave is not None:
        e = r.eye_wave
        parts.append(_img(pdf, _eye_png(r, e, f"PRBS{ch.prbs} waveform eye: "
                                              f"{e.height_v * 1e3:.1f} mV × {e.width_ui:.2f} UI")))
    if r.tdr_t is not None:
        parts.append(_img(pdf, _tdr_png(r)))
        if getattr(r, "tdr_note", ""):
            parts.append(f"<p class='note'>{_e(r.tdr_note)}</p>")
    parts.append(_img(pdf, _sparams_png(r)))
    if r.waves is not None:
        parts.append(_img(pdf, _waves_png(r)))
    name, data = _touchstone(r)
    attachments = []
    if pdf:
        attachments.append((name, data, "The channel's S-parameters (Touchstone)"))
        parts.append(f"<p class='note'>The channel's S-parameters are attached: {_e(name)}.</p>")
    else:
        href = "data:text/plain;base64," + base64.b64encode(data).decode("ascii")
        parts.append(f"<p><a download='{_e(name)}' href='{href}'>Download the S-parameters "
                     f"({_e(name)})</a>: 50 Ω, at the die ports above.</p>")

    # ---------------- sections
    parts.append("<h2 id='sections'>Line sections</h2>")
    srows = []
    for sec in r.sections:
        s0 = sec.segs[0]
        srows.append([_e(" + ".join(sec.nets)), f"{s0.a[0]:.2f}, {s0.a[1]:.2f}",
                      _e(snap.stackup.layer(s0.layer).name if snap.stackup.layer(s0.layer)
                         else s0.layer),
                      f"{s0.width_mm:.3f}", f"{sec.length_mm:.2f}",
                      ", ".join(f"{z:.1f}" for z in sec.z0),
                      f"{sec.zdiff:.1f}" if sec.zdiff else ""])
    parts.append(_table(["Nets", "From (mm)", "Layer", "Width (mm)", "Length (mm)", "Z0 (Ω)",
                         "Zdiff (Ω)"], srows))

    # ---------------- scan
    if scan is not None:
        parts.append("<h2 id='scan'>Quick Scan</h2>")
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        nets = sorted(scan.nets, key=lambda n: (order[n.status], n.net))[:SCAN_ROWS_MAX]
        parts.append(_table(
            ["Net", "Result", "Z0 (Ω)", "Zdiff (Ω)", "Length (mm)", "Worst NEXT", "Findings"],
            [[_e(n.net + (f" / {n.partner}" if n.partner else "")), _status(n.status),
              f"{n.z_min:.1f}–{n.z_max:.1f}",
              f"{n.zdiff_min:.1f}–{n.zdiff_max:.1f}" if n.zdiff_min is not None else "",
              f"{n.length_mm:.1f}",
              _e(f"{n.worst_kb:.3f} {n.worst_aggressor}") if n.worst_aggressor else "",
              "<br>".join(_e(f"{f.status} {f.check}: {f.text}") for f in n.findings
                          if f.status != "PASS")] for n in nets]))
        if len(scan.nets) > SCAN_ROWS_MAX:
            parts.append(f"<p class='note'>The first {SCAN_ROWS_MAX} of {len(scan.nets)} "
                         "nets, failing first.</p>")

    # ---------------- notes
    parts.append("<h2 id='notes'>Notes</h2>")
    parts.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in r.notes) + "</ul>"
                 if r.notes else "<p>None.</p>")
    parts.append("<p class='note'>Line sections: a 2D field solve of each cross-section "
                 "(Djordjevic-Sarkar dielectric, the chosen roughness), cascaded as coupled "
                 "lossy lines with the vias' own models. Eyes: the pulse response's "
                 "statistical eye with the interface's jitter and noise; the PRBS eye is that "
                 "pattern's waveform with neither.</p>")
    parts.append(notes_section(pdf))
    return finish(path, f"KiPI Signal Integrity report: {ch.name}", parts, CSS, pdf,
                  attachments)


def _buffer(b) -> str:
    if b.kind == "ibis":
        return f"IBIS {Path(b.ibis_file).name} [{b.ibis_model or 'first model'}], {b.corner}"
    return (f"generic: {b.vcc_v:g} V, {b.r_out_ohm:g} Ω, {b.edge_s * 1e12:.0f} ps edge, "
            f"{b.c_pf:g} pF")
