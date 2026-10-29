"""A channel's signal integrity solve: from the board and the setup to every result and
check (:class:`ChannelResult`).

1. The channel's nets are followed through series parts and extracted
   (:mod:`kipi.si.extract`): coupled line sections, vias, the series parts.
2. Its ends' pads get the buffers' packages (IBIS R/L/C, or none) and the driver's
   series resistor; ports sit at the dies.
3. The port admittance over a uniform grid up to f_max gives the S-parameters (mixed-mode
   for a pair), insertion and return loss, mode conversion and the TDR.
4. With the interface's linear terminations, the transfer to the receiver gives the pulse
   response: the statistical and waveform eyes, the equaliser that opens them most,
   COM, and each aggressor's crosstalk into them.
5. With the buffers (IBIS or generic) at the ports, a transient gives the waveforms at
   the dies and their overshoot, ringback and flight time.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kipi.core.model.board import BoardSnapshot, Pad
from kipi.core.progress import NULL, ProgressReporter
from kipi.si import eye as eyes
from kipi.si import sparams
from kipi.si.board import SIBoard
from kipi.si.extract import (BuildOptions, build_channel, follow_series, parse_value,
                             series_parts)
from kipi.si.materials import Roughness
from kipi.si.model import Buffer as BufferSpec, Channel, SISettings
from kipi.si.scan import FAIL, PASS, WARN, Finding

C_MM_S = 299792458e3

EYE_PHASES = 128               # the plotted eyes' phases a UI (at least)
PULSE_SAMPLES = 32             # the pulse response's samples a UI (at least)
# A slow interface's edge is a small part of its UI (a 100 Mb/s clock's 0.5 ns edge, 5 %
# of a 10 ns UI): the pulse has this many samples across the edge, and the plotted eye
# this many phases, so the edge is drawn as a line and not a few scattered points.
EDGE_SAMPLES, EDGE_PHASES = 8, 20


def per_ui(ui_s: float, rise_s: float, per_edge: int, least: int, most: int) -> int:
    """Samples a UI that put ``per_edge`` across the edge (0–100 %, the 20–80 % rise
    / 0.6), between ``least`` and ``most``."""
    edge = max(rise_s, 1e-15) / 0.6
    return int(np.clip(math.ceil(per_edge * ui_s / edge), least, most))


class SIError(ValueError):
    pass


@dataclass
class Waves:
    t: np.ndarray
    v: Dict[str, np.ndarray]               # name -> (T,)


@dataclass
class ChannelResult:
    channel: Channel
    nets: List[str]
    port_names: List[str]
    freqs: np.ndarray
    s: np.ndarray                          # (F, P, P) single-ended, at the dies, 50 Ω
    mm: Optional[np.ndarray] = None        # mixed-mode of the victim's 4 ports
    il_db: Optional[np.ndarray] = None
    rl_db: Optional[np.ndarray] = None
    conv_db: Optional[np.ndarray] = None   # Scd21 (pair)
    xtalk_db: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # NEXT, FEXT
    tdr_t: Optional[np.ndarray] = None
    tdr_z: Optional[np.ndarray] = None
    z_ref: float = 50.0
    z_tol_pct: float = 10.0                 # the impedance check's band, for the TDR plot
    sections: list = field(default_factory=list)
    waves: Optional[Waves] = None
    eye: Optional[eyes.EyeResult] = None
    eye_raw: Optional[eyes.EyeResult] = None     # without equalisation
    eye_wave: Optional[eyes.EyeResult] = None
    pulse: Optional[eyes.Pulse] = None
    com_db: Optional[float] = None
    delay_s: float = 0.0
    length_mm: float = 0.0
    skew_s: Optional[float] = None
    tdr_note: str = ""                      # why the TDR reads off the line (series R)
    checks: List[Finding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    solved_setup: object = None

    @property
    def status(self) -> str:
        st = [c.status for c in self.checks]
        return FAIL if FAIL in st else WARN if WARN in st else PASS

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * self.channel.value("rate_bps")


def roughness(s: SISettings) -> Roughness:
    return Roughness(s.roughness, s.rq_um, s.huray_radius_um, s.huray_sr)


def leg_nets(snap: BoardSnapshot, ch: Channel) -> Tuple[List[str], List[str]]:
    """The nets of the channel's legs through its series parts: (P's, N's); N's [] for a
    single-ended channel."""
    skip = ch.not_fitted
    pos = follow_series(snap, ch.net, not_fitted=skip) if ch.follow_series else [ch.net]
    if not ch.partner:
        return pos, []
    return pos, (follow_series(snap, ch.partner, not_fitted=skip) if ch.follow_series
                 else [ch.partner])


def chains(snap: BoardSnapshot, ch: Channel) -> List[List[str]]:
    """The channel's nets as coupled groups: [[P0, N0], [P1, N1], ...] through series
    parts (or [[n0], [n1], ...] single-ended)."""
    pos, neg = leg_nets(snap, ch)
    if not ch.partner:
        return [[n] for n in pos]
    groups = [[p, n] for p, n in zip(pos, neg)]
    groups += [[p] for p in pos[len(neg):]] + [[n] for n in neg[len(pos):]]
    return groups


def _series_refs(snap, nets) -> set:
    return {r for r, *_ in series_parts(snap, nets)}


def ends(snap: BoardSnapshot, ch: Channel, nets: Sequence[str]) -> Tuple[str, str]:
    """The driving and receiving parts' references: as set, else the two parts on the
    channel's first net and its nets farthest apart (an IC driving when only one is)."""
    skip = _series_refs(snap, nets) | set(ch.not_fitted)
    pads = [p for n in nets for p in snap.items_on_net(n)[2] if p.footprint and p.footprint not in skip]
    if not pads:
        raise SIError(f"{ch.net} has no pads")
    drv, rcv = ch.driver_ref, ch.receiver_ref
    if drv and rcv:
        return drv, rcv
    if drv or rcv:
        # One end set: the other is the part farthest from it.
        known = drv or rcv
        mine = [p for p in pads if p.footprint == known]
        far = max(((min((p.position[0] - q.position[0]) ** 2 + (p.position[1] - q.position[1]) ** 2
                        for q in mine), p.footprint) for p in pads if p.footprint != known),
                  default=None) if mine else None
        if far is None:
            raise SIError(f"{ch.net}: no part but {known} to be the other end")
        return (drv, far[1]) if drv else (far[1], rcv)
    first = [p for p in pads if p.net in (ch.net, ch.partner)]
    best, pair = -1.0, None
    for a in (first or pads):
        for b in pads:
            if b.footprint == a.footprint:
                continue
            d = (a.position[0] - b.position[0]) ** 2 + (a.position[1] - b.position[1]) ** 2
            if d > best:
                best, pair = d, (a, b)
    if pair is None:
        raise SIError(f"{ch.net} reaches one part only: no receiver")
    a, b = pair
    if b.footprint.startswith("U") and not a.footprint.startswith("U") \
            and b.net in (ch.net, ch.partner):
        a, b = b, a
    return a.footprint, b.footprint


def _pad_on(snap, ref: str, nets: Sequence[str]) -> Optional[Pad]:
    for n in nets:
        for p in snap.items_on_net(n)[2]:
            if p.footprint == ref:
                return p
    return None


def _ibis_buffer(spec: BufferSpec):
    from kipi.si import ibis
    if spec.kind != "ibis" or not spec.ibis_file:
        return None, None
    f = ibis.read(spec.ibis_file)
    name = spec.ibis_model or (f.model_for_pin(spec.ibis_pin) if spec.ibis_pin else None) \
        or next(iter(f.models))
    if name not in f.models:
        raise SIError(f"{spec.ibis_file}: no model {name!r}")
    return f, ibis.buffer(f.models[name], spec.corner)


def device_buffer(spec: BufferSpec, driver: bool):
    """The behavioural buffer of a channel end (IBIS, else generic) and its package
    (R, L, C)."""
    from kipi.si import ibis
    f, b = _ibis_buffer(spec)
    if b is not None:
        return b, f.package_rlc(spec.ibis_pin or None, spec.corner)
    if driver:
        m = ibis.generic_driver(spec.vcc_v, spec.r_out_ohm, spec.edge_s, spec.c_pf * 1e-12)
    else:
        m = ibis.generic_receiver(spec.vcc_v, spec.c_pf * 1e-12)
    return ibis.buffer(m), (0.0, 0.0, 0.0)


def freq_grid(ch: Channel, s: SISettings) -> np.ndarray:
    rate = ch.value("rate_bps")
    rise = ch.value("rise_s")
    f_max = s.f_max_hz or float(np.clip(max(5 * rate / 2, 3.0 / rise), 3e9, 60e9))
    # The step sets the time window 1/df: at least 64 UI, so the pulse has settled.
    df = min(s.df_hz, rate / 64, f_max / 400)
    return sparams.uniform_grid(f_max, df)


def solve_channel(snap: BoardSnapshot, setup, name: str, progress: ProgressReporter = NULL,
                  board: Optional[SIBoard] = None) -> ChannelResult:
    t_start = time.time()
    ch = next((c for c in setup.si_channels if c.name == name), None)
    if ch is None:
        raise SIError(f"no channel {name!r}")
    ch = copy.deepcopy(ch)
    s = setup.si
    from kipi.si.model import not_fitted
    own = set(ch.not_fitted)
    ch.not_fitted = not_fitted(snap, s, ch)
    dnp_notes = []
    if s.variant and s.variant not in snap.variants:
        dnp_notes.append(f"The board has no assembly variant {s.variant!r}: its own DNP "
                         "is used")
    nets = [n for n in (ch.net, ch.partner) if n]
    for n in nets:
        if n not in snap.net_names():
            raise SIError(f"the board has no net {n!r}")
    with progress.timed("Indexing the board's copper"):
        board = board or SIBoard(snap, getattr(setup.ac, "loss_tangent", None), s.etch_mm,
                                 progress.check_cancel)
    groups = chains(snap, ch)
    pairs = [(g[0], g[1]) for g in groups if len(g) == 2] if ch.partner else []
    for a in ch.aggressors:
        if a in snap.net_names() and a not in nets:
            groups[0].append(a)
    all_nets = [n for g in groups for n in g]
    victim_nets = [n for n in all_nets if n not in ch.aggressors]
    drv_ref, rcv_ref = ends(snap, ch, victim_nets)
    progress.info(f"SI channel {ch.name}: nets {', '.join(all_nets)}; driver {drv_ref}, "
                  f"receiver {rcv_ref}")
    opts = BuildOptions(s.segment_mm, s.window_mm, roughness(s), s.via_models,
                        not_fitted=tuple(ch.not_fitted))
    with progress.timed("Extracting the channel"):
        chan = build_channel(board, groups, opts, progress.check_cancel, pairs,
                             _reporter(progress, "cross-sections field-solved"))
    ckt = chan.circuit
    res_notes = dnp_notes + list(chan.notes)
    on_channel = {p.footprint for n in all_nets for p in snap.items_on_net(n)[2]}
    left_off = [r for r in ch.not_fitted if r not in own and r in on_channel]
    if left_off:
        build = f"variant {s.variant}" if s.variant in snap.variants else "KiCad's DNP"
        res_notes.append(f"Not fitted ({build}): {', '.join(left_off)}")
    # Ports at the dies.
    drv_b, drv_pkg = device_buffer(ch.driver, True)
    rcv_b, rcv_pkg = device_buffer(ch.receiver, False)
    legs = [ch.net] + ([ch.partner] if ch.partner else [])
    port_names, port_keys = [], []

    def add_end(ref, leg_nets, pkg, series_r, label):
        pad = _pad_on(snap, ref, leg_nets)
        if pad is None:
            raise SIError(f"{ref} has no pad on {'/'.join(leg_nets)}")
        if ("pad", pad.id) not in ckt.nodes:
            raise SIError(f"{ref}.{pad.number} on {pad.net} is not reached by any trace")
        r, l, c = pkg
        die = ("die", pad.id)
        if r + series_r > 0 or l > 0:
            ckt.add_l(("pad", pad.id), die, l, r + series_r, f"{ref} package")
        else:
            ckt.add_r(("pad", pad.id), die, 1e-4, f"{ref}")
        if c > 0:
            ckt.add_c(die, None, c, f"{ref} C_pkg")
        port_names.append(f"{label} {ref}.{pad.number} ({pad.net})")
        port_keys.append(die)
        ckt.add_port(port_names[-1], die)

    chain_of = {leg: [g[i] for g in groups if len(g) > i and g[i] not in ch.aggressors]
                for i, leg in enumerate(legs)}
    for leg in legs:
        add_end(drv_ref, chain_of[leg], drv_pkg, ch.series_r_ohm, "TX")
    for leg in legs:
        add_end(rcv_ref, chain_of[leg], rcv_pkg, 0.0, "RX")
    agg_ports = {}
    for a in ch.aggressors:
        if a not in all_nets:
            continue
        from kipi.noise.checks import guess_ends
        da, ra = guess_ends(snap, a)
        if da is None or ra is None:
            res_notes.append(f"Aggressor {a}: cannot find both of its ends; left out")
            continue
        k0 = len(port_names)
        add_end(da.ref, [a], (0, 0, 0), 0.0, "AGG near")
        add_end(ra.ref, [a], (0, 0, 0), 0.0, "AGG far")
        agg_ports[a] = (k0, k0 + 1)
    freqs = freq_grid(ch, s)
    progress.info(f"Sweeping {len(freqs)} frequencies to {freqs[-1] / 1e9:.1f} GHz "
                  f"({len(ckt.nodes)} nodes)")
    with progress.timed("Solving the channel's S-parameters"):
        y = ckt.solve_y(freqs, progress.check_cancel, _reporter(progress, "frequencies"))
    smat = sparams.s_from_y(y, 50.0)
    res = ChannelResult(ch, all_nets, port_names, freqs, smat, sections=chan.sections,
                        notes=res_notes, solved_setup=None)
    res.z_tol_pct = s.z_tol_pct
    pair = bool(ch.partner)
    if pair:
        res.mm = sparams.mixed_mode(smat[:, :4, :4], [(0, 1), (2, 3)])
        res.il_db = sparams.db(res.mm[:, 1, 0])
        res.rl_db = sparams.db(res.mm[:, 0, 0])
        res.conv_db = sparams.db(res.mm[:, 3, 0])
        res.z_ref = 100.0
        res.tdr_t, res.tdr_z = sparams.tdr(freqs, res.mm[:, 0, 0], 100.0, ch.value("rise_s"))
    else:
        res.il_db = sparams.db(smat[:, 1, 0])
        res.rl_db = sparams.db(smat[:, 0, 0])
        res.tdr_t, res.tdr_z = sparams.tdr(freqs, smat[:, 0, 0], 50.0, ch.value("rise_s"))
    for a, (kn, _kf) in agg_ports.items():
        res.xtalk_db[a] = (sparams.db(smat[:, 0, kn]), sparams.db(smat[:, len(legs), kn]))
    # Along the whole leg, through its series parts (a pair's skew past its AC caps too).
    pos_nets, neg_nets = leg_nets(snap, ch)
    res.length_mm = sum(chan.net_length(n) for n in pos_nets)
    res.delay_s = sum(chan.net_delay(n) for n in pos_nets)
    if pair:
        # The legs' length difference at the pair's delay per mm: what length matching
        # matches, not the sum of the sections' delays (each a little different
        # coupled or alone, which would make the skew follow how the legs were paired).
        per_mm = res.delay_s / max(res.length_mm, 1e-9)
        res.skew_s = abs(res.length_mm - sum(chan.net_length(n) for n in neg_nets)) * per_mm
    res.tdr_note = _tdr_note(snap, ch, chan, pos_nets, drv_ref)
    progress.check_cancel()
    with progress.timed("Eye diagrams"):
        _eyes(res, freqs, y, ch, legs, agg_ports, progress)
    progress.check_cancel()
    with progress.timed("Transient with the buffers"):
        try:
            _transient(res, freqs, y, ch, legs, drv_b, rcv_b, agg_ports, progress)
        except SIError as e:
            res.notes.append(str(e))
    _checks(res, ch, s, [p for p, _ in pairs])
    res.seconds = time.time() - t_start
    progress.info(f"SI channel {ch.name}: {res.status} in {res.seconds:.1f} s")
    return res


def _reporter(progress: ProgressReporter, what: str):
    """An ``on_progress(done, total)`` that moves the progress bar and logs every 10 %."""
    last = [0]

    def report(done: int, total: int):
        progress.fraction(done / max(total, 1))
        tenth = 10 * done // max(total, 1)
        if tenth > last[0]:
            last[0] = tenth
            progress.info(f"  {done} of {total} {what}")
    return report


def _linear_ports(ch: Channel, n_ports: int, legs, agg_ports, drive_agg: Optional[str] = None):
    from kipi.si.transient import LinearPort
    rs, rt = ch.value("r_source_ohm"), ch.value("r_term_ohm")
    c_rx = ch.proto().c_rx_f
    ports = [LinearPort(r_load=50.0) for _ in range(n_ports)]
    nl = len(legs)
    for i in range(nl):
        if drive_agg is None:
            ports[i] = LinearPort(drive=1.0 if i == 0 else -1.0, r_source=rs)
        else:
            ports[i] = LinearPort(r_load=rs)
        ports[nl + i] = LinearPort(r_load=rt, c_shunt=c_rx)
    for a, (kn, _kf) in agg_ports.items():
        if a == drive_agg:
            ports[kn] = LinearPort(drive=1.0, r_source=rs)
    return ports


def _eyes(res: ChannelResult, freqs, y, ch: Channel, legs, agg_ports, progress):
    from kipi.si.transient import port_transfer
    p = y.shape[-1]
    nl = len(legs)
    rs, rt = ch.value("r_source_ohm"), ch.value("r_term_ohm")
    launch = (2 if nl == 2 else 1) * rt / (rs + rt)
    v = port_transfer(freqs, y, _linear_ports(ch, p, legs, agg_ports))
    h = (v[:, nl] - v[:, nl + 1]) if nl == 2 else v[:, nl]
    h = h / launch
    proto = ch.proto()
    ui = 1.0 / ch.value("rate_bps")
    rise = ch.value("rise_s")
    swing = ch.value("swing_v")
    spui = per_ui(ui, rise, EDGE_SAMPLES, PULSE_SAMPLES, 512)
    phases = per_ui(ui, rise, EDGE_PHASES, EYE_PHASES, 1024)
    rj, dj = proto.jitter_ui(ch.value("rate_bps"))
    t, st = sparams.step(freqs, h, rise, oversample=max(1, int(math.ceil(spui / (2 * freqs[-1] * ui)))))
    pulse = eyes.pulse_from_step(t, st, ui, spui)
    res.pulse = pulse
    xt = []
    for a in agg_ports:
        va = port_transfer(freqs, y, _linear_ports(ch, p, legs, agg_ports, a))
        ha = ((va[:, nl] - va[:, nl + 1]) if nl == 2 else va[:, nl]) / launch
        ta, sa = sparams.step(freqs, ha, rise, oversample=4)
        xt.append(eyes.pulse_from_step(ta, sa, ui, spui))
    ber = proto.ber
    res.eye_raw = eyes.statistical_eye(pulse, ber, rj, dj, proto.noise_v,
                                       xtalk=xt, swing=swing, phases_per_ui=phases)
    eq = eyes.Equalizer()
    pulse_eq = pulse
    has_eq = (len(proto.ffe_presets) > 1 or proto.ctle_range_db != (0.0, 0.0)
              or proto.dfe_taps)
    if ch.equalize and has_eq:
        eq, pulse_eq = eyes.optimize(freqs, h, ui, rise, proto.ctle_range_db, proto.ffe_presets,
                                     proto.dfe_taps, spui)
        progress.info(f"Equaliser: {eq.describe()}")
    res.eye = eyes.statistical_eye(pulse_eq, ber, rj, dj, proto.noise_v,
                                   eq=eq, xtalk=xt, swing=swing, phases_per_ui=phases)
    res.com_db, _, _ = eyes.com(pulse_eq, ber, rj, dj, proto.noise_v, xt, eq, swing)
    res.eye.com_db = res.com_db
    bits = eyes.prbs(ch.prbs, int(np.clip(2 * (2 ** ch.prbs - 1), 512, 4000)))
    up = np.zeros(len(bits) * spui)
    up[::spui] = bits
    n = len(up) + len(pulse_eq.v) - 1       # by FFT: a slow interface has many samples
    wv = np.fft.irfft(np.fft.rfft(up, n) * np.fft.rfft(pulse_eq.v, n), n)[:len(up)] * swing
    tt = np.arange(len(wv)) * ui / spui
    res.eye_wave = eyes.waveform_eye(tt, wv, ui)
    res.eye_wave.eq = eq


def _transient(res: ChannelResult, freqs, y, ch: Channel, legs, drv_b, rcv_b, agg_ports,
               progress):
    from kipi.si.transient import Device, Stimulus, simulate
    nl = len(legs)
    p = y.shape[-1]
    ui = 1.0 / ch.value("rate_bps")
    edge = max(drv_b.edge_time("rise"), drv_b.edge_time("fall"))
    # Slow single-ended logic: bits long enough for the reflections to settle.
    ui_t = ui if ch.proto().diff else max(ui, 4 * edge, 2 * res.delay_s + 4 * edge)
    bits = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    if ui_t > ui:
        bits = [0, 1, 1, 0, 0]
    t_stop = (len(bits) + 1) * ui_t + 3 * res.delay_s
    dt = min(edge / 12, ui_t / 40)
    if t_stop / dt > 60000:
        dt = t_stop / 60000
    devs = [Device(r_term=50.0) for _ in range(p)]
    for i in range(nl):
        devs[i] = Device(drv_b, Stimulus(bits, ui_t, ui_t / 2, invert=(i == 1)))
        rt = ch.r_term_ohm if ch.r_term_ohm is not None else (
            ch.proto().r_term_ohm if ch.proto().r_term_ohm < 1e5 else None)
        devs[nl + i] = Device(rcv_b, r_term=rt, v_term=ch.v_term_v)
    for kn, kf in agg_ports.values():
        devs[kn] = Device(r_term=ch.value("r_source_ohm"))
        devs[kf] = Device(r_term=ch.value("r_term_ohm"))
    tr = simulate(freqs, y, devs, t_stop, dt, res.port_names, progress.check_cancel)
    res.waves = Waves(tr.t, {res.port_names[i]: tr.v[:, i] for i in range(min(p, 2 * nl))})
    if nl == 2:
        res.waves.v["TX diff"] = tr.v[:, 0] - tr.v[:, 1]
        res.waves.v["RX diff"] = tr.v[:, 2] - tr.v[:, 3]


def edge_metrics(t: np.ndarray, v: np.ndarray, v_low: float, v_high: float,
                 vil: float, vih: float) -> Dict[str, float]:
    """Overshoot (above ``v_high``) and undershoot (below ``v_low``); ringback: how close
    the signal comes back to the threshold it crossed, while it holds a level (negative:
    back across it); and whether every edge goes through ``vil``–``vih`` without turning
    back by more than 2 % of the swing."""
    sw = max(v_high - v_low, 1e-12)
    state = np.zeros(len(v), int)                 # +1 high (≥ vih), −1 low (≤ vil), 0 between
    state[v >= vih] = 1
    state[v <= vil] = -1
    settled = np.flatnonzero(state != 0)
    ring = math.inf
    monotonic = True
    if len(settled):
        # Runs of one level; the samples between two runs of opposite levels are an edge.
        runs = []                                 # (level, first, last)
        for k in settled:
            if runs and runs[-1][0] == state[k]:
                runs[-1][2] = k
            else:
                runs.append([state[k], k, k])
        for n, (lvl, a, b) in enumerate(runs):
            if n == 0:
                continue                          # the level the run started in: no edge
            hold = v[a:b + 1]
            # Ringback: once at the level (within 10 % of the swing), how near the
            # threshold it came again.
            near = np.flatnonzero(hold >= v_high - 0.1 * sw) if lvl > 0 else \
                np.flatnonzero(hold <= v_low + 0.1 * sw)
            if len(near):
                hold = hold[near[0]:near[-1] + 1]
                ring = min(ring, float(hold.min() - vih) if lvl > 0
                           else float(vil - hold.max()))
            prev_end = runs[n - 1][2]
            if runs[n - 1][0] != lvl:
                edge = v[prev_end:a + 1]
                back = (np.maximum.accumulate(edge) - edge) if lvl > 0 else \
                    (edge - np.minimum.accumulate(edge))
                if back.max() > 0.02 * sw:
                    monotonic = False
            else:
                # It left the level for the band and came back: a non-monotonic edge or
                # ringback into the band.
                ring = min(ring, float(v[prev_end:a + 1].min() - vih) if lvl > 0
                           else float(vil - v[prev_end:a + 1].max()))
    return {"overshoot_v": float(v.max() - v_high), "undershoot_v": float(v_low - v.min()),
            "ringback_v": ring if math.isfinite(ring) else float("nan"),
            "monotonic": monotonic}


def _tdr_note(snap, ch: Channel, chan, nets: Sequence[str], drv_ref: str) -> str:
    """What makes the TDR, looking in from the driver, read above the line: a series
    resistor (a source termination) adds its resistance from where the step reaches it."""
    nets = list(nets)
    pad = _pad_on(snap, drv_ref, nets)
    if pad is not None and len(nets) > 1 and pad.net == nets[-1]:
        nets.reverse()                       # the driver at the chain's far end
    found = []
    if ch.series_r_ohm > 0:
        found.append((f"Series R at {drv_ref}", ch.series_r_ohm, 0.0))
    for ref, kind, value, pair in series_parts(snap, nets, ch.not_fitted):
        ohms = parse_value(value) if kind == "R" else None
        if not ohms or ohms < 1.0:           # a capacitor, or a 0 Ω link
            continue
        a, b = pair.split("|")
        k = min(nets.index(a), nets.index(b))
        found.append((ref, ohms, 2 * sum(chan.net_delay(n) for n in nets[:k + 1])))
    if not found:
        return ""
    legs = 2 if ch.partner else 1
    added = legs * sum(o for _r, o, _t in found)
    where = ", ".join(f"{r} ({o:g} Ω) from {t * 1e12:.0f} ps" for r, o, t in found)
    return (f"In series: {where}. From there the TDR, looking in from {drv_ref}, reads the "
            f"line's impedance plus {added:g} Ω" + (" (one on each leg)" if legs == 2 else "")
            + ", so a series-terminated line reads above the target band. The impedance "
              "check judges the traces themselves.")


def _checks(res: ChannelResult, ch: Channel, s: SISettings, pos: Sequence[str] = ()):
    """``pos``: a pair's positive nets along the channel (through any series parts)."""
    c = res.checks
    proto = ch.proto()
    pair = bool(ch.partner)
    target = ch.proto().z_target_ohm
    # Impedance along the channel. A pair's off length is judged against its whole route,
    # including the stretches where P runs without N (no Zdiff there).
    own = [sec for sec in res.sections if sec.nets[0] in (pos if pair else (ch.net,))]

    def length(sec):                    # the leg's own (a pair's legs differ on bends)
        return sec.segs[0].length_mm if sec.segs else sec.length_mm
    vals = [(sec.zdiff if pair else sec.z0[0], length(sec)) for sec in own
            if not pair or sec.zdiff is not None]
    if vals:
        z = np.array([a for a, _ in vals])
        w = np.array([b for _, b in vals])
        total = sum(length(sec) for sec in own)
        tol = s.z_tol_pct / 100
        off = w[np.abs(z - target) > tol * target].sum()
        st = PASS if off == 0 else WARN if off < 0.1 * total else FAIL
        c.append(Finding("impedance", st, f"{'Zdiff' if pair else 'Z0'} {z.min():.1f}–"
                                          f"{z.max():.1f} Ω (target {target:g} Ω ± "
                                          f"{s.z_tol_pct:g} %), off over {off:.1f} of "
                                          f"{total:.1f} mm"))
    f_ny = res.nyquist_hz
    k = int(np.argmin(np.abs(res.freqs - f_ny)))
    il_lim = ch.value("il_max_db")
    if il_lim is not None and res.il_db is not None:
        il = float(res.il_db[k])
        st = PASS if il >= il_lim else WARN if il >= il_lim - 3 else FAIL
        c.append(Finding("insertion loss", st, f"{il:.2f} dB at Nyquist "
                                               f"({f_ny / 1e9:.3g} GHz), limit {il_lim:g} dB"))
    rl_lim = ch.value("rl_max_db")
    if rl_lim is not None and res.rl_db is not None:
        sel = res.freqs <= f_ny
        rl = float(res.rl_db[sel].max()) if sel.any() else float(res.rl_db[0])
        st = PASS if rl <= rl_lim else WARN if rl <= rl_lim + 3 else FAIL
        c.append(Finding("return loss", st, f"worst {rl:.1f} dB up to Nyquist, limit "
                                            f"{rl_lim:g} dB"))
    if res.conv_db is not None:
        sel = res.freqs <= f_ny
        cv = float(res.conv_db[sel].max())
        c.append(Finding("mode conversion", PASS if cv < -25 else WARN if cv < -15 else FAIL,
                         f"Scd21 worst {cv:.1f} dB up to Nyquist"))
    if res.skew_s is not None:
        ui = 1.0 / ch.value("rate_bps")
        frac = res.skew_s / ui
        c.append(Finding("skew", PASS if frac < 0.05 else WARN if frac < 0.15 else FAIL,
                         f"P/N skew {res.skew_s * 1e12:.2f} ps ({100 * frac:.1f} % of a UI)"))
    if res.eye is not None:
        eh, ew = ch.value("eye_h_v"), ch.value("eye_w_ui")
        e = res.eye
        txt = (f"eye {1e3 * e.height_v:.1f} mV × {e.width_ui:.2f} UI at BER {e.ber:g}"
               + (f" with {e.eq.describe()}" if e.eq.describe() != "none" else ""))
        if eh is not None and ew is not None:
            ok = e.height_v >= eh and e.width_ui >= ew
            marginal = e.height_v >= 0.7 * eh and e.width_ui >= 0.7 * ew
            c.append(Finding("eye", PASS if ok else WARN if marginal else FAIL,
                             txt + f"; mask {1e3 * eh:g} mV × {ew:g} UI"))
        else:
            c.append(Finding("eye", PASS if e.height_v > 0 else FAIL, txt))
    if res.com_db is not None and proto.diff:
        c.append(Finding("COM", PASS if res.com_db >= 3 else WARN if res.com_db >= 0 else FAIL,
                         f"COM {res.com_db:.2f} dB (≥ 3 dB passes)"))
    if res.waves is not None:
        key = "RX diff" if pair else next(k for k in res.waves.v if k.startswith("RX"))
        v = res.waves.v[key]
        vt = res.waves.t
        mid = 0.5 * (v.max() + v.min())
        v_high = float(np.median(v[v >= mid]))       # the settled levels: most samples
        v_low = float(np.median(v[v < mid])) if (v < mid).any() else float(v.min())
        sw = max(v_high - v_low, 1e-9)
        vil, vih = v_low + 0.35 * sw, v_low + 0.65 * sw
        m = edge_metrics(vt, v, v_low, v_high, vil, vih)
        over = m["overshoot_v"] / sw
        c.append(Finding("overshoot", PASS if over < 0.1 else WARN if over < 0.2 else FAIL,
                         f"overshoot {1e3 * m['overshoot_v']:.0f} mV "
                         f"({100 * over:.0f} % of the swing), undershoot "
                         f"{1e3 * m['undershoot_v']:.0f} mV"))
        if math.isfinite(m["ringback_v"]):
            rb = m["ringback_v"]
            c.append(Finding("ringback", PASS if rb > 0.05 * sw else WARN if rb > 0 else FAIL,
                             f"ringback margin {1e3 * rb:.0f} mV to the thresholds"))
        c.append(Finding("monotonic", PASS if m["monotonic"] else WARN,
                         "edges monotonic through the thresholds" if m["monotonic"]
                         else "an edge reverses inside the threshold band"))
    for a, (nxt, fxt) in res.xtalk_db.items():
        sel = res.freqs <= res.nyquist_hz
        worst = max(float(nxt[sel].max()), float(fxt[sel].max()))
        c.append(Finding("crosstalk", PASS if worst < -30 else WARN if worst < -20 else FAIL,
                         f"from {a}: NEXT/FEXT worst {worst:.1f} dB up to Nyquist"))
