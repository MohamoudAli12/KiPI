"""IBIS models (I/O Buffer Information Specification, versions 1–7): a reader for the
parts a board simulation uses, and the behavioural buffer they describe.

**Read**: [Component] with its [Package] and [Pin] list (per-pin R/L/C), [Model Selector],
and each [Model]: Model_type, C_comp, Vinl/Vinh ([Model Spec] too), the reference
voltages, the I-V tables ([Pulldown], [Pullup], [GND Clamp], [POWER Clamp]), [Ramp]
and the [Rising Waveform]/[Falling Waveform] V-t tables with their fixtures. Every number
takes IBIS's scale suffixes (T G M k m u n p f; ``M`` is mega) and trailing units; NA
values fall back to the typical column.

**Behaviour** (the standard "switching coefficient" buffer). The current the buffer takes
in at its die pad is

    I(V, t) = Ku(t)·Ipu(Vpu − V) + Kd(t)·Ipd(V − Vpd) + Igc(V − Vgc) + Ipc(Vpc − V),

with C_comp from the pad to ground. Ku and Kd go from the low state (Ku = 0, Kd = 1) to
the high one over an edge. With two V-t waveforms per edge they are solved from both at
once (2 equations, 2 unknowns, each sample: the fixture's current into the pad
equals C_comp·dV/dt plus I); with one, Kd = 1 − Ku is assumed; with neither, [Ramp]'s
20–80 % slew makes a linear edge.

A generic buffer (:func:`generic_driver`) is made the same way from a supply, an output
resistance and an edge time, for a net with no IBIS model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CORNERS = ("typ", "min", "max")
_SCALE = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3, "m": 1e-3, "u": 1e-6,
          "n": 1e-9, "p": 1e-12, "f": 1e-15}
_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([TGMkKmunpf]?)")


class IbisError(ValueError):
    pass


def number(text: str) -> Optional[float]:
    """An IBIS number ("1.5nH", "-2.3mA", "4.7k", "NA"); None for NA or unreadable."""
    t = (text or "").strip()
    if not t or t.upper() == "NA":
        return None
    m = _NUM.match(t)
    if not m:
        return None
    v = float(m.group(1))
    return v * _SCALE.get(m.group(2), 1.0) if m.group(2) else v


def _triple(tokens: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [number(t) for t in tokens[:3]] + [None] * (3 - len(tokens[:3]))
    typ = vals[0]
    return typ, vals[1] if vals[1] is not None else typ, vals[2] if vals[2] is not None else typ


@dataclass
class Table:
    """An I-V (x = V) or V-t (x = t) table: x and one column per corner."""
    x: np.ndarray
    y: Dict[str, np.ndarray]

    def at(self, corner: str, x) -> np.ndarray:
        return np.interp(x, self.x, self.y.get(corner, self.y["typ"]))

    def slope_at(self, corner: str, x) -> np.ndarray:
        """dy/dx (for Newton), from the table's slopes, computed once."""
        cache = self.__dict__.setdefault("_slopes", {})
        dy = cache.get(corner)
        if dy is None:
            y = self.y.get(corner, self.y["typ"])
            dy = cache[corner] = (np.gradient(y, self.x) if len(self.x) > 1
                                  else np.zeros_like(y))
        return np.interp(x, self.x, dy)


@dataclass
class Waveform:
    table: Table
    r_fixture: float = 50.0
    v_fixture: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    c_fixture: float = 0.0


@dataclass
class Model:
    name: str
    model_type: str = "I/O"
    c_comp: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    voltage_range: Optional[Tuple[float, float, float]] = None
    pullup_ref: Optional[Tuple[float, float, float]] = None
    pulldown_ref: Optional[Tuple[float, float, float]] = None
    power_clamp_ref: Optional[Tuple[float, float, float]] = None
    gnd_clamp_ref: Optional[Tuple[float, float, float]] = None
    vinl: Optional[float] = None
    vinh: Optional[float] = None
    pulldown: Optional[Table] = None
    pullup: Optional[Table] = None
    gnd_clamp: Optional[Table] = None
    power_clamp: Optional[Table] = None
    ramp_r: Optional[Tuple[float, float]] = None       # (dV, dt) typical 20-80 %
    ramp_f: Optional[Tuple[float, float]] = None
    rising: List[Waveform] = field(default_factory=list)
    falling: List[Waveform] = field(default_factory=list)

    @property
    def is_driver(self) -> bool:
        return self.pullup is not None or self.pulldown is not None

    def _ref(self, which, corner) -> float:
        i = CORNERS.index(corner)
        own = getattr(self, which)
        if own is not None:
            return own[i]
        if which in ("pulldown_ref", "gnd_clamp_ref"):
            return 0.0
        return self.voltage_range[i] if self.voltage_range else 0.0

    def vcc(self, corner="typ") -> float:
        return self._ref("pullup_ref", corner)

    def thresholds(self, corner="typ") -> Tuple[float, float]:
        vcc = self.vcc(corner)
        vil = self.vinl if self.vinl is not None else 0.35 * vcc
        vih = self.vinh if self.vinh is not None else 0.65 * vcc
        return vil, vih


@dataclass
class PinInfo:
    pin: str
    signal: str
    model: str
    r: Optional[float] = None
    l: Optional[float] = None
    c: Optional[float] = None


@dataclass
class IbisFile:
    path: str
    component: str = ""
    manufacturer: str = ""
    package: Tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))   # R, L, C
    pins: Dict[str, PinInfo] = field(default_factory=dict)
    models: Dict[str, Model] = field(default_factory=dict)
    selectors: Dict[str, List[str]] = field(default_factory=dict)

    def model_names(self) -> List[str]:
        return list(self.models)

    def model_for_pin(self, pin: str) -> Optional[str]:
        p = self.pins.get(pin)
        if p is None:
            return None
        if p.model in self.selectors:
            return self.selectors[p.model][0]
        return p.model

    def package_rlc(self, pin: Optional[str] = None, corner="typ") -> Tuple[float, float, float]:
        i = CORNERS.index(corner)
        r, l, c = (self.package[0][i] or 0.0, self.package[1][i] or 0.0,
                   self.package[2][i] or 0.0)
        p = self.pins.get(pin) if pin else None
        if p is not None:
            r = p.r if p.r is not None else r
            l = p.l if p.l is not None else l
            c = p.c if p.c is not None else c
        return r, l, c


# ------------------------------------------------------------------ parser
def read(path) -> IbisFile:
    text = Path(path).read_text(encoding="latin-1")
    return parse(text, str(path))


def parse(text: str, path: str = "") -> IbisFile:
    out = IbisFile(path)
    comment = "|"
    blocks: List[Tuple[str, str, List[List[str]]]] = []
    cur = None
    for raw in text.splitlines():
        line = raw
        if line.lower().startswith("[comment char]"):
            rest = line.split("]", 1)[1].strip()
            if rest:
                comment = rest[0]
            continue
        if comment in line:
            line = line[:line.index(comment)]
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            kw, _, rest = line[1:].partition("]")
            cur = (kw.strip().lower().replace("_", " "), rest.strip(), [])
            blocks.append(cur)
        elif cur is not None:
            cur[2].append(line.replace("=", " = ").split())
    model: Optional[Model] = None
    in_pin = False
    for kw, arg, rows in blocks:
        in_pin = kw == "pin"
        if kw == "component":
            out.component = arg
        elif kw == "manufacturer":
            out.manufacturer = arg
        elif kw == "package":
            vals = {}
            for r in rows:
                if r and r[0].lower() in ("r_pkg", "l_pkg", "c_pkg"):
                    vals[r[0].lower()] = _triple(r[1:])
            out.package = (vals.get("r_pkg", (0.0,) * 3), vals.get("l_pkg", (0.0,) * 3),
                           vals.get("c_pkg", (0.0,) * 3))
        elif in_pin:
            for r in rows:
                if len(r) >= 3:
                    pi = PinInfo(r[0], r[1], r[2])
                    if len(r) >= 6:
                        pi.r, pi.l, pi.c = number(r[3]), number(r[4]), number(r[5])
                    out.pins[r[0]] = pi
        elif kw == "model selector":
            out.selectors[arg] = [r[0] for r in rows if r]
        elif kw == "model":
            model = Model(arg)
            out.models[arg] = model
            for r in rows:
                _model_param(model, r)
        elif model is not None:
            _model_block(model, kw, arg, rows)
    if not out.models:
        raise IbisError("no [Model] in the file")
    return out


def _model_param(m: Model, r: List[str]):
    if not r:
        return
    key = r[0].lower()
    vals = [t for t in r[1:] if t != "="]
    if key == "model_type" and vals:
        m.model_type = vals[0]
    elif key == "c_comp":
        m.c_comp = tuple(v or 0.0 for v in _triple(vals))
    elif key == "vinl" and vals:
        m.vinl = number(vals[0])
    elif key == "vinh" and vals:
        m.vinh = number(vals[0])


def _table(rows: List[List[str]], ncol: int = 3) -> Optional[Table]:
    xs, cols = [], {c: [] for c in CORNERS}
    for r in rows:
        if len(r) < 2:
            continue
        x = number(r[0])
        if x is None:
            continue
        t, mn, mx = _triple(r[1:1 + ncol])
        if t is None:
            continue
        xs.append(x)
        cols["typ"].append(t)
        cols["min"].append(mn)
        cols["max"].append(mx)
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    x = np.asarray(xs)[order]
    x, keep = np.unique(x, return_index=True)
    return Table(x, {c: np.asarray(v)[order][keep] for c, v in cols.items()})


def _model_block(m: Model, kw: str, arg: str, rows: List[List[str]]):
    if kw == "model spec":
        for r in rows:
            if r and r[0].lower() == "vinl":
                m.vinl = number(r[1])
            elif r and r[0].lower() == "vinh":
                m.vinh = number(r[1])
        return
    if kw in ("voltage range", "pullup reference", "pulldown reference",
              "power clamp reference", "gnd clamp reference"):
        toks = arg.split() + [t for r in rows for t in r]
        tr = _triple(toks)
        attr = {"voltage range": "voltage_range", "pullup reference": "pullup_ref",
                "pulldown reference": "pulldown_ref", "power clamp reference": "power_clamp_ref",
                "gnd clamp reference": "gnd_clamp_ref"}[kw]
        setattr(m, attr, tuple(v or 0.0 for v in tr))
        return
    if kw in ("pulldown", "pullup", "gnd clamp", "power clamp"):
        setattr(m, kw.replace(" ", "_"), _table(rows))
        return
    if kw == "ramp":
        for r in rows:
            if not r:
                continue
            key = r[0].lower()
            if key in ("dv/dt_r", "dv/dt_f") and len(r) > 1:
                dv, _, dt = r[1].partition("/")
                pair = (number(dv), number(dt))
                if pair[0] is not None and pair[1]:
                    setattr(m, "ramp_r" if key.endswith("r") else "ramp_f", pair)
        return
    if kw in ("rising waveform", "falling waveform"):
        fix = {"r_fixture": 50.0, "v_fixture": 0.0, "v_fixture_min": None,
               "v_fixture_max": None, "c_fixture": 0.0}
        data = []
        for r in rows:
            k = r[0].lower()
            if k in fix and len(r) >= 3:
                fix[k] = number(r[2])
            else:
                data.append(r)
        t = _table(data)
        if t is None:
            return
        vf = fix["v_fixture"] or 0.0
        wf = Waveform(t, fix["r_fixture"] or 50.0,
                      (vf, fix["v_fixture_min"] if fix["v_fixture_min"] is not None else vf,
                       fix["v_fixture_max"] if fix["v_fixture_max"] is not None else vf),
                      fix["c_fixture"] or 0.0)
        (m.rising if kw.startswith("rising") else m.falling).append(wf)


# -------------------------------------------------------------- behaviour
@dataclass
class Buffer:
    """A model at one corner, ready for a transient: the pad current I(V, t) and dI/dV."""
    model: Model
    corner: str = "typ"
    ku_r: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (t, Ku) over a rising edge
    kd_r: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ku_f: Optional[Tuple[np.ndarray, np.ndarray]] = None
    kd_f: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def c_comp(self) -> float:
        return self.model.c_comp[CORNERS.index(self.corner)] or 0.0

    def static(self, v):
        """The clamps' current and its slope at pad voltage ``v``."""
        m, c = self.model, self.corner
        i = np.zeros_like(np.asarray(v, dtype=float))
        g = np.zeros_like(i)
        if m.gnd_clamp is not None:
            x = v - m._ref("gnd_clamp_ref", c)
            i = i + m.gnd_clamp.at(c, x)
            g = g + m.gnd_clamp.slope_at(c, x)
        if m.power_clamp is not None:
            x = m._ref("power_clamp_ref", c) - v
            i = i + m.power_clamp.at(c, x)
            g = g - m.power_clamp.slope_at(c, x)
        return i, g

    def drive(self, v, ku, kd):
        """The driver's current and slope at pad voltage ``v`` with coefficients ku, kd."""
        m, c = self.model, self.corner
        i, g = self.static(v)
        if m.pullup is not None:
            x = m._ref("pullup_ref", c) - v
            i = i + ku * m.pullup.at(c, x)
            g = g - ku * m.pullup.slope_at(c, x)
        if m.pulldown is not None:
            x = v - m._ref("pulldown_ref", c)
            i = i + kd * m.pulldown.at(c, x)
            g = g + kd * m.pulldown.slope_at(c, x)
        return i, g

    def coefficients(self, edge: str, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Ku, Kd) at times ``t`` after an edge ("rise" or "fall") starts."""
        ku, kd = (self.ku_r, self.kd_r) if edge == "rise" else (self.ku_f, self.kd_f)
        return np.interp(t, *ku), np.interp(t, *kd)

    def edge_time(self, edge: str) -> float:
        ku = self.ku_r if edge == "rise" else self.ku_f
        return float(ku[0][-1])


def buffer(model: Model, corner: str = "typ") -> Buffer:
    """The model's switching coefficients at ``corner``, from its waveforms or ramp."""
    b = Buffer(model, corner)
    if not model.is_driver:
        return b
    for edge, wfs, ramp in (("rise", model.rising, model.ramp_r),
                            ("fall", model.falling, model.ramp_f)):
        if len(wfs) >= 2:
            ku, kd = _two_waveforms(b, wfs[:2], edge)
        elif len(wfs) == 1:
            ku, kd = _one_waveform(b, wfs[0], edge)
        else:
            ku, kd = _ramp(b, ramp, edge)
        if edge == "rise":
            b.ku_r, b.kd_r = ku, kd
        else:
            b.ku_f, b.kd_f = ku, kd
    return b


def _grid(wfs: List[Waveform], c: str, n: int = 400):
    t_end = max(w.table.x[-1] for w in wfs)
    t = np.linspace(0.0, t_end, n)
    return t


def _fixture_terms(b: Buffer, w: Waveform, t: np.ndarray):
    c = b.corner
    v = w.table.at(c, t)
    dv = np.gradient(v, t)
    vf = w.v_fixture[CORNERS.index(c)]
    i_net = (vf - v) / w.r_fixture - (b.c_comp + w.c_fixture) * dv
    return v, i_net


def _two_waveforms(b: Buffer, wfs, edge):
    m, c = b.model, b.corner
    t = _grid(wfs, c)
    rows = []
    for w in wfs:
        v, i_net = _fixture_terms(b, w, t)
        ipu = m.pullup.at(c, m._ref("pullup_ref", c) - v) if m.pullup else np.zeros_like(v)
        ipd = m.pulldown.at(c, v - m._ref("pulldown_ref", c)) if m.pulldown else np.zeros_like(v)
        icl, _ = b.static(v)
        rows.append((ipu, ipd, i_net - icl))
    (a1, b1, r1), (a2, b2, r2) = rows
    det = a1 * b2 - a2 * b1
    ok = np.abs(det) > 1e-12 * (np.abs(a1 * b2) + np.abs(a2 * b1) + 1e-30)
    ku = np.where(ok, (r1 * b2 - r2 * b1) / np.where(ok, det, 1), np.nan)
    kd = np.where(ok, (a1 * r2 - a2 * r1) / np.where(ok, det, 1), np.nan)
    ku, kd = _fill(ku), _fill(kd)
    return _settle(t, ku, kd, edge)


def _one_waveform(b: Buffer, w: Waveform, edge):
    m, c = b.model, b.corner
    t = _grid([w], c)
    v, i_net = _fixture_terms(b, w, t)
    ipu = m.pullup.at(c, m._ref("pullup_ref", c) - v) if m.pullup else np.zeros_like(v)
    ipd = m.pulldown.at(c, v - m._ref("pulldown_ref", c)) if m.pulldown else np.zeros_like(v)
    icl, _ = b.static(v)
    # Kd = 1 − Ku:  Ku·(Ipu − Ipd) = i_net − Icl − Ipd
    den = ipu - ipd
    ku = np.where(np.abs(den) > 1e-12, (i_net - icl - ipd) / np.where(np.abs(den) > 1e-12, den, 1),
                  np.nan)
    ku = _fill(ku)
    return _settle(t, ku, 1 - ku, edge)


def _ramp(b: Buffer, ramp, edge):
    vcc = b.model.vcc(b.corner)
    if ramp is None:
        tr = 1e-9
    else:
        dv, dt = ramp
        tr = dt / 0.6 * (0.6 * vcc / dv if dv else 1.0)
    t = np.linspace(0, tr, 50)
    s = 0.5 - 0.5 * np.cos(np.pi * t / tr)
    ku = s if edge == "rise" else 1 - s
    return (t, ku), (t, 1 - ku)


def _fill(x):
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def _settle(t, ku, kd, edge):
    """Clip to sane values and pin the end states."""
    ku = np.clip(ku, -0.2, 1.2)
    kd = np.clip(kd, -0.2, 1.2)
    if edge == "rise":
        ku[0], kd[0] = 0.0, 1.0
        ku[-1], kd[-1] = 1.0, 0.0
    else:
        ku[0], kd[0] = 1.0, 0.0
        ku[-1], kd[-1] = 0.0, 1.0
    return (t, ku), (t, kd)


# ---------------------------------------------------------- generic buffers
def _linear_table(r_ohm: float, v_span: float, i_max: Optional[float] = None) -> Table:
    v = np.linspace(-v_span, 2 * v_span, 61)
    i = v / r_ohm
    if i_max:
        i = i_max * np.tanh(i / i_max)
    return Table(v, {c: i for c in CORNERS})


def _diode_table(v_span: float) -> Table:
    v = np.linspace(-v_span, v_span, 61)
    i = np.where(v < 0, -1e-12 * (np.exp(np.minimum(-v, 1.2) / 0.026) - 1), 0.0)
    i = np.where(v < -0.4, -(np.abs(v) - 0.4) / 5.0, i)
    return Table(v, {c: i for c in CORNERS})


def _clamps(vcc: float) -> Tuple[Table, Table]:
    """Diode clamps to ground and to the supply. Each table's x is how far the pad is past
    its rail inward (V − 0 for ground, Vcc − V for the supply); past −0.4 V the diode
    conducts: out of the part below ground, into it above the supply."""
    gnd = _diode_table(vcc)
    return gnd, Table(gnd.x, {c: -v for c, v in gnd.y.items()})


def generic_driver(vcc: float, r_out_ohm: float, edge_s: float, c_comp_f: float = 2e-12,
                   name: str = "generic") -> Model:
    """A push-pull CMOS driver: a resistive pull-up and pull-down of ``r_out_ohm``
    (saturating at the current a 5 Ω short would give), clamps to the rails, and a
    raised-cosine edge of ``edge_s`` (0–100 %)."""
    imax = vcc / max(r_out_ohm * 0.5, 1.0)
    m = Model(name, "I/O", (c_comp_f,) * 3, (vcc,) * 3)
    m.pulldown = _linear_table(r_out_ohm, vcc, imax)
    up = _linear_table(r_out_ohm, vcc, imax)
    m.pullup = Table(up.x, {c: -v for c, v in up.y.items()})      # current out when high
    m.gnd_clamp, m.power_clamp = _clamps(vcc)
    m.ramp_r = m.ramp_f = (0.6 * vcc, 0.6 * edge_s)
    m.vinl, m.vinh = 0.35 * vcc, 0.65 * vcc
    return m


def generic_receiver(vcc: float, c_comp_f: float = 2e-12, name: str = "generic_in") -> Model:
    m = Model(name, "Input", (c_comp_f,) * 3, (vcc,) * 3)
    m.gnd_clamp, m.power_clamp = _clamps(vcc)
    m.vinl, m.vinh = 0.35 * vcc, 0.65 * vcc
    return m
