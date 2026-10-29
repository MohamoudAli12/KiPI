"""The Signal Integrity part of the user's setup: channels and settings.

A **channel** is a net (or a differential pair), followed through any series parts (an
AC-coupling capacitor, a series resistor) to the nets beyond, from a driving part to a
receiving one. It names its interface (:mod:`kipi.si.protocols`), which sets the rate,
swing, edges, terminations, equaliser and limits, any of which it can override, and the
buffers at each end: generic ones (a supply, an output resistance, an edge, a pad
capacitance) or IBIS models. Aggressor nets add their crosstalk.

These live in the same setup file as the rest of KiPI's setup. Nothing here imports the
DC model, so the setup can import this module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from kipi.si.protocols import PROTOCOLS, Protocol

ROUGHNESS_MODELS = (("smooth", "Smooth"), ("hammerstad", "Hammerstad (RMS)"),
                    ("groisse", "Groisse (RMS)"), ("huray", "Huray (snowball)"))
CORNERS = (("typ", "Typical"), ("min", "Slow (min)"), ("max", "Fast (max)"))


@dataclass
class Buffer:
    """A channel end's buffer: ``generic`` (from the values here) or ``ibis`` (a model
    in an IBIS file, at a corner)."""
    kind: str = "generic"
    vcc_v: float = 3.3
    r_out_ohm: float = 33.0
    edge_s: float = 0.5e-9               # 0–100 % of a generic edge
    c_pf: float = 2.0
    ibis_file: str = ""
    ibis_model: str = ""
    ibis_pin: str = ""                   # for its package parasitics; "" the [Package]'s
    corner: str = "typ"


@dataclass
class Channel:
    name: str
    net: str                             # the first net (a pair's positive leg)
    partner: str = ""                    # a pair's negative leg; "" single-ended
    protocol: str = "custom"
    driver_ref: str = ""                 # "" guessed: one end of the net
    receiver_ref: str = ""
    driver: Buffer = field(default_factory=Buffer)
    receiver: Buffer = field(default_factory=Buffer)
    # Overrides of the protocol's values (None: the protocol's).
    rate_bps: Optional[float] = None
    swing_v: Optional[float] = None
    rise_s: Optional[float] = None
    r_source_ohm: Optional[float] = None
    r_term_ohm: Optional[float] = None
    v_term_v: float = 0.0
    series_r_ohm: float = 0.0            # at the driver
    il_max_db: Optional[float] = None
    rl_max_db: Optional[float] = None
    eye_h_v: Optional[float] = None
    eye_w_ui: Optional[float] = None
    aggressors: List[str] = field(default_factory=list)
    follow_series: bool = True
    not_fitted: List[str] = field(default_factory=list)   # parts left off (a variant's)
    ticked: bool = False                 # in the Channels list: solved by Solve
    prbs: int = 7
    use_ibis_eye: bool = False           # eye from the IBIS transient, not the linear model
    equalize: bool = True

    def __post_init__(self):
        self.name = self.name or (f"{self.net}/{self.partner}" if self.partner else self.net)

    @property
    def is_pair(self) -> bool:
        return bool(self.partner)

    def proto(self) -> Protocol:
        return PROTOCOLS.get(self.protocol, PROTOCOLS["custom"])

    def value(self, key: str):
        own = getattr(self, key, None)
        return own if own is not None else getattr(self.proto(), key)


@dataclass
class SISettings:
    # Cross-sections
    segment_mm: float = 1.0
    window_mm: float = 1.5
    etch_mm: float = 0.0                 # a trace's top narrower by this on each side
    roughness: str = "huray"
    rq_um: float = 1.0
    huray_radius_um: float = 0.5
    huray_sr: float = 1.5
    via_models: bool = True
    # The build solved: a KiCad 10 assembly variant's name, or "" for the board's own DNP.
    variant: str = ""
    # Sweep
    f_max_hz: Optional[float] = None     # None: from the rate (at least 5 × Nyquist)
    df_hz: float = 20e6
    # Quick Scan
    scan_segment_mm: float = 2.0
    z_target_ohm: float = 50.0
    z_tol_pct: float = 10.0
    diff_target_ohm: Optional[float] = None
    kb_limit: float = 0.03
    scan_f_max_hz: float = 10e9
    skew_limit_mm: float = 0.15
    stitch_radius_mm: float = 2.0
    scan_nets: str = "all"               # "all" signals, or "scope": the nets ticked
    scan_slow: bool = False              # "all" also takes the slow control nets
    # 3D (openEMS)
    em_f_max_hz: float = 20e9
    em_fine_mm: float = 0.05
    em_margin_mm: float = 2.0
    em_end_db: float = -40.0
    em_threads: int = 0
    report_format: str = "html"


def buffer_from_dict(d: Optional[dict]) -> Buffer:
    known = {f.name for f in dataclasses.fields(Buffer)}
    return Buffer(**{k: v for k, v in (d or {}).items() if k in known})


def channel_from_dict(d: dict) -> Channel:
    known = {f.name for f in dataclasses.fields(Channel)} - {"driver", "receiver"}
    kw = {k: v for k, v in d.items() if k in known}
    kw["aggressors"] = [str(x) for x in d.get("aggressors", [])]
    kw["not_fitted"] = [str(x) for x in d.get("not_fitted", [])]
    if kw.get("protocol") not in PROTOCOLS:
        kw["protocol"] = "custom"
    return Channel(driver=buffer_from_dict(d.get("driver")),
                   receiver=buffer_from_dict(d.get("receiver")), **kw)


def settings_from_dict(d: Optional[dict]) -> SISettings:
    known = {f.name for f in dataclasses.fields(SISettings)}
    return SISettings(**{k: v for k, v in (d or {}).items() if k in known})


def new_channel(net: str, partner: str = "", protocol: Optional[str] = None) -> Channel:
    """A channel for ``net`` (and ``partner``), its interface guessed from the name."""
    from kipi.si.pairs import by_name, diff_target
    if protocol is None:
        guesses = (("usb4", "usb4g3"), ("tbt", "usb4g3"), ("thunderbolt", "usb4g3"),
                   ("usb", "usb2"), ("pcie", "pcie3"), ("sata", "sata3"), ("hdmi", "hdmi20"),
                   ("tmds", "hdmi20"), ("lvds", "lvds"), ("mipi", "mipi_dphy"),
                   ("csi", "mipi_dphy"), ("dsi", "mipi_dphy"), ("sgmii", "sgmii"),
                   ("dq", "ddr4_dq"))
        protocol = by_name(net, guesses, "pcie3") or ("lvds" if partner else "cmos")
    ch = Channel("", net, partner, protocol)
    if partner and protocol == "custom":
        ch.r_term_ohm = diff_target(net) / 2
    match_generic(ch)
    return ch


def not_fitted(snap, s: SISettings, ch: Channel) -> List[str]:
    """The parts ``ch`` does not go through: the board's DNP in the build ``s`` solves (a
    variant the board does not have: its own DNP) and the channel's own list."""
    variant = s.variant if s.variant in getattr(snap, "variants", {}) else ""
    board = snap.dnp_refs(variant) if snap is not None else []
    return sorted(set(board) | set(ch.not_fitted))


def match_generic(ch: Channel):
    """Set ``ch``'s generic buffers to its interface: the driver's output resistance,
    edge and supply give the interface's swing into its termination (a pair's legs
    each swing half of it), and both ends' capacitance is the interface's pad's (a
    SerDes pad's half a picofarad, not a CMOS pin's few)."""
    p = ch.proto()
    rs, rt = ch.value("r_source_ohm"), ch.value("r_term_ohm")
    d = ch.driver
    d.r_out_ohm = rs
    d.edge_s = ch.value("rise_s") / 0.6
    if ch.partner and rt < 1e5:
        d.vcc_v = round(0.5 * ch.value("swing_v") * (rs + rt) / rt, 3)
    elif not ch.partner:
        d.vcc_v = ch.value("swing_v")
    ch.receiver.vcc_v = d.vcc_v
    d.c_pf = ch.receiver.c_pf = p.c_rx_f * 1e12
