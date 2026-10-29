"""Interface presets: data rate, impedance target, transmitter, equaliser range and the
limits a channel is checked against.

The limits are typical board-channel budgets taken from the public parts of each
standard (or common design guides where a standard's own budget is for the whole link),
for a quick pass or fail; a design that must comply should be checked against the
standard's own channel and compliance figures. Every value can be changed in a channel's
settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Protocol:
    key: str
    label: str
    rate_bps: float                         # line rate
    diff: bool = True
    z_target_ohm: float = 100.0             # Zdiff for a pair, Z0 single-ended
    swing_v: float = 0.8                    # transmitter swing, peak to peak differential
    rise_s: float = 40e-12                  # 20–80 % edge at the transmitter
    r_source_ohm: float = 50.0              # per leg
    r_term_ohm: float = 50.0                # per leg, to ground (or Vtt)
    c_rx_f: float = 0.5e-12                 # receiver pad capacitance
    il_max_db: Optional[float] = None       # insertion loss limit at Nyquist (board)
    rl_max_db: Optional[float] = -10.0      # return loss limit up to Nyquist
    eye_h_v: Optional[float] = None         # receiver eye mask height
    eye_w_ui: Optional[float] = None        # and width
    ber: float = 1e-12
    ffe_presets: Tuple[Tuple[float, ...], ...] = ((1.0,),)
    ctle_range_db: Tuple[float, float] = (0.0, 0.0)
    dfe_taps: int = 0
    rj_ui: float = 0.01                     # random jitter (rms) and deterministic,
    dj_ui: float = 0.05                     # shares of a UI
    rj_s: Optional[float] = None            # or in seconds, where a share of a long UI
    dj_s: Optional[float] = None            # would be more than the edge (slow logic)
    noise_v: float = 0.002
    note: str = ""

    @property
    def ui_s(self) -> float:
        return 1.0 / self.rate_bps

    @property
    def nyquist_hz(self) -> float:
        return self.rate_bps / 2

    def jitter_ui(self, rate_bps: Optional[float] = None) -> Tuple[float, float]:
        """(random rms, deterministic) jitter in UI at ``rate_bps`` (the interface's)."""
        rate = rate_bps or self.rate_bps
        rj = self.rj_s * rate if self.rj_s is not None else self.rj_ui
        dj = self.dj_s * rate if self.dj_s is not None else self.dj_ui
        return rj, dj


def _de(*db):
    """FFE presets from de-emphasis in dB: [c0, c1] with c1 = −(1 − 10^(−dB/20))/2
    normalised so |c0| + |c1| = 1."""
    out = [(1.0,)]
    for d in db:
        r = 10 ** (-d / 20)
        c1 = -(1 - r) / (1 + r)
        out.append((1 + c1, c1))
    return tuple(out)


def _ps_de(*presets):
    """FFE presets from (preshoot, de-emphasis) pairs in dB, as PCIe Gen 3 and USB4 give
    them: taps [c−1, c0, c+1] with |c−1| + c0 + |c+1| = 1, preshoot
    20·log10((c0 + |c−1| − |c+1|)/(c0 − |c−1| − |c+1|)) and de-emphasis
    20·log10((c0 − |c−1| + |c+1|)/(c0 − |c−1| − |c+1|))."""
    out = []
    for ps, de in presets:
        r_ps, r_de = 10 ** (ps / 20), 10 ** (-de / 20)
        x = 1 / (1 + r_de * (r_ps - 1))         # 1 − 2|c−1|
        pre = (1 - x) / 2
        post = (1 - r_ps * r_de * x) / 2
        # Only the taps in use: no pre-cursor without preshoot, no post-cursor without
        # de-emphasis.
        out.append(((-pre,) if pre > 1e-9 else ()) + (1 - pre - post,)
                   + ((-post,) if post > 1e-9 else ()))
    return tuple(out)


# A spread of the USB4 transmitter's presets (preshoot, de-emphasis): the searches pick
# the one that opens the eye most.
_USB4_PRESETS = _ps_de((0, 0), (0, 2), (0, 3.5), (0, 6), (2, 0), (2, 3.5), (3.5, 0),
                       (3.5, 3.5), (3.5, 6))


PROTOCOLS: Dict[str, Protocol] = {p.key: p for p in [
    Protocol("custom", "Custom", 1e9, note="your own rate and limits"),
    Protocol("lvds", "LVDS (1 Gb/s)", 1e9, z_target_ohm=100, swing_v=0.7, rise_s=150e-12,
             r_term_ohm=50, il_max_db=-6, eye_h_v=0.1, eye_w_ui=0.6),
    Protocol("usb2", "USB 2.0 High Speed", 480e6, z_target_ohm=90, swing_v=0.8, rise_s=300e-12,
             r_source_ohm=45, r_term_ohm=45, il_max_db=-3, eye_h_v=0.3, eye_w_ui=0.7),
    Protocol("usb3g1", "USB 3.2 Gen 1 (5 Gb/s)", 5e9, z_target_ohm=90, swing_v=1.0,
             rise_s=60e-12, il_max_db=-8, eye_h_v=0.1, eye_w_ui=0.45,
             ffe_presets=_de(3.5), ctle_range_db=(0, -9)),
    Protocol("usb3g2", "USB 3.2 Gen 2 (10 Gb/s)", 10e9, z_target_ohm=90, swing_v=1.0,
             rise_s=35e-12, il_max_db=-12, eye_h_v=0.07, eye_w_ui=0.4,
             ffe_presets=_de(3.5, 6), ctle_range_db=(0, -12), dfe_taps=1),
    Protocol("usb4g2", "USB4 Gen 2 (10 Gb/s)", 10.3125e9, z_target_ohm=85, swing_v=1.0,
             rise_s=25e-12, r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-8,
             eye_h_v=0.05, eye_w_ui=0.35, ffe_presets=_USB4_PRESETS,
             ctle_range_db=(0, -12), dfe_taps=1),
    Protocol("usb4g3", "USB4 Gen 3 (20 Gb/s)", 20.625e9, z_target_ohm=85, swing_v=1.0,
             rise_s=15e-12, r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-10,
             eye_h_v=0.04, eye_w_ui=0.3, ffe_presets=_USB4_PRESETS,
             ctle_range_db=(0, -15), dfe_taps=2,
             note="per lane; Gen 4 (PAM3) is not modelled"),
    Protocol("pcie2", "PCIe Gen 2 (5 GT/s)", 5e9, z_target_ohm=85, swing_v=1.0, rise_s=40e-12,
             r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-8, eye_h_v=0.12, eye_w_ui=0.4,
             ffe_presets=_ps_de((0, 3.5), (0, 6)), ctle_range_db=(0, -6)),
    Protocol("pcie3", "PCIe Gen 3 (8 GT/s)", 8e9, z_target_ohm=85, swing_v=1.0, rise_s=40e-12,
             r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-10, eye_h_v=0.025, eye_w_ui=0.3,
             ffe_presets=_de(3.5, 6, 9.5), ctle_range_db=(0, -12), dfe_taps=1),
    Protocol("pcie4", "PCIe Gen 4 (16 GT/s)", 16e9, z_target_ohm=85, swing_v=1.0,
             rise_s=25e-12, r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-12,
             eye_h_v=0.015, eye_w_ui=0.3, ffe_presets=_de(3.5, 6, 9.5),
             ctle_range_db=(0, -15), dfe_taps=2),
    Protocol("pcie5", "PCIe Gen 5 (32 GT/s)", 32e9, z_target_ohm=85, swing_v=1.0,
             rise_s=15e-12, r_source_ohm=42.5, r_term_ohm=42.5, il_max_db=-14,
             eye_h_v=0.01, eye_w_ui=0.3, ffe_presets=_de(3.5, 6, 9.5),
             ctle_range_db=(0, -15), dfe_taps=3),
    Protocol("sata3", "SATA III (6 Gb/s)", 6e9, z_target_ohm=100, swing_v=0.6, rise_s=50e-12,
             il_max_db=-8, eye_h_v=0.24, eye_w_ui=0.4, ffe_presets=_de(3.5),
             ctle_range_db=(0, -6)),
    Protocol("hdmi20", "HDMI 2.0 TMDS (6 Gb/s)", 6e9, z_target_ohm=100, swing_v=0.8,
             rise_s=50e-12, il_max_db=-6, eye_h_v=0.15, eye_w_ui=0.6, ctle_range_db=(0, -6)),
    Protocol("sgmii", "SGMII / 1000BASE-X (1.25 Gb/s)", 1.25e9, z_target_ohm=100, swing_v=0.8,
             rise_s=100e-12, il_max_db=-4, eye_h_v=0.2, eye_w_ui=0.6),
    Protocol("10gkr", "10GBASE-KR (10.3125 Gb/s)", 10.3125e9, z_target_ohm=100, swing_v=1.0,
             rise_s=30e-12, il_max_db=-15, eye_h_v=0.03, eye_w_ui=0.3,
             ffe_presets=_de(3.5, 6, 9.5), ctle_range_db=(0, -12), dfe_taps=5),
    Protocol("mipi_dphy", "MIPI D-PHY (2.5 Gb/s)", 2.5e9, z_target_ohm=100, swing_v=0.4,
             rise_s=100e-12, il_max_db=-5, eye_h_v=0.07, eye_w_ui=0.5),
    Protocol("ddr4_dq", "DDR4 DQ (3200 MT/s)", 3.2e9, diff=False, z_target_ohm=40, swing_v=1.2,
             rise_s=80e-12, r_source_ohm=34, r_term_ohm=48, il_max_db=-3, rl_max_db=-8,
             eye_h_v=0.13, eye_w_ui=0.5),
    Protocol("cmos", "CMOS logic (100 Mb/s)", 100e6, diff=False, z_target_ohm=50, swing_v=3.3,
             rise_s=1e-9, r_source_ohm=25, r_term_ohm=1e6, c_rx_f=3e-12, il_max_db=None,
             rl_max_db=None, rj_s=5e-12, dj_s=50e-12,
             note="ringing and overshoot matter more than loss"),
]}
