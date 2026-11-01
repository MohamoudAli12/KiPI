"""IBIS parsing and buffers, the transient with buffers at the ports, and the eyes,
equalisers and COM against cases with known answers."""

import math

import numpy as np
import pytest

from kipi.si import eye as eyes
from kipi.si import ibis, sparams
from kipi.si.network import Circuit
from kipi.si.transient import Device, LinearPort, Stimulus, port_transfer, simulate

C0_MM_S = 299792458e3

IBS = """\
[IBIS Ver]   5.0
[File Name]  test.ibs
[Component]  TESTCHIP
[Manufacturer] Nobody
[Package]
| typ min max
R_pkg  0.2   0.1   0.3
L_pkg  2nH   1nH   3nH
C_pkg  0.5pF 0.4pF 0.6pF
[Pin]  signal_name  model_name  R_pin  L_pin  C_pin
1      DQ0          OUT50       0.5    3n     0.7p
2      GND          GND
[Model]  OUT50
Model_type  Output
C_comp  2pF  1.8pF  2.2pF
[Voltage Range]  3.3  3.0  3.6
[Pulldown]
| V   I(typ)  I(min)  I(max)
-3.3  -66m  -60m  -72m
0     0      0      0
3.3   66m    60m    72m
6.6   132m   120m   144m
[Pullup]
-3.3  66m   60m   72m
0     0     0     0
3.3   -66m  -60m  -72m
6.6   -132m -120m -144m
[Ramp]
dV/dt_r  1.98/0.5n  1.8/0.6n  2.16/0.4n
dV/dt_f  1.98/0.5n  1.8/0.6n  2.16/0.4n
R_load = 50
[End]
"""


@pytest.fixture
def ibs(tmp_path):
    p = tmp_path / "test.ibs"
    p.write_text(IBS)
    return ibis.read(p)


def test_number_suffixes():
    assert ibis.number("3n") == pytest.approx(3e-9)
    assert ibis.number("0.7pF") == pytest.approx(0.7e-12)
    assert ibis.number("66m") == pytest.approx(0.066)
    assert ibis.number("NA") is None


def test_parse(ibs):
    assert ibs.component == "TESTCHIP"
    assert "OUT50" in ibs.models
    assert ibs.model_for_pin("1") == "OUT50"
    r, l, c = ibs.package_rlc("1")
    assert (r, l, c) == pytest.approx((0.5, 3e-9, 0.7e-12))
    assert ibs.package_rlc(None)[1] == pytest.approx(2e-9)
    m = ibs.models["OUT50"]
    assert m.vcc() == pytest.approx(3.3)


def test_ibis_buffer_is_50_ohms(ibs):
    b = ibis.buffer(ibs.models["OUT50"])
    # Driving low (pull-down on): 3.3 V at the pin draws 66 mA, 50 Ω.
    i, _ = b.drive(np.array([3.3]), 0.0, 1.0)
    assert float(i[0]) == pytest.approx(0.066, rel=0.02)
    assert b.c_comp == pytest.approx(2e-12)
    assert 0.3e-9 < b.edge_time("rise") < 1.0e-9


def test_generic_driver_dc():
    b = ibis.buffer(ibis.generic_driver(3.3, 33.0, 1e-9))
    # Pull-down on at 1 V: 1 V / 33 Ω.
    i, _ = b.drive(np.array([1.0]), 0.0, 1.0)
    assert float(i[0]) == pytest.approx(1 / 33, rel=0.03)
    # The clamps hold the receiver near the rails: none at mid-supply, much just outside.
    rx = ibis.buffer(ibis.generic_receiver(3.3))
    assert abs(float(rx.static(np.array([1.65]))[0][0])) < 1e-6
    assert abs(float(rx.static(np.array([4.5]))[0][0])) > 1e-3


def _line_y(z0, er, length_mm, freqs):
    v = C0_MM_S / math.sqrt(er)
    l, c = z0 / v, 1 / (z0 * v)

    def rlgc(f):
        w = 2 * math.pi * np.asarray(f)
        return (1j * w * l)[:, None, None], (1j * w * c)[:, None, None]
    ckt = Circuit()
    ckt.add_line(["a"], ["b"], length_mm, rlgc)
    ckt.add_port("1", "a")
    ckt.add_port("2", "b")
    return ckt.solve_y(freqs), length_mm / v


def test_transient_matched_line():
    freqs = sparams.uniform_grid(20e9, 20e6)
    y, delay = _line_y(50.0, 4.0, 150.0, freqs)
    drv = ibis.buffer(ibis.generic_driver(2.0, 50.0, 100e-12))
    devs = [Device(drv, Stimulus([0, 1, 1, 1, 1, 1, 1, 1], 1e-9, 0.5e-9)), Device(r_term=50.0)]
    tr = simulate(freqs, y, devs, 6e-9, 5e-12)
    v_far = tr.v[:, 1]
    # Half the swing arrives one delay after the edge and stays (matched both ends).
    assert v_far[-1] == pytest.approx(1.0, abs=0.03)
    t_cross = tr.t[np.argmax(v_far > 0.5)]
    assert t_cross == pytest.approx(1.5e-9 + delay + 50e-12, abs=60e-12)
    assert v_far.max() < 1.08


def test_linear_transfer_halves_into_a_match():
    freqs = sparams.uniform_grid(10e9, 50e6)
    y, delay = _line_y(50.0, 4.0, 100.0, freqs)
    v = port_transfer(freqs, y, [LinearPort(drive=1.0, r_source=50.0), LinearPort(r_load=50.0)])
    assert np.allclose(np.abs(v[:, 1]), 0.5, atol=1e-3)
    phase = np.unwrap(np.angle(v[:, 1]))
    assert -phase[-1] / (2 * math.pi * freqs[-1]) == pytest.approx(delay, rel=1e-3)


def test_prbs_is_balanced_and_periodic():
    p = eyes.prbs(7)
    assert len(p) == 127 and p.sum() == 64
    assert np.array_equal(eyes.prbs(7, 254)[:127], eyes.prbs(7, 254)[127:])


def _ideal_pulse(spui=32, ui=1e-10):
    t = np.arange(40 * spui) * ui / spui
    v = np.zeros(len(t))
    v[4 * spui:5 * spui] = 1.0
    return eyes.Pulse(t, v, ui, spui)


def test_clean_eye_is_fully_open():
    e = eyes.statistical_eye(_ideal_pulse(), 1e-12, swing=0.8)
    assert e.height_v == pytest.approx(0.8, rel=0.01)
    assert e.width_ui > 0.95


def test_jitter_closes_the_eye_as_dual_dirac():
    rj, dj = 0.01, 0.05
    e = eyes.statistical_eye(_ideal_pulse(), 1e-12, rj, dj, swing=0.8)
    q = 7.03                                    # Q at BER 1e-12
    assert e.width_ui == pytest.approx(1 - dj - 2 * q * rj, abs=0.04)


def test_eye_before_the_pulse_record_is_drawn():
    """A pulse whose top starts at once (a slow line: short delay, long UI): the half UI
    before its record still has the previous bits, so the eye is drawn across ±1 UI."""
    spui = 32
    t = np.arange(40 * spui) * 1e-10 / spui
    v = np.zeros(len(t))
    v[:spui] = 1.0
    e = eyes.statistical_eye(eyes.Pulse(t, v, 1e-10, spui), 1e-12, swing=1.0)
    drawn = e.density.sum(axis=1) > 0
    assert drawn.all()


def test_slow_edges_get_samples_and_jitter_in_time():
    from kipi.si.protocols import PROTOCOLS
    from kipi.si.solver import EDGE_PHASES, EYE_PHASES, per_ui
    ui, rise = 10e-9, 0.3e-9                     # a 100 Mb/s clock, 0.5 ns edge
    n = per_ui(ui, rise, EDGE_PHASES, EYE_PHASES, 1024)
    assert n * (rise / 0.6) / ui >= EDGE_PHASES
    assert per_ui(125e-12, 40e-12, EDGE_PHASES, EYE_PHASES, 1024) == EYE_PHASES  # PCIe 3
    rj, dj = PROTOCOLS["cmos"].jitter_ui(100e6)
    assert dj == pytest.approx(50e-12 * 100e6) and rj == pytest.approx(5e-12 * 100e6)
    assert PROTOCOLS["pcie3"].jitter_ui() == (0.01, 0.05)


def test_ctle_peaks():
    f = np.array([0.0, 5e9])                    # DC and Nyquist of 10 Gb/s
    h = eyes.ctle(f, -6.0, 10e9)
    assert abs(h[0]) == pytest.approx(10 ** (-6 / 20))
    assert abs(h[1]) > 1.5 * abs(h[0])          # boosts Nyquist over DC
    assert abs(eyes.ctle(f, 0.0, 10e9)[1]) < 1.0    # no peaking at 0 dB


def test_isi_and_dfe():
    p = _ideal_pulse()
    p.v[5 * p.spui:6 * p.spui] = 0.3             # one post-cursor of 0.3
    raw = eyes.statistical_eye(p, 1e-12)
    # 0/1 signalling: the worst '1' is 1 (after a 0), the worst '0' is 0.3 (after a 1).
    assert raw.height_v == pytest.approx(0.7, abs=0.02)
    dfe = eyes.statistical_eye(p, 1e-12, eq=eyes.Equalizer(dfe_taps=1))
    assert dfe.height_v == pytest.approx(1.0, abs=0.02)
    com_raw, _, _ = eyes.com(p, 1e-12, 0, 0, 1e-3)
    com_dfe, _, _ = eyes.com(p, 1e-12, 0, 0, 1e-3, eq=eyes.Equalizer(dfe_taps=1))
    assert com_dfe > com_raw > 0


def test_eye_contours_and_bathtub():
    rj, dj = 0.01, 0.05
    e = eyes.statistical_eye(_ideal_pulse(), 1e-12, rj, dj, swing=0.8, phases_per_ui=128)
    assert e.contour_bers == [1e-3, 1e-6, 1e-9, 1e-12]
    widths = [np.isfinite(c[:, 0]).sum() for c in e.contours]
    assert widths == sorted(widths, reverse=True)       # lower BER, narrower eye
    assert e.center_ui == pytest.approx(0.0, abs=0.02)
    assert e.bathtub[np.argmin(abs(e.phases_ui))] < 1e-12
    assert e.bathtub[np.argmin(abs(e.phases_ui - 0.5))] > 0.1     # at the crossing
    # The voltage bathtub: open at the threshold, closing toward the rails.
    thr = np.argmin(abs(e.levels_v - e.threshold_v))
    assert e.vbathtub.shape == e.levels_v.shape
    assert e.vbathtub[thr] < 1e-12
    assert e.vbathtub[np.argmin(abs(e.levels_v - 0.8))] > 0.1     # at the 1 level
    assert e.vbathtub[np.argmin(abs(e.levels_v))] > 0.1           # at the 0 level


def test_waveform_eye_centred_and_drawn_as_lines():
    p = _ideal_pulse()
    bits = eyes.prbs(7, 254)
    up = np.zeros(len(bits) * p.spui)
    up[::p.spui] = bits
    # Edges a third of a UI long, so the eye has traces to draw.
    k = np.ones(p.spui // 3) / (p.spui // 3)
    wv = np.convolve(np.convolve(up, p.v[4 * p.spui:5 * p.spui + 1]), k)[:len(up)]
    t = np.arange(len(wv)) * p.ui_s / p.spui
    e = eyes.waveform_eye(t + 0.3 * p.ui_s, wv, p.ui_s)
    assert e.center_ui == 0.0 and e.height_v == pytest.approx(1.0, abs=0.02)
    assert e.width_ui == pytest.approx(1.0, abs=0.02)
    # The crossings sit at ±0.5 UI and every phase column of an edge has hits (a line).
    mid = np.argmin(abs(e.levels_v - 0.5))
    hits = e.phases_ui[e.density[:, mid] > 0]
    assert np.all(np.abs(np.abs(hits) - 0.5) < 0.05)
    edge = (np.abs(e.phases_ui - 0.5) < 0.12)
    assert (e.density[edge].sum(axis=1) > 0).all()
    rows = e.density[edge][:, (e.levels_v > 0.1) & (e.levels_v < 0.9)]
    assert (rows.sum(axis=1) > 0).mean() > 0.9


def test_eye_plot_draws_both_kinds():
    from matplotlib.figure import Figure
    from kipi.si.eyeplot import DARK, LIGHT, draw_eye
    e = eyes.statistical_eye(_ideal_pulse(), 1e-12, 0.01, 0.05, swing=0.8)
    fig = Figure()
    draw_eye(fig, e, (0.2, 0.4), DARK)
    assert len(fig.axes) == 4               # eye, colour bar, bathtub, voltage bathtub
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "BER 1e-12" in labels and "Receiver mask" in labels
    t = np.arange(4000) * 1e-10 / 32
    w = eyes.waveform_eye(t, (np.sin(2 * np.pi * t / 2e-10) > 0).astype(float), 1e-10)
    draw_eye(fig, w, None, LIGHT)
    assert len(fig.axes) == 2                           # no bathtub for a waveform
