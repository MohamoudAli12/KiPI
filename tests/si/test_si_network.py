"""The channel circuit: transmission-line sections against the telegrapher's closed forms,
coupled sections cascade exactly, mixed-mode conversion, TDR and Touchstone output."""

import math

import numpy as np
import pytest

from kipi.core.model import touchstone
from kipi.si import sparams
from kipi.si.network import Circuit, line_blocks

C0_MM_S = 299792458e3


def _lossless(z0, er):
    v = C0_MM_S / math.sqrt(er)
    l, c = z0 / v, 1 / (z0 * v)

    def rlgc(f):
        w = 2 * math.pi * np.asarray(f)
        return (1j * w * l)[:, None, None], (1j * w * c)[:, None, None]
    return rlgc, v


def test_matched_line():
    rlgc, v = _lossless(50.0, 4.0)
    ckt = Circuit()
    ckt.add_line(["a"], ["b"], 100.0, rlgc)
    ckt.add_port("1", "a")
    ckt.add_port("2", "b")
    f = np.array([1e8, 1e9, 5e9])
    s = ckt.solve_s(f)
    assert np.abs(s[:, 0, 0]).max() < 1e-6
    expected = np.exp(-2j * math.pi * f * 100.0 / v)
    assert np.allclose(s[:, 1, 0], expected, atol=1e-6)


def test_mismatched_line_reflection():
    rlgc, v = _lossless(75.0, 4.0)
    ckt = Circuit()
    ckt.add_line(["a"], ["b"], 30.0, rlgc)
    ckt.add_port("1", "a")
    ckt.add_port("2", "b")
    f = np.array([2e9])
    s = ckt.solve_s(f)[0]
    bl = 2 * math.pi * f[0] * 30.0 / v
    zl = 50.0
    zin = 75.0 * (zl + 1j * 75.0 * math.tan(bl)) / (75.0 + 1j * zl * math.tan(bl))
    assert s[0, 0] == pytest.approx((zin - 50) / (zin + 50), abs=1e-9)


def test_coupled_sections_cascade_exactly():
    l = np.array([[3.0e-10, 0.8e-10], [0.8e-10, 3.1e-10]])
    c = np.array([[1.3e-13, -0.3e-13], [-0.3e-13, 1.2e-13]])
    r = np.array([[0.02, 0.003], [0.003, 0.02]])

    def rlgc(f):
        w = 2 * math.pi * np.asarray(f)[:, None, None]
        return r + 1j * w * l, 1j * w * c * (1 - 0.02j)
    f = np.array([1e8, 3e9, 1e10])
    one, two = Circuit(), Circuit()
    one.add_line(["a1", "a2"], ["b1", "b2"], 80.0, rlgc)
    two.add_line(["a1", "a2"], ["m1", "m2"], 30.0, rlgc)
    two.add_line(["m1", "m2"], ["b1", "b2"], 50.0, rlgc)
    for ck in (one, two):
        for k in ("a1", "a2", "b1", "b2"):
            ck.add_port(k, k)
    s1, s2 = one.solve_s(f), two.solve_s(f)
    assert np.allclose(s1, s2, atol=1e-9)
    assert sparams.reciprocity(s1) < 1e-9
    assert sparams.passivity(s1) <= 1.0


def test_long_lossy_line_is_stable():
    z = np.full((1, 1, 1), 50 + 1j)
    y = np.full((1, 1, 1), 0.02 + 1j)
    y11, y12 = line_blocks(z, y, 1e4)        # γl ~ 1e4: no overflow
    assert np.isfinite(y11).all() and np.isfinite(y12).all()


def test_mixed_mode_of_uncoupled_pair():
    rlgc, _ = _lossless(50.0, 4.0)
    ckt = Circuit()
    ckt.add_line(["p1"], ["p2"], 50.0, rlgc)
    ckt.add_line(["n1"], ["n2"], 50.0, rlgc)
    for k in ("p1", "n1", "p2", "n2"):
        ckt.add_port(k, k)
    s = ckt.solve_s(np.array([1e9]))
    mm = sparams.mixed_mode(s, [(0, 1), (2, 3)])
    assert abs(mm[0, 1, 0]) == pytest.approx(1.0, abs=1e-6)        # Sdd21
    assert abs(mm[0, 0, 2 + 0]) < 1e-9                              # no mode conversion


def test_tdr_sees_a_75_ohm_section():
    r50, v = _lossless(50.0, 4.0)
    r75, _ = _lossless(75.0, 4.0)
    ckt = Circuit()
    ckt.add_line(["a"], ["m"], 50.0, r50)
    ckt.add_line(["m"], ["n"], 50.0, r75)
    ckt.add_line(["n"], ["b"], 50.0, r50)
    ckt.add_port("1", "a")
    ckt.add_port("2", "b")
    f = sparams.uniform_grid(30e9, 20e6)
    s = ckt.solve_s(f)
    t, z = sparams.tdr(f, s[:, 0, 0], 50.0, rise_s=30e-12)
    td = 50.0 / v
    mid = (t > 2 * td + 0.2 * 2 * td) & (t < 2 * td + 0.8 * 2 * td)
    assert np.median(z[mid]) == pytest.approx(75.0, rel=0.02)
    before = t < 1.5 * td
    assert np.median(z[before]) == pytest.approx(50.0, rel=0.01)


def test_tdr_sees_a_mismatch_right_at_the_port():
    """An 86 Ω line straight off a 100 Ω port: its reflection starts at t = 0, half the
    edge before it, and the TDR still reads 86 Ω (it read ~92 Ω, half the step lost)."""
    f = sparams.uniform_grid(60e9, 20e6)
    zl, z0, td = 86.0, 100.0, 400e-12
    tn = np.tan(2 * math.pi * f * td)
    zin = zl * (z0 + 1j * zl * tn) / (zl + 1j * z0 * tn)
    for rise in (15e-12, 40e-12):
        t, z = sparams.tdr(f, (zin - z0) / (zin + z0), z0, rise)
        line = (t > 0.25 * 2 * td) & (t < 0.75 * 2 * td)
        assert np.median(z[line]) == pytest.approx(zl, rel=0.005)


def test_touchstone_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    for p in (2, 4):
        s = (rng.normal(size=(5, p, p)) + 1j * rng.normal(size=(5, p, p))) * 0.1
        f = np.linspace(1e8, 5e8, 5)
        path = sparams.write_touchstone(tmp_path / "x", f, s, port_names=[f"P{i}" for i in range(p)])
        net = touchstone.read(path)
        assert np.allclose(net.s, s, atol=1e-8)
        assert np.allclose(net.freqs_hz, f)
