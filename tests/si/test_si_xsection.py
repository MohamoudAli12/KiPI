"""The multiconductor cross-section solver against closed forms: microstrip and stripline
impedance, Wheeler's incremental-inductance conductor loss, coupled-line symmetry, and the
wideband Debye dielectric's causality and fit."""

import math

import numpy as np
import pytest

from kipi.si import closed_form as cf
from kipi.si.materials import Dielectric, Roughness, surface_resistance
from kipi.si.xsection import Band, Cache, Conductor, solve

W = 8.0


def _microstrip(w, h, t, er, tand=0.0):
    cs = [Conductor.rect(-W, W, -0.035, 0.0), Conductor.rect(-w / 2, w / 2, h, h + t, "s")]
    return solve(cs, [Band(0, h, Dielectric(er, tand))], (-W, W, -0.5, W))


@pytest.mark.parametrize("w, h, t, er", [(0.3, 0.2, 0.035, 4.3), (0.15, 0.1, 0.035, 4.3),
                                         (0.1, 0.2, 0.018, 3.6), (1.0, 0.2, 0.035, 4.5)])
def test_microstrip_impedance_and_loss(w, h, t, er):
    r = _microstrip(w, h, t, er)
    z0, ee = cf.microstrip(w, h, er, t)
    assert r.z0() == pytest.approx(z0, rel=0.025)
    assert r.er_eff() == pytest.approx(ee, rel=0.02)
    rs = surface_resistance(1e9)
    d = 1e-5 * min(w, h, t)

    def za(ww, hh, tt):
        return cf.microstrip(ww, hh, 1.0, tt)[0]
    wheeler = rs * (za(w - 2 * d, h + 2 * d, t - 2 * d) - za(w, h, t)) / d / cf.ETA0
    assert rs * r.r_hf[0, 0] == pytest.approx(wheeler, rel=0.05)


@pytest.mark.parametrize("w, b", [(0.2, 0.4), (0.1, 0.3), (0.4, 0.5)])
def test_stripline_wheeler(w, b):
    er = 4.2
    cs = [Conductor.rect(-W, W, -0.035, 0.0), Conductor.rect(-W, W, b, b + 0.035),
          Conductor.rect(-w / 2, w / 2, b / 2 - 0.0005, b / 2 + 0.0005, "s")]
    r = solve(cs, [Band(0, b, Dielectric(er, 0.0))], (-W, W, -0.5, b + 0.5), fine_mm=0.001)
    assert r.z0() == pytest.approx(cf.stripline(w, b, er, 0.001), rel=0.01)
    assert cf.stripline(w, b, er, 0.001) < cf.stripline(w, b, er)   # Cohn's: zero thickness
    assert r.er_eff() == pytest.approx(er, rel=1e-6)


def test_coupled_pair_is_symmetric_and_bounded():
    h = 0.15
    cs = [Conductor.rect(-W, W, -0.035, 0.0), Conductor.rect(-0.25, -0.05, h, h + 0.035, "p"),
          Conductor.rect(0.05, 0.25, h, h + 0.035, "n")]
    r = solve(cs, [Band(0, h, Dielectric(4.0, 0.02))], (-W, W, -0.5, W))
    assert r.l[0, 1] == pytest.approx(r.l[1, 0])
    assert r.c[0, 1] < 0
    assert r.l[0, 0] == pytest.approx(r.l[1, 1], rel=0.01)
    p = r.pair()
    assert p["zodd"] < r.z0(0) < p["zeven"]
    assert p["td_odd"] < p["td_even"]                    # microstrip: odd mode is faster
    xt = r.crosstalk(0, 1)
    assert 0 < xt["kb"] < 0.25
    assert xt["kf_s_per_mm"] < 0                         # microstrip FEXT is negative


def test_stripline_has_no_fext():
    b = 0.4
    cs = [Conductor.rect(-W, W, -0.035, 0.0), Conductor.rect(-W, W, b, b + 0.035),
          Conductor.rect(-0.25, -0.05, 0.19, 0.21, "a"), Conductor.rect(0.05, 0.25, 0.19, 0.21, "b")]
    r = solve(cs, [Band(0, b, Dielectric(4.0, 0.02))], (-W, W, -0.5, b + 0.5))
    assert abs(r.crosstalk(0, 1)["kf_s_per_mm"]) < 1e-16


def test_dielectric_fit_and_causality():
    d = Dielectric(4.0, 0.02, 1e9)
    e = d.eps(1e9)
    assert e.real == pytest.approx(4.0, rel=1e-9)
    assert -e.imag / e.real == pytest.approx(0.02, rel=1e-9)
    f = np.logspace(6, 10.5, 50)
    eps = d.eps(f)
    assert np.all(np.diff(eps.real) < 0)                 # εr falls with frequency
    tand = -eps.imag / eps.real
    assert tand.min() > 0.015 and tand.max() < 0.025     # nearly flat over the band


def test_rlgc_limits():
    r = _microstrip(0.3, 0.2, 0.035, 4.3, 0.02)
    z, y = r.rlgc([1e3, 1e10], Roughness("huray", radius_um=0.5, sr=2.0))
    assert z[0, 0, 0].real == pytest.approx(r.r_dc[0], rel=1e-3)   # DC resistance
    rs = surface_resistance(1e10)
    assert z[1, 0, 0].real > 1.5 * rs * r.r_hf[0, 0]                # rough copper loses more
    g_over_wc = y[1, 0, 0].real / y[1, 0, 0].imag
    assert g_over_wc == pytest.approx(0.02, rel=0.25)


def test_roughness_factors():
    f = np.array([1e6, 1e9, 1e11])
    for m in ("hammerstad", "groisse", "huray"):
        k = Roughness(m, rq_um=1.0, radius_um=0.5, sr=2.0).factor(f)
        assert k[0] == pytest.approx(1.0, abs=0.05)
        assert np.all(np.diff(k) > 0)
    assert Roughness("hammerstad").factor(1e12) == pytest.approx(2.0, rel=0.01)


def test_cache_reuses_equal_cuts():
    cache = Cache()
    cs = [Conductor.rect(-W, W, -0.035, 0.0), Conductor.rect(-0.1, 0.1, 0.1, 0.135, "s")]
    bands = [Band(0, 0.1, Dielectric(4.3))]
    a = cache.solve(cs, bands, (-W, W, -0.5, W))
    b = cache.solve(cs, bands, (-W, W, -0.5, W))
    assert a is b and cache.hits == 1 and cache.misses == 1
    assert math.isfinite(a.z0())
