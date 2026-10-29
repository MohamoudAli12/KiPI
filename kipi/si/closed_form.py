"""Closed-form line impedances: the textbook formulas the field solver is checked against,
and a quick first guess for the impedance planner.

* Microstrip: Hammerstad and Jensen (1980), with their thickness correction.
* Stripline (centred, zero thickness): Cohn's exact conformal map,
  Z0 = 30π/√εr · K(k)/K(k′), k = sech(πw/2b); with thickness, Wheeler's (1978) formula.
* Conductor loss of a microstrip: Wheeler's incremental inductance rule on
  Hammerstad–Jensen's Z0, the usual reference for a field solver's R(f).
"""

from __future__ import annotations

import math

from scipy.special import ellipk

ETA0 = 376.730313668


def microstrip(w: float, h: float, er: float, t: float = 0.0):
    """(Z0 ohms, εr_eff) of a microstrip of width w over a plane h below, thickness t."""
    u = w / h
    if t > 0:
        du1 = t / (math.pi * h) * math.log(1 + 4 * math.e / (t / h * (1 / math.tanh(
            math.sqrt(6.517 * u))) ** 2))
        dur = 0.5 * (1 + 1 / math.cosh(math.sqrt(er - 1))) * du1
        u1, ur = u + du1, u + dur
    else:
        u1 = ur = u

    def z_air(x):
        f = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / x) ** 0.7528)
        return ETA0 / (2 * math.pi) * math.log(f / x + math.sqrt(1 + 4 / x ** 2))

    def e_eff(x):
        a = (1 + math.log((x ** 4 + (x / 52) ** 2) / (x ** 4 + 0.432)) / 49
             + math.log(1 + (x / 18.1) ** 3) / 18.7)
        b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        return (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / x) ** (-a * b)

    ee = e_eff(ur) * (z_air(u1) / z_air(ur)) ** 2
    return z_air(ur) / math.sqrt(e_eff(ur)), ee


def stripline(w: float, b: float, er: float, t: float = 0.0) -> float:
    """Z0 of a stripline of width w centred between planes b apart (thickness t)."""
    if t <= 0:
        k = 1 / math.cosh(math.pi * w / (2 * b))
        kp = math.sqrt(1 - k * k)
        return 30 * math.pi / math.sqrt(er) * ellipk(k ** 2) / ellipk(kp ** 2)
    x = t / b
    m = 6 / (3 + 2 * x / (1 - x))
    dw = x / (math.pi * (1 - x)) * (1 - 0.5 * math.log((x / (2 - x)) ** 2
                                                       + (0.0796 * x / (w / b + 1.1 * x)) ** m))
    wp = w / (b - t) + dw
    return 30 / math.sqrt(er) * math.log(1 + 4 / (math.pi * wp) * (
        8 / (math.pi * wp) + math.sqrt((8 / (math.pi * wp)) ** 2 + 6.27)))


def microstrip_rc_per_mm(w: float, h: float, er: float, rs_ohm: float) -> float:
    """A microstrip's skin-effect resistance per mm by Wheeler's rule,
    R = Rs/(μ0)·∂L/∂n with L from the air-filled Z0: R ≈ Rs·(Z0a(w−δ, h+δ) − Z0a)/(η0·δ)·…
    Evaluated by a small numeric step (δ → 0)."""
    d = 1e-5 * min(w, h)

    def z_air(ww, hh):
        return microstrip(ww, hh, 1.0)[0]
    dz = z_air(w - 2 * d, h + 2 * d) - z_air(w, h)   # every wall recedes by d
    # dL/dn = dZ0a/dn / c, and R = Rs·dL/dn / μ0 = Rs·dZ0a/dn / η0.
    return rs_ohm * dz / d / ETA0
