"""Frequency-dependent materials for signal integrity: dielectrics and copper.

**Dielectrics** follow the wideband Debye model of Djordjevic and Sarkar (2001):

    ε(ω) = ε∞ + Δε/(m2 − m1) · log10((ω2 + jω) / (ω1 + jω)),   ωk = 2π·10^mk

It is causal (Kramers–Kronig holds by construction) and its loss tangent is nearly flat
over the band, which is what laminate datasheets give. Δε and ε∞ are fitted from the εr
and tan δ quoted at one frequency, so a stackup's εr and tan δ are all a user enters. A
time-domain result from a non-causal model (constant εr and tan δ) shows a response
before the signal arrives; this one does not.

**Copper** is the DC resistance plus the skin effect's Rs·√f·(1+j), the HSPICE W
element's form: the (1+j) carries the internal inductance that keeps the skin effect
causal. Rough copper multiplies the √f part by a roughness factor K(f): Hammerstad's
(the RMS roughness only), Groisse's, or Huray's "snowball" model, the most accurate at
high frequency (spheres of radius r, with a sphere ratio SR, the spheres' surface over
the flat area's).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kipi.ac.solver.physics import MU0_H_PER_MM

RHO_CU_OHM_MM = 1.724e-5            # annealed copper at 20 °C
# The band the wideband Debye model's poles span, decades of rad/s.
DS_M1, DS_M2 = 4.0, 12.0

# Typical laminates, by a lower-case piece of their material name: (εr, tan δ) at 1 GHz.
LAMINATES = {
    "fr4": (4.3, 0.020), "fr-4": (4.3, 0.020), "370hr": (4.04, 0.021),
    "is410": (4.0, 0.020), "isola": (3.9, 0.018), "tu-768": (4.2, 0.016),
    "megtron 4": (3.8, 0.005), "megtron 6": (3.4, 0.002), "megtron6": (3.4, 0.002),
    "megtron": (3.6, 0.004), "i-speed": (3.6, 0.006), "tachyon": (3.0, 0.002),
    "ro4350": (3.48, 0.0037), "ro4003": (3.38, 0.0027), "rogers": (3.5, 0.003),
    "ptfe": (2.2, 0.0009), "polyimide": (3.5, 0.008), "kapton": (3.4, 0.007),
    "nelco": (3.7, 0.009), "n4000-13": (3.6, 0.009),
}
DEFAULT_ER = 4.3
DEFAULT_TAND = 0.02


def laminate(material: str):
    """(εr, tan δ) of a known laminate whose name contains ``material``'s key, else None."""
    m = (material or "").lower()
    for key in sorted(LAMINATES, key=len, reverse=True):
        if key in m:
            return LAMINATES[key]
    return None


def default_tand(material: str) -> float:
    lam = laminate(material)
    return lam[1] if lam else DEFAULT_TAND


@dataclass(frozen=True)
class Dielectric:
    """A wideband Debye dielectric fitted to ``er`` and ``tand`` at ``f_hz``."""
    er: float
    tand: float = DEFAULT_TAND
    f_hz: float = 1e9

    def _f(self, omega):
        w1, w2 = 2 * math.pi * 10 ** DS_M1, 2 * math.pi * 10 ** DS_M2
        return np.log10((w2 + 1j * omega) / (w1 + 1j * omega)) / (DS_M2 - DS_M1)

    def _fit(self):
        f0 = self._f(2 * math.pi * self.f_hz)
        d_eps = -self.er * self.tand / f0.imag if self.tand > 0 else 0.0
        return self.er - d_eps * f0.real, d_eps

    def eps(self, f_hz) -> np.ndarray:
        """Complex relative permittivity ε′ − jε″ at ``f_hz``."""
        f = np.maximum(np.asarray(f_hz, dtype=float), 1.0)
        einf, d_eps = self._fit()
        return einf + d_eps * self._f(2 * math.pi * f)


ROUGH_MODELS = ("smooth", "hammerstad", "groisse", "huray")


@dataclass(frozen=True)
class Roughness:
    """Copper surface roughness: ``model`` one of :data:`ROUGH_MODELS`. Hammerstad and
    Groisse take the RMS roughness ``rq_um``; Huray the sphere radius ``radius_um`` and
    sphere ratio ``sr`` (the spheres' surface per flat area, ~1–2.5 for typical foils:
    standard ED copper ~2.0, reverse-treated ~1.4, HVLP ~0.8)."""
    model: str = "smooth"
    rq_um: float = 1.0
    radius_um: float = 0.5
    sr: float = 1.5

    def factor(self, f_hz, rho_ohm_mm: float = RHO_CU_OHM_MM) -> np.ndarray:
        f = np.maximum(np.asarray(f_hz, dtype=float), 1.0)
        delta_um = 1e3 * np.sqrt(rho_ohm_mm / (math.pi * f * MU0_H_PER_MM))
        if self.model == "hammerstad":
            return 1 + 2 / math.pi * np.arctan(1.4 * (self.rq_um / delta_um) ** 2)
        if self.model == "groisse":
            return 1 + np.exp(-(delta_um / (2 * self.rq_um)) ** 1.6)
        if self.model == "huray":
            a = self.radius_um
            return 1 + 1.5 * self.sr / (1 + delta_um / a + delta_um ** 2 / (2 * a ** 2))
        return np.ones_like(f)


def surface_resistance(f_hz, rho_ohm_mm: float = RHO_CU_OHM_MM) -> np.ndarray:
    """Smooth copper's surface resistance √(πfμ0ρ), ohms per square."""
    return np.sqrt(math.pi * np.asarray(f_hz, dtype=float) * MU0_H_PER_MM * rho_ohm_mm)
