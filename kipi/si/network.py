"""A channel as a frequency-domain circuit: multiconductor line sections, vias, lumped
parts and N-port blocks, solved for its port S-parameters.

Every element stamps an admittance block between its nodes (ground is node -1):

* **Coupled lines** (:meth:`Circuit.add_line`): an N-conductor section of per-length
  Z(ω) = R + jωL and Y(ω) = G + jωC. Its 2N-port admittance comes from the modes of
  Z·Y = T·Γ²·T⁻¹ (Paul, *Analysis of Multiconductor Transmission Lines*): with
  Yc = Z⁻¹·T·Γ·T⁻¹,

      Y11 = Y22 = Yc·T·coth(Γl)·T⁻¹,   Y12 = Y21 = −Yc·T·csch(Γl)·T⁻¹,

  computed for every frequency at once, coth and csch in their overflow-free forms. The
  section is exact at any length, so a segment only has to be short enough for the
  geometry (its cross-section) to be constant along it.
* **Lumped parts**: R, L, C, any two-terminal Z(ω), any N-port Y(ω) (a via model, a
  Touchstone file of a connector or package).

The node equations are solved at each frequency with a sparse LU, and the internal
nodes eliminated (Schur complement) to the port admittance Yp, then
S = (I − √Z0·Yp·√Z0)(I + √Z0·Yp·√Z0)⁻¹.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

F_MIN_HZ = 1e3                  # the "DC" point: lines are solved no lower than this
LEAK_S = 1e-12                  # every node's leak to ground, so none floats


def line_blocks(z: np.ndarray, y: np.ndarray, length_mm: float):
    """(Y11, Y12) of an N-conductor line section, (F, N, N) each, from its per-mm series
    impedance ``z`` and shunt admittance ``y`` (F, N, N)."""
    n = z.shape[-1]
    if n == 1:
        zs, ys = z[:, 0, 0], y[:, 0, 0]
        g = np.sqrt(zs * ys)
        g = np.where(g.real < 0, -g, g)
        x = g * length_mm
        e2 = np.exp(-2 * x)
        coth = (1 + e2) / (1 - e2)
        csch = 2 * np.exp(-x) / (1 - e2)
        yc = g / zs
        return (yc * coth)[:, None, None], (-yc * csch)[:, None, None]
    lam, t = np.linalg.eig(z @ y)
    g = np.sqrt(lam)
    g = np.where(g.real < 0, -g, g)
    x = g * length_mm
    e2 = np.exp(-2 * x)
    coth = (1 + e2) / (1 - e2)
    csch = 2 * np.exp(-x) / (1 - e2)
    ti = np.linalg.inv(t)
    yc_t = np.linalg.solve(z, t * g[:, None, :])          # Z⁻¹·T·Γ
    y11 = (yc_t * coth[:, None, :]) @ ti
    y12 = -(yc_t * csch[:, None, :]) @ ti
    return y11, y12


@dataclass
class _Block:
    nodes: List[int]
    y: Callable[[np.ndarray], np.ndarray]     # freqs -> (F, k, k)
    label: str = ""


@dataclass
class Port:
    name: str
    node: int
    z0: float = 50.0


@dataclass
class Circuit:
    nodes: Dict[Hashable, int] = field(default_factory=dict)
    blocks: List[_Block] = field(default_factory=list)
    ports: List[Port] = field(default_factory=list)

    def node(self, key: Hashable) -> int:
        """The node called ``key`` (made on first use); ``None`` is ground."""
        if key is None:
            return -1
        if key not in self.nodes:
            self.nodes[key] = len(self.nodes)
        return self.nodes[key]

    # ----------------------------------------------------------------- elements
    def add_block(self, keys: Sequence[Hashable], y: Callable[[np.ndarray], np.ndarray],
                  label: str = ""):
        """An N-port admittance ``y(freqs)`` (F, N, N) between ``keys`` and ground."""
        self.blocks.append(_Block([self.node(k) for k in keys], y, label))

    def add_line(self, a: Sequence[Hashable], b: Sequence[Hashable], length_mm: float,
                 rlgc: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], label: str = ""):
        """An N-conductor section from nodes ``a`` to ``b`` (conductor k from a[k] to
        b[k]); ``rlgc(freqs)`` gives its per-mm Z and Y, (F, N, N) each."""
        n = len(a)

        def y(f):
            z, yy = rlgc(f)
            y11, y12 = line_blocks(z, yy, max(length_mm, 1e-6))
            out = np.zeros((len(f), 2 * n, 2 * n), complex)
            out[:, :n, :n] = out[:, n:, n:] = y11
            out[:, :n, n:] = out[:, n:, :n] = y12
            return out
        self.add_block([*a, *b], y, label)

    def add_z(self, n1: Hashable, n2: Hashable, z: Callable[[np.ndarray], np.ndarray],
              label: str = ""):
        """A two-terminal impedance ``z(freqs)`` (F,) between two nodes."""
        def y(f):
            yy = 1.0 / np.asarray(z(f), complex)
            return np.stack([np.stack([yy, -yy], -1), np.stack([-yy, yy], -1)], -2)
        self.add_block([n1, n2], y, label)

    def add_r(self, n1, n2, r_ohm: float, label: str = ""):
        self.add_z(n1, n2, lambda f: np.full(len(f), max(r_ohm, 1e-9), complex), label)

    def add_l(self, n1, n2, l_h: float, r_ohm: float = 0.0, label: str = ""):
        self.add_z(n1, n2, lambda f: r_ohm + 1e-9 + 2j * math.pi * np.maximum(f, F_MIN_HZ) * l_h,
                   label)

    def add_c(self, n1, n2, c_f: float, label: str = ""):
        def y(f):
            yy = 2j * math.pi * np.maximum(f, F_MIN_HZ) * c_f
            return np.stack([np.stack([yy, -yy], -1), np.stack([-yy, yy], -1)], -2)
        self.add_block([n1, n2], y, label)

    def add_shunt(self, n: Hashable, y: Callable[[np.ndarray], np.ndarray], label: str = ""):
        """An admittance ``y(freqs)`` (F,) from a node to ground."""
        self.add_block([n], lambda f: np.asarray(y(f), complex)[:, None, None], label)

    def add_sparams(self, keys: Sequence[Hashable], freqs_hz: np.ndarray, s: np.ndarray,
                    z0: float = 50.0, label: str = ""):
        """An N-port from S-parameters (each port to ground), interpolated on frequency
        (magnitude and unwrapped phase); held at its end values outside its band."""
        from kipi.si.sparams import interpolate, y_from_s
        ys = y_from_s(s, z0)
        self.add_block(keys, lambda f: interpolate(freqs_hz, ys, f), label)

    def add_port(self, name: str, key: Hashable, z0: float = 50.0) -> Port:
        p = Port(name, self.node(key), z0)
        self.ports.append(p)
        return p

    # -------------------------------------------------------------------- solve
    def solve_y(self, freqs_hz: Sequence[float], check_cancel=None,
                on_progress=None) -> np.ndarray:
        """The port admittance matrix (F, P, P); ``on_progress(done, total)`` is called
        as the frequencies are solved."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        f = np.maximum(np.asarray(freqs_hz, dtype=float), F_MIN_HZ)
        n = len(self.nodes)
        rows, cols, data = [], [], []
        for b in self.blocks:
            yb = b.y(f)                                       # (F, k, k)
            idx = np.asarray(b.nodes)
            keep = np.flatnonzero(idx >= 0)
            if not len(keep):
                continue
            ii, jj = np.meshgrid(keep, keep, indexing="ij")
            rows.append(idx[ii].ravel())
            cols.append(idx[jj].ravel())
            data.append(yb[:, ii, jj].reshape(len(f), -1))
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        data.append(np.full((len(f), n), LEAK_S, complex))
        r, c = np.concatenate(rows), np.concatenate(cols)
        d = np.concatenate(data, axis=1)
        pn = np.array([p.node for p in self.ports])
        inner = np.setdiff1d(np.arange(n), pn)
        out = np.zeros((len(f), len(pn), len(pn)), complex)
        for k in range(len(f)):
            if check_cancel and k % 64 == 0:
                check_cancel()
            m = coo_matrix((d[k], (r, c)), shape=(n, n)).tocsc()
            ypp = m[pn][:, pn].toarray()
            if len(inner):
                mii = m[inner][:, inner].tocsc()
                yip = m[inner][:, pn].toarray()
                ypi = m[pn][:, inner].toarray()
                ypp = ypp - ypi @ splu(mii).solve(yip)
            out[k] = ypp
            if on_progress and (k % 16 == 15 or k == len(f) - 1):
                on_progress(k + 1, len(f))
        return out

    def solve_s(self, freqs_hz, check_cancel=None) -> np.ndarray:
        from kipi.si.sparams import s_from_y
        y = self.solve_y(freqs_hz, check_cancel)
        return s_from_y(y, np.array([p.z0 for p in self.ports]))

    def solve_nodes(self, freqs_hz, injections: Dict[Hashable, np.ndarray],
                    loads: Optional[Dict[Hashable, float]] = None,
                    observe: Sequence[Hashable] = ()) -> np.ndarray:
        """Node voltages at ``observe`` (F, k) for currents ``injections`` (key -> (F,))
        into nodes, with ``loads`` (key -> resistance to ground) added: the circuit driven
        directly, for transfer functions."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        f = np.maximum(np.asarray(freqs_hz, dtype=float), F_MIN_HZ)
        extra = Circuit(dict(self.nodes), list(self.blocks))
        for key, r in (loads or {}).items():
            extra.add_shunt(key, lambda ff, r=r: np.full(len(ff), 1.0 / r, complex))
        n = len(extra.nodes)
        out = np.zeros((len(f), len(observe)), complex)
        blocks = [(np.asarray(b.nodes), b.y(f)) for b in extra.blocks]
        obs = [extra.nodes[k] for k in observe]
        for k in range(len(f)):
            rows, cols, data = [np.arange(n)], [np.arange(n)], [np.full(n, LEAK_S, complex)]
            for idx, yb in blocks:
                keep = np.flatnonzero(idx >= 0)
                ii, jj = np.meshgrid(keep, keep, indexing="ij")
                rows.append(idx[ii].ravel())
                cols.append(idx[jj].ravel())
                data.append(yb[k][ii, jj].ravel())
            m = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n)).tocsc()
            rhs = np.zeros(n, complex)
            for key, cur in injections.items():
                rhs[extra.nodes[key]] += cur[k]
            out[k] = splu(m).solve(rhs)[obs]
        return out
