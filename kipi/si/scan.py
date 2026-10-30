"""Quick Scan: every signal net's impedance, coupling and return path, with no models.

Each net's traces are cut every ``segment_mm``; each cut is solved
(:mod:`kipi.si.xsection`) with the net's pair partner and its nearest neighbouring
signals as conductors of their own, the rest of the copper as the reference. That gives,
along every trace:

* **Z0** (each conductor's with the others quiet) and, for a pair, **Zdiff**, against the
  target (± tolerance);
* **crosstalk**: the backward (NEXT) coefficient Kb = ¼(Lm/L + Cm/C) to each neighbour,
  and the length run beside it;
* **reference**: the nearest plane above and below covering the trace, so where it has
  none, where it runs over a gap or a plane's edge, and where a via changes its
  reference plane: with stitching vias of the plane's net near, or not, or onto another
  net's plane (then only a capacitor carries the return).

and per net: its length and delay, its vias' stubs (a stub's quarter-wave resonance
c/(4·l·√εr) below ``f_max`` is flagged), and a pair's skew. Each net then passes, warns
or fails each check.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kipi.core.model.board import BoardSnapshot
from kipi.core.progress import NULL, ProgressReporter
from kipi.si.board import SIBoard, normal_of
from kipi.si.extract import SNAP_MM, _Nodes, net_segments
from kipi.si.pairs import diff_target, manual_of, pair_map

C_MM_S = 299792458e3
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class ScanSettings:
    segment_mm: float = 2.0
    window_mm: float = 1.5
    neighbours: int = 3                 # signal neighbours solved as conductors per cut
    z_target_ohm: float = 50.0
    z_tol_pct: float = 10.0
    diff_target_ohm: Optional[float] = None   # None: by interface name (pairs.diff_target)
    kb_limit: float = 0.03              # NEXT coefficient: warn above, fail above twice it
    coupled_min_mm: float = 5.0         # a neighbour counts once run beside this far
    f_max_hz: float = 10e9              # stubs resonating below this are flagged
    skew_limit_mm: float = 0.15
    stitch_radius_mm: float = 2.0       # a return via this near a layer change is enough
    targets: Dict[str, float] = field(default_factory=dict)   # net -> Z0 (or Zdiff) target
    include_slow: bool = False          # also nets named as slow controls (SLOW_NET)
    manual: tuple = ((), ())            # the user's (pairs, single-ended nets)
    # Speed: a scan's cuts are screened, not signed off. A coarser mesh (×mesh_scale) is
    # within about 2 % of the channel solver's; ground copper's edges are compared to
    # ground_grid_mm (cuts that differ by less share one solve); workers: processes
    # solving cuts at once (0: one per CPU core but one).
    mesh_scale: float = 2.0
    ground_grid_mm: float = 0.05
    workers: int = 0
    # Each cut keeps only what bears on the trace: the copper between its nearest solid
    # planes above and below (they shield what lies beyond), out to screen_h of its
    # height over the nearer one each side of it (Z0 and coupling fall off over a few).
    screen_h: float = 5.0


@dataclass
class ScanSegment:
    net: str
    layer: int
    a: Tuple[float, float]
    b: Tuple[float, float]
    width_mm: float
    z0: float
    zdiff: Optional[float] = None
    kb: Dict[str, float] = field(default_factory=dict)
    ref_above: Optional[str] = None             # "In1.Cu GND"
    ref_below: Optional[str] = None
    gap: bool = False
    delay_s: float = 0.0

    @property
    def length_mm(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    @property
    def referenced(self) -> bool:
        return bool(self.ref_above or self.ref_below)


@dataclass
class Finding:
    check: str                  # "impedance", "crosstalk", "reference", "stub", "skew"
    status: str                 # PASS / WARN / FAIL
    text: str
    where: Optional[Tuple[float, float]] = None


@dataclass
class NetScan:
    net: str
    partner: str = ""
    length_mm: float = 0.0
    delay_s: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    z_mean: float = 0.0
    zdiff_min: Optional[float] = None
    zdiff_max: Optional[float] = None
    zdiff_mean: Optional[float] = None
    target: float = 0.0
    worst_kb: float = 0.0
    worst_aggressor: str = ""
    coupled_mm: Dict[str, float] = field(default_factory=dict)
    unreferenced_mm: float = 0.0
    gap_mm: float = 0.0
    vias: int = 0
    max_stub_mm: float = 0.0
    stub_f_hz: Optional[float] = None
    skew_mm: Optional[float] = None
    skew_s: Optional[float] = None
    findings: List[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        st = [f.status for f in self.findings]
        return FAIL if FAIL in st else WARN if WARN in st else PASS


@dataclass
class ScanResult:
    nets: List[NetScan]
    segments: List[ScanSegment]
    settings: ScanSettings
    solves: int = 0
    cached: int = 0
    seconds: float = 0.0
    board_name: str = ""
    skipped_slow: List[str] = field(default_factory=list)   # slow control nets left out

    def by_net(self) -> Dict[str, NetScan]:
        return {n.net: n for n in self.nets}


def signal_nets(snap: BoardSnapshot) -> List[str]:
    """Nets with tracks that are not power or ground."""
    from kipi.core.model.nets import classify_nets
    kinds = classify_nets(snap)
    with_tracks = {t.net for t in snap.tracks}
    return sorted(n for n, k in kinds.items() if k == "signal" and n in with_tracks)


# Nets named as slow controls: enables, resets, power-good, LEDs, interrupts, I2C, JTAG,
# straps. Their edges are slow or rare, so impedance and crosstalk do not matter to them;
# the scan leaves them out unless asked (or ticked). Matched on the name's last part.
SLOW_NET = re.compile(
    r"(^|[_\-~!.])(N?RST|N?RESET|RESET_?N|N?EN|ENABLE|EN[0-9]+|PWR_?EN|PG|PGOOD|POWER_?GOOD|"
    r"LED[A-Z0-9]*|BOOT[0-9]*|N?FAULT|N?ALERT|N?IRQ|N?INT[0-9]*|N?SHDN|STBY|SLEEP|WAKE|"
    r"MODE[0-9]*|SEL[0-9]*|SDA|SCL|GPIO[0-9]*|BTN[0-9]*|BUTTON[0-9]*|TP[0-9]+|TMS|TCK|TDI|"
    r"TDO|TRST|N?PROG[A-Z_]*|DONE|N?INIT[A-Z_]*|CFG[A-Z0-9_]*|N?OE|N?WP|N?PD|STRAP[0-9]*)"
    r"([_\-0-9]|$)", re.IGNORECASE)


def is_slow(net: str) -> bool:
    """Whether ``net`` is named as a slow control signal (:data:`SLOW_NET`)."""
    return bool(SLOW_NET.search(net.rsplit("/", 1)[-1]))


@dataclass
class _Cut:
    net: str
    seg: object
    length_mm: float
    conds: list
    box: tuple
    found: List[str]
    pieces: list
    partner: str = ""
    key: tuple = ()


def scan(snap: BoardSnapshot, nets: Optional[Sequence[str]] = None,
         settings: Optional[ScanSettings] = None, progress: ProgressReporter = NULL,
         board: Optional[SIBoard] = None) -> ScanResult:
    """Scan ``nets`` (every signal net if None: but the slow control nets, unless the
    settings include them). The cuts are gathered first, cuts alike (to the cache's
    rounding) solved once, and the solves run in parallel."""
    t0 = time.time()
    s = settings or ScanSettings()
    with progress.timed("Indexing the board's copper"):
        board = board or SIBoard(snap, check_cancel=progress.check_cancel)
        board._tile_index()
    all_signals = signal_nets(snap)
    partners = pair_map(all_signals, s.manual)
    skipped = []
    if nets is None:
        nets = all_signals
        if not s.include_slow:
            skipped = [n for n in nets if is_slow(n) and n not in partners]
            nets = [n for n in nets if n not in set(skipped)]
    nets = [n for n in nets if n in set(all_signals)]
    signal_set = set(all_signals)
    cache = _scan_cache(board, s)
    m0 = cache.misses
    progress.info(f"Quick Scan: {len(nets)} signal net(s), cut every {s.segment_mm:g} mm"
                  + (f"; {len(skipped)} slow control net(s) left out" if skipped else ""))
    # 1. Every cut's geometry.
    cuts: Dict[str, List[_Cut]] = {}
    kept = board.__dict__.setdefault("_scan_cuts", {})
    how = (s.segment_mm, s.window_mm, s.neighbours, s.screen_h)
    with progress.timed("Cutting the traces"):
        for k, net in enumerate(nets):
            progress.check_cancel()
            progress.fraction(0.15 * k / max(len(nets), 1))
            if (net, how) not in kept:          # a rescan reuses the cuts
                kept[(net, how)] = _cuts(board, net, partners.get(net, ""), signal_set, s)
            cuts[net] = kept[(net, how)]
    # 2. The distinct cuts not solved before, solved (in parallel when many).
    todo: Dict[tuple, _Cut] = {}
    every = [c for lst in cuts.values() for c in lst]
    for c in every:
        c.key = cache.key(c.conds, board.bands, c.box)
        if c.key not in cache._d and c.key not in todo:
            todo[c.key] = c
    progress.info(f"{len(every)} cuts, {len(todo)} to solve "
                  f"({len(every) - len(todo)} alike or solved before)")
    with progress.timed(f"Solving {len(todo)} cross-sections"):
        _solve_all(todo, board, cache, s, progress)
    # 3. Each net's segments and summary.
    out_nets, out_segs = [], []
    for net in nets:
        segs = [_segment(board, c, cache) for c in cuts[net]]
        out_segs += segs
        out_nets.append(_summarise(board, net, partners.get(net, ""), segs, s))
    _pair_skew(out_nets, s)
    progress.fraction(1.0)
    res = ScanResult(out_nets, out_segs, s, cache.misses - m0,
                     len(every) - (cache.misses - m0), time.time() - t0, snap.board_name,
                     skipped)
    progress.info(f"Quick Scan done: {len(nets)} nets, {len(out_segs)} cuts "
                  f"({res.solves} solved, {res.cached} reused) in {res.seconds:.1f} s")
    return res


def _scan_cache(board: SIBoard, s: ScanSettings):
    """The board's cache of scan cuts (kept for the next scan), for these settings."""
    from kipi.si.xsection import Cache
    key = (s.mesh_scale, s.ground_grid_mm)
    caches = board.__dict__.setdefault("_scan_caches", {})
    if key not in caches:
        caches[key] = Cache(ground_grid_mm=s.ground_grid_mm, mesh_scale=s.mesh_scale,
                            size=1_000_000)
    return caches[key]


def _cuts(board: SIBoard, net: str, partner: str, signals: set, s: ScanSettings
          ) -> List[_Cut]:
    out = []
    segs = []
    nodes = _Nodes(board.snap, [net])
    for seg in net_segments(board.snap, net, s.segment_mm):
        mid, nrm, ln = normal_of(seg.a, seg.b)
        if ln >= 1e-6:
            segs.append((seg, mid, nrm, ln))
    # A piece under the net's own pad or via pad is that pad's copper, not a line: a jog
    # into a via, say, would be judged over the via's antipad.
    own = [x for x in segs if nodes.key(net, x[0].layer, x[1])[0] == "pt"]
    segs = own or segs
    if not segs:
        return out
    every = board.pieces_many([x[1] for x in segs], [x[2] for x in segs], s.window_mm)
    for (seg, mid, nrm, ln), pieces in zip(segs, every):
        named: Dict[str, Tuple[int, float]] = {}
        if partner:
            pp = [p for p in pieces if p.net == partner]
            if pp:
                best = min(pp, key=lambda p: (p.layer != seg.layer, abs(0.5 * (p.x0 + p.x1))))
                named[partner] = (best.layer, 0.5 * (best.x0 + best.x1))
        neigh = {}
        for p in pieces:
            if p.net in signals and p.net not in (net, partner):
                d = min(abs(p.x0), abs(p.x1))
                if p.net not in neigh or d < neigh[p.net][0]:
                    neigh[p.net] = (d, p.layer, 0.5 * (p.x0 + p.x1))
        for n, (_, lay, x) in sorted(neigh.items(), key=lambda kv: kv[1][0])[:s.neighbours]:
            named[n] = (lay, x)
        conds, box, found = board.section(mid, nrm, seg.layer, seg.width_mm, net, s.window_mm,
                                          named, pieces)
        conds, box, found = screened(conds, box, found, s.screen_h, partner)
        out.append(_Cut(net, seg, ln, conds, box, found, pieces, partner))
    return out


def _extent(c):
    xs = [x for x, _ in c.points]
    zs = [z for _, z in c.points]
    return min(xs), max(xs), min(zs), max(zs)


def screened(conds, box, found: List[str], screen_h: float, partner: str = ""):
    """A cut (the trace first, as :meth:`SIBoard.section` gives it) cut down to what
    bears on the trace: the copper between its nearest planes above and below that span
    the whole cut, inside ±(w/2 + screen_h·h), h the trace's height over the nearer of
    them (screen_h ≤ 0: the cut as it is). (conductors, box, the named nets kept)."""
    from kipi.si.xsection import GROUND, Conductor
    if screen_h <= 0:
        return conds, box, found
    sx0, sx1, sz0, sz1 = _extent(conds[0])
    ext = [_extent(c) for c in conds]
    full = [e for c, e in zip(conds, ext) if c.name == GROUND
            and e[0] <= box[0] + 1e-6 and e[1] >= box[1] - 1e-6]
    above = min((e[2] for e in full if e[2] >= sz1 - 1e-9), default=None)
    below = max((e[3] for e in full if e[3] <= sz0 + 1e-9), default=None)
    hs = [d for d in ((above - sz1) if above is not None else None,
                      (sz0 - below) if below is not None else None) if d is not None]
    if not hs:
        return conds, box, found
    reach = screen_h * max(min(hs), 0.02)
    half = (sx1 - sx0) / 2 + reach
    if partner in found:                # a pair: the partner and its surroundings too
        px = [abs(x) for c in conds if c.name == partner for x, _ in c.points]
        half = max(half, max(px) + reach)
    half = min(half, box[1], -box[0])
    out = []
    for c, (x0, x1, z0, z1) in zip(conds, ext):
        if (above is not None and z0 > above + 1e-9) or (below is not None and z1 < below - 1e-9):
            continue                                # behind a plane
        a, b = max(x0, -half), min(x1, half)
        if b - a <= 1e-6:
            continue
        if (a, b) == (x0, x1):
            out.append(c)
        else:                                       # cut at the window's edge
            out.append(Conductor(tuple((min(max(x, -half), half), z) for x, z in c.points),
                                 c.name, c.rho_ohm_mm))
    kept = {c.name for c in out}
    return out, (-half, half, box[2], box[3]), [n for n in found if n in kept]


def _solve_one(conds, bands, box, mesh_scale):
    from kipi.si.xsection import solve
    try:
        return solve(conds, bands, box, mesh_scale=mesh_scale)
    except Exception as e:                  # a cut the mesher cannot do: reported, not fatal
        return e


PARALLEL_MIN = 24              # fewer solves than this run here: a pool takes ~1 s to start


def _workers(s: ScanSettings) -> int:
    import os
    n = s.workers or max((os.cpu_count() or 1) - 1, 1)
    return max(min(n, 32), 1)


def _solve_all(todo: Dict[tuple, _Cut], board: SIBoard, cache, s: ScanSettings,
               progress: ProgressReporter, on_progress=None):
    """Solve the cuts of ``todo`` (key -> anything with ``conds`` and ``box``) into
    ``cache``, on a pool of processes when there are enough of them. ``s`` gives
    ``workers`` and ``mesh_scale``; ``on_progress(done, total)`` replaces the scan's own
    progress share."""
    items = list(todo.items())
    n = len(items)
    if not n:
        return
    done = 0

    def finish(key, res):
        nonlocal done
        done += 1
        if on_progress:
            on_progress(done, n)
        else:
            progress.fraction(0.15 + 0.8 * done / n)
        if isinstance(res, Exception):
            from kipi.si.xsection import solve
            res = solve(todo[key].conds, board.bands, todo[key].box)   # the fine mesh
        cache.misses += 1
        cache.put(key, res)

    workers = _workers(s)
    if workers > 1 and n >= PARALLEL_MIN:
        try:
            _solve_parallel(items, board, s, workers, progress, finish)
            return
        except (OSError, RuntimeError, ImportError) as e:   # no processes here: serially
            progress.info(f"Solving in one process ({e})")
            items = [(k, c) for k, c in items if k not in cache._d]
    for key, c in items:
        progress.check_cancel()
        finish(key, _solve_one(c.conds, board.bands, c.box, s.mesh_scale))


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _one_thread():
    """A worker's maths on one thread: the workers are the parallelism, and a thread pool
    in each of them would only fight the others for the cores."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _solve_parallel(items, board: SIBoard, s: ScanSettings, workers: int,
                    progress: ProgressReporter, finish):
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    # spawn, not fork: the app has threads (Qt's), which a fork would copy mid-lock.
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                               initializer=_one_thread)
    import os
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})     # read as the workers start
    try:
        pending = {}
        it = iter(items)
        for key, c in it:                   # a few per worker in flight, not all at once
            pending[pool.submit(_solve_one, c.conds, board.bands, c.box, s.mesh_scale)] = key
            if len(pending) >= 4 * workers:
                break
        while pending:
            progress.check_cancel()
            ready, _ = wait(list(pending), timeout=0.25, return_when=FIRST_COMPLETED)
            for f in ready:
                finish(pending.pop(f), f.result())
                nxt = next(it, None)
                if nxt is not None:
                    pending[pool.submit(_solve_one, nxt[1].conds, board.bands, nxt[1].box,
                                        s.mesh_scale)] = nxt[0]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _segment(board: SIBoard, c: _Cut, cache) -> ScanSegment:
    from kipi.si.xsection import _renamed
    lp = _renamed(cache._d[c.key], c.conds)
    net, seg = c.net, c.seg
    i = lp.names.index(net)
    sc = ScanSegment(net, seg.layer, seg.a, seg.b, seg.width_mm, lp.z0(i),
                     delay_s=lp.delay_s_per_mm(i) * c.length_mm)
    if c.partner and c.partner in c.found:
        sc.zdiff = lp.pair(i, lp.names.index(c.partner))["zdiff"]
    for n in c.found:
        if n != c.partner:
            sc.kb[n] = float(lp.crosstalk(i, lp.names.index(n))["kb"])
    above, below, partial = board.references(c.pieces, seg.layer, seg.width_mm)
    name = {cu.id: cu.name for cu in board.stack.copper}
    sc.ref_above = f"{name[above.layer]} {above.net}" if above else None
    sc.ref_below = f"{name[below.layer]} {below.net}" if below else None
    sc.gap = partial and not (above and below)
    return sc


def _target(net: str, partner: str, s: ScanSettings) -> float:
    if net in s.targets:
        return s.targets[net]
    if partner:
        return s.diff_target_ohm or diff_target(net)
    return s.z_target_ohm


def _summarise(board: SIBoard, net: str, partner: str, segs: List[ScanSegment],
               s: ScanSettings) -> NetScan:
    ns = NetScan(net, partner)
    if not segs:
        ns.findings.append(Finding("impedance", WARN, "No traces to scan"))
        return ns
    ln = np.array([x.length_mm for x in segs])
    z = np.array([x.z0 for x in segs])
    ns.length_mm = float(ln.sum())
    ns.delay_s = float(sum(x.delay_s for x in segs))
    ns.z_min, ns.z_max = float(z.min()), float(z.max())
    ns.z_mean = float((z * ln).sum() / ln.sum())
    ns.target = _target(net, partner, s)
    tol = s.z_tol_pct / 100
    zd = [(x.zdiff, x.length_mm, x) for x in segs if x.zdiff]
    if partner and zd:
        v = np.array([a for a, _, _ in zd])
        w = np.array([b for _, b, _ in zd])
        ns.zdiff_min, ns.zdiff_max = float(v.min()), float(v.max())
        ns.zdiff_mean = float((v * w).sum() / w.sum())
        uncoupled = ns.length_mm - float(w.sum())
        _impedance_finding(ns, [(a, b, x) for a, b, x in zd], ns.target, tol, "Zdiff")
        if uncoupled > 0.25 * ns.length_mm and uncoupled > 2.0:
            ns.findings.append(Finding("impedance", WARN,
                                       f"{uncoupled:.1f} mm of {ns.length_mm:.1f} mm runs "
                                       f"apart from {partner}"))
    else:
        _impedance_finding(ns, [(x.z0, x.length_mm, x) for x in segs], ns.target, tol, "Z0")
    # Crosstalk
    for x in segs:
        for n, kb in x.kb.items():
            if kb > 0.01:
                ns.coupled_mm[n] = ns.coupled_mm.get(n, 0.0) + x.length_mm
            if kb > ns.worst_kb and ns.coupled_mm.get(n, 0.0) >= 0:
                ns.worst_kb, ns.worst_aggressor = kb, n
    long_enough = {n for n, mm in ns.coupled_mm.items() if mm >= s.coupled_min_mm}
    worst = max(((max(x.kb.get(n, 0) for x in segs), n) for n in long_enough), default=None)
    if worst and worst[0] > s.kb_limit:
        st = FAIL if worst[0] > 2 * s.kb_limit else WARN
        where = next(x for x in segs if x.kb.get(worst[1], 0) == worst[0])
        ns.findings.append(Finding("crosstalk", st,
                                   f"NEXT {100 * worst[0]:.1f} % from {worst[1]} over "
                                   f"{ns.coupled_mm[worst[1]]:.1f} mm "
                                   f"(limit {100 * s.kb_limit:.1f} %)", _mid(where)))
    else:
        ns.findings.append(Finding("crosstalk", PASS,
                                   f"Worst NEXT {100 * ns.worst_kb:.1f} %"
                                   + (f" from {ns.worst_aggressor}" if ns.worst_aggressor else "")))
    # Reference
    unref = [x for x in segs if not x.referenced]
    gaps = [x for x in segs if x.gap]
    ns.unreferenced_mm = float(sum(x.length_mm for x in unref))
    ns.gap_mm = float(sum(x.length_mm for x in gaps))
    if unref:
        long_unref = ns.unreferenced_mm > max(3 * s.segment_mm, 0.1 * ns.length_mm)
        ns.findings.append(Finding("reference", FAIL if long_unref else WARN,
                                   f"{ns.unreferenced_mm:.1f} mm with no reference plane",
                                   _mid(unref[0])))
    if gaps:
        ns.findings.append(Finding("reference", WARN,
                                   f"{ns.gap_mm:.1f} mm over a plane gap or edge", _mid(gaps[0])))
    _vias(board, ns, segs, s)
    if not any(f.check == "reference" for f in ns.findings):
        ns.findings.append(Finding("reference", PASS, "Referenced along its length"))
    return ns


def _impedance_finding(ns: NetScan, vals, target: float, tol: float, what: str):
    bad = [(v, l, x) for v, l, x in vals if abs(v - target) > tol * target]
    off = sum(l for _, l, _ in bad)
    if not bad:
        ns.findings.append(Finding("impedance", PASS,
                                   f"{what} {min(v for v, _, _ in vals):.1f}–"
                                   f"{max(v for v, _, _ in vals):.1f} Ω, target {target:g} Ω"))
        return
    worst = max(bad, key=lambda b: abs(b[0] - target))
    st = FAIL if off > 0.1 * ns.length_mm or abs(worst[0] - target) > 2 * tol * target else WARN
    ns.findings.append(Finding("impedance", st,
                               f"{what} off target ({target:g} Ω ± {100 * tol:g} %) over "
                               f"{off:.1f} mm, worst {worst[0]:.1f} Ω", _mid(worst[2])))


def _mid(x: ScanSegment):
    return (0.5 * (x.a[0] + x.b[0]), 0.5 * (x.a[1] + x.b[1]))


def _vias(board: SIBoard, ns: NetScan, segs: List[ScanSegment], s: ScanSettings):
    snap = board.snap
    tracks, vias, pads, _ = snap.items_on_net(ns.net)
    ns.vias = len(vias)
    stack = board.stack
    er = max((b.material.er for b in board.bands), default=4.2)
    for v in vias:
        r = max(v.diameter_mm, v.drill_mm) / 2 + SNAP_MM
        used = set()
        for t in tracks:
            for p in (t.start, t.end):
                if math.hypot(p[0] - v.position[0], p[1] - v.position[1]) <= r:
                    used.add(t.layer)
        for p in pads:
            if math.hypot(p.position[0] - v.position[0], p.position[1] - v.position[1]) <= r:
                used.update(p.shapes)
        span = stack.layers_between(v.layers[0], v.layers[-1])
        if not used or not span:
            continue
        idx = [span.index(l) for l in used if l in span]
        if not idx:
            continue
        lo, hi = min(idx), max(idx)
        top, bottom = stack.layer(span[0]), stack.layer(span[-1])
        stub = max(stack.layer(span[lo]).z_bottom_mm - top.z_bottom_mm if lo > 0 else 0.0,
                   stack.layer(span[hi]).z_bottom_mm - bottom.z_bottom_mm if hi < len(span) - 1
                   else 0.0)
        stub = abs(stub)
        if stub > ns.max_stub_mm:
            ns.max_stub_mm = stub
            ns.stub_f_hz = C_MM_S / (4 * stub * math.sqrt(er)) if stub > 0 else None
        _reference_change(board, ns, v, used, segs, s)
    if ns.max_stub_mm > 0 and ns.stub_f_hz and ns.stub_f_hz < s.f_max_hz:
        ns.findings.append(Finding("stub", WARN if ns.stub_f_hz > 0.5 * s.f_max_hz else FAIL,
                                   f"Via stub {ns.max_stub_mm:.2f} mm resonates near "
                                   f"{ns.stub_f_hz / 1e9:.1f} GHz (below {s.f_max_hz / 1e9:g} GHz)"))
    elif ns.vias:
        ns.findings.append(Finding("stub", PASS, f"{ns.vias} via(s), longest stub "
                                                 f"{ns.max_stub_mm:.2f} mm"))


def _reference_change(board: SIBoard, ns: NetScan, via, used, segs, s: ScanSettings):
    """A via between layers with different reference planes: is the return stitched?"""
    near = {}
    for x in segs:
        for p in (x.a, x.b):
            if math.hypot(p[0] - via.position[0], p[1] - via.position[1]) <= via.diameter_mm:
                near.setdefault(x.layer, x)
    refs = {lay: {r for r in (x.ref_above, x.ref_below) if r} for lay, x in near.items()}
    if len(refs) < 2:
        return
    sets = list(refs.values())
    if not all(sets):
        return                                          # counted as unreferenced already
    common = set.intersection(*sets)
    if common:
        return                                          # a plane both layers see
    nets_a = {r.split(" ", 1)[1] for r in sets[0]}
    nets_b = {r.split(" ", 1)[1] for r in sets[1]}
    shared = nets_a & nets_b
    if not shared:
        ns.findings.append(Finding("reference", WARN,
                                   f"Via at ({via.position[0]:.2f}, {via.position[1]:.2f}) "
                                   f"moves the return from {'/'.join(sorted(nets_a))} to "
                                   f"{'/'.join(sorted(nets_b))}: only capacitors carry it",
                                   via.position))
        return
    net = sorted(shared)[0]
    _, gvias, _, _ = board.snap.items_on_net(net)
    d = min((math.hypot(g.position[0] - via.position[0], g.position[1] - via.position[1])
             for g in gvias if len(g.layers) > 1), default=math.inf)
    if d > s.stitch_radius_mm:
        ns.findings.append(Finding("reference", WARN,
                                   f"Via at ({via.position[0]:.2f}, {via.position[1]:.2f}) "
                                   f"changes reference plane with no {net} stitching via "
                                   f"within {s.stitch_radius_mm:g} mm"
                                   + (f" (nearest {d:.1f} mm)" if math.isfinite(d) else ""),
                                   via.position))


def _pair_skew(nets: List[NetScan], s: ScanSettings):
    by = {n.net: n for n in nets}
    for n in nets:
        p = by.get(n.partner)
        if p is None or n.net > n.partner:
            continue
        skew = abs(n.length_mm - p.length_mm)
        skew_s = abs(n.delay_s - p.delay_s)
        for x in (n, p):
            x.skew_mm, x.skew_s = skew, skew_s
            st = PASS if skew <= s.skew_limit_mm else WARN if skew <= 3 * s.skew_limit_mm else FAIL
            x.findings.append(Finding("skew", st, f"Pair skew {skew:.3f} mm "
                                                  f"({skew_s * 1e12:.2f} ps), limit "
                                                  f"{s.skew_limit_mm:g} mm"))


def settings_of(si) -> ScanSettings:
    """The Quick Scan's settings from the setup's :class:`kipi.si.model.SISettings`."""
    return ScanSettings(segment_mm=si.scan_segment_mm, window_mm=si.window_mm,
                        z_target_ohm=si.z_target_ohm, z_tol_pct=si.z_tol_pct,
                        diff_target_ohm=si.diff_target_ohm, kb_limit=si.kb_limit,
                        f_max_hz=si.scan_f_max_hz, skew_limit_mm=si.skew_limit_mm,
                        stitch_radius_mm=si.stitch_radius_mm,
                        include_slow=getattr(si, "scan_slow", False),
                        manual=manual_of(si))
