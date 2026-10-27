"""Run geometry work on several cores. Shapely hands each operation to GEOS without the
GIL, so plain threads run unions, differences and triangulations side by side."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

WORKERS = max(1, min(8, (os.cpu_count() or 1)))

_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="kipi-geom")
        return _pool


def union_all(parts, chunk: int = 1000):
    """``shapely.union_all(parts)`` of polygons, several times quicker for thousands of
    small ones (silkscreen strokes).

    Only polygons that touch need joining: the parts fall into groups that touch only
    each other (a word, a logo), each group is joined on its own, on several threads,
    and the groups' polygons, which cannot overlap, are simply collected. A group of
    more than ``2 * chunk`` parts is joined in strips across the board, then together.
    """
    import numpy as np
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    parts = np.asarray(parts, dtype=object)
    if (len(parts) < 64 or WORKERS < 2
            or not all(shapely.get_type_id(parts) == shapely.GeometryType.POLYGON)):
        return shapely.union_all(parts)
    i, j = shapely.STRtree(parts).query(parts, predicate="intersects")
    n, label = connected_components(
        coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=(len(parts), len(parts))),
        directed=False)
    order = np.argsort(label, kind="stable")
    groups = np.split(parts[order], np.flatnonzero(np.diff(label[order])) + 1)
    # A part alone is kept as it is, mended if invalid (KiCad draws the odd glyph with
    # a ring that crosses itself), as joining it to others would have.
    alone = np.array([g[0] for g in groups if len(g) == 1], dtype=object)
    ok = shapely.is_valid(alone)
    out = list(alone[ok]) + list(shapely.make_valid(alone[~ok]))
    big = [g for g in groups if len(g) > 2 * chunk]
    small = [g for g in groups if 1 < len(g) <= 2 * chunk]
    batches = [small[k::WORKERS] for k in range(WORKERS)]
    for result in pmap(lambda gs: [shapely.union_all(g) for g in gs], batches):
        out.extend(result)
    for g in big:
        g = np.asarray(g, dtype=object)
        strips = np.array_split(g[np.argsort(shapely.get_x(shapely.centroid(g)))],
                                max(WORKERS, len(g) // chunk))
        out.append(shapely.union_all(np.asarray(pmap(shapely.union_all, strips),
                                                dtype=object)))
    polys = [p for g in out for p in shapely.get_parts(g)
             if p.geom_type == "Polygon" and not p.is_empty]
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def pmap(fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
    """``[fn(x) for x in items]``, on the pool's threads when there is more than one item.

    The calling thread works too: it runs the first item, then each item no pool thread
    has started yet, and waits only for those already running. So a call from inside a
    pool thread (a net's layers, then each layer's tiles) cannot deadlock, and uses any
    cores that are free.
    """
    items = list(items)
    if len(items) < 2 or WORKERS < 2:
        return [fn(x) for x in items]
    pool = _get_pool()
    futures = [pool.submit(fn, x) for x in items[1:]]
    try:
        out = [fn(items[0])]
        for x, f in zip(items[1:], futures):
            out.append(fn(x) if f.cancel() else f.result())
    except BaseException:
        for f in futures:
            f.cancel()
        raise
    return out
