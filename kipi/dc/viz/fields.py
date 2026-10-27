"""The result fields a DC solve shows, and how many decimals their color scale needs.

Apart from :mod:`kipi.dc.viz.results` so the setup panels read them without loading
pyvista at KiPI's start.
"""

from __future__ import annotations

import math

import numpy as np

# name -> (array, association, label, unit). Association "via" means the field lives
# on the via barrels rather than the copper layers.
FIELDS = {
    "voltage": ("V", "point", "Voltage", "V"),
    "current_density": ("J", "cell", "Current density", "A/mm²"),
    "power_density": ("P", "cell", "Power density", "mW/mm²"),
    "temperature": ("T", "point", "Temperature", "°C"),
    "via_current": ("I_mA", "via", "Via current", "mA"),
}


def scale_decimals(lo: float, hi: float) -> int:
    """Decimals that tell apart six values spread over ``lo``..``hi``: a 3.3 V rail that
    sags by 0.4 mV needs five, or every one reads "3.3"."""
    span = abs(hi - lo)
    if not span > 0 or not math.isfinite(span):
        return 4
    return min(9, max(0, int(np.ceil(-np.log10(span / 5))) + 1))
