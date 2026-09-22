"""Grid-cell digital elevation model (DEM) of LiDAR scan points."""

from __future__ import annotations

import math
from collections.abc import Iterable
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_NUMERIC_TYPES = (int, float)


def _as_decimal(value) -> Decimal:
    """Validate a scalar parameter and convert it via ``Decimal(str(value))``."""
    if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
        raise TypeError("point coordinates, intensity and sigma must be non-bool int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("point coordinates, intensity and sigma must be finite")
    return Decimal(str(value))


def _sorted_sum(terms: list[Decimal]) -> Decimal:
    """Sum terms in ascending Decimal value order for order-independent results."""
    total = Decimal(0)
    for term in sorted(terms):
        total += term
    return total


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    result = float(value.quantize(_QUANTUM))
    return 0.0 if result == 0.0 else result


def build_dem(points: Iterable[tuple | list], cell_size: int | float = 1.0) -> tuple:
    """Build a 2D DEM from points with inverse-variance weighting.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Coordinates
    divided by ``cell_size`` and floored toward negative infinity give the cell
    index ``(ix, iy)``. Within a cell, z is weighted by ``w = 1 / sigma**2`` and
    the cell sigma is ``sqrt(1 / sum(w))``.

    Returns a tuple of ``(ix, iy, z, sigma, count)`` tuples for occupied cells
    only, sorted lexicographically by ``(ix, iy)``; an empty input returns ``()``.
    """
    if isinstance(cell_size, bool) or not isinstance(cell_size, _NUMERIC_TYPES):
        raise TypeError("cell_size must be a non-bool int or float")
    if isinstance(cell_size, float) and not math.isfinite(cell_size):
        raise ValueError("cell_size must be finite")

    try:
        point_iter = iter(points)
    except TypeError:
        raise TypeError("points must be an iterable of points") from None

    groups: dict[tuple[int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dcell = Decimal(str(cell_size))
        if dcell <= 0:
            raise ValueError("cell_size must be positive")

        for point in point_iter:
            if not isinstance(point, (tuple, list)) or len(point) != 5:
                raise TypeError("each point must be a tuple or list of 5 items "
                                "(x, y, z, intensity, sigma)")

            dx = _as_decimal(point[0])
            dy = _as_decimal(point[1])
            dz = _as_decimal(point[2])
            _as_decimal(point[3])  # intensity: validated but not used in the DEM
            dsigma = _as_decimal(point[4])
            if dsigma <= 0:
                raise ValueError("sigma must be positive")

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))

            weight = Decimal(1) / (dsigma * dsigma)

            entry = groups.get((ix, iy))
            if entry is None:
                entry = [0, [], []]
                groups[(ix, iy)] = entry
            entry[0] += 1
            entry[1].append(weight)
            entry[2].append(weight * dz)

        result = []
        for (ix, iy), (count, weights, wz) in groups.items():
            sum_w = _sorted_sum(weights)
            z = _sorted_sum(wz) / sum_w
            sigma = (Decimal(1) / sum_w).sqrt()
            result.append((
                ix, iy,
                _quantize(z), _quantize(sigma),
                count,
            ))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)


def build_dsm(points: Iterable[tuple | list], cell_size: int | float = 1.0) -> tuple:
    """Build a 2D digital surface model keeping the highest point per cell.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Coordinates
    divided by ``cell_size`` and floored toward negative infinity give the cell
    index ``(ix, iy)``. Within a cell the point with the highest z is kept;
    exact z ties are resolved by ascending ``(sigma, intensity, x, y)``.

    Returns a tuple of ``(ix, iy, z, sigma, count)`` tuples for occupied cells
    only, where ``count`` is the number of points in the cell, sorted
    lexicographically by ``(ix, iy)``; an empty input returns ``()``.
    """
    if isinstance(cell_size, bool) or not isinstance(cell_size, _NUMERIC_TYPES):
        raise TypeError("cell_size must be a non-bool int or float")
    if isinstance(cell_size, float) and not math.isfinite(cell_size):
        raise ValueError("cell_size must be finite")

    try:
        point_iter = iter(points)
    except TypeError:
        raise TypeError("points must be an iterable of points") from None

    groups: dict[tuple[int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dcell = Decimal(str(cell_size))
        if dcell <= 0:
            raise ValueError("cell_size must be positive")

        for point in point_iter:
            if not isinstance(point, (tuple, list)) or len(point) != 5:
                raise TypeError("each point must be a tuple or list of 5 items "
                                "(x, y, z, intensity, sigma)")

            dx = _as_decimal(point[0])
            dy = _as_decimal(point[1])
            dz = _as_decimal(point[2])
            dintensity = _as_decimal(point[3])
            dsigma = _as_decimal(point[4])
            if dsigma <= 0:
                raise ValueError("sigma must be positive")

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))

            candidate = (dz, dsigma, dintensity, dx, dy)
            entry = groups.get((ix, iy))
            if entry is None:
                groups[(ix, iy)] = [1, candidate]
            else:
                entry[0] += 1
                best = entry[1]
                if candidate[0] > best[0] or (
                        candidate[0] == best[0] and candidate[1:] < best[1:]):
                    entry[1] = candidate

        result = []
        for (ix, iy), (count, best) in groups.items():
            result.append((ix, iy, _quantize(best[0]), _quantize(best[1]), count))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)
