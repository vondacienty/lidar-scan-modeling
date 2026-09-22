"""Canopy height model (CHM) of LiDAR scan points against a ground model."""

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


def _as_index(value) -> int:
    """Validate a grid index or count as a non-bool int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ground indices and counts must be non-bool int")
    return value


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    result = float(value.quantize(_QUANTUM))
    return 0.0 if result == 0.0 else result


def build_chm(points: Iterable[tuple | list], ground: Iterable[tuple | list],
              cell_size: int | float = 1.0) -> tuple:
    """Build a 2D CHM from points and a ground model, keeping the highest point.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list and each
    ground cell a 5-item ``(ix, iy, z, sigma, count)`` tuple/list. Coordinates
    divided by ``cell_size`` and floored toward negative infinity give the cell
    index ``(ix, iy)`` matched against the ground cells. Within a cell the point
    with the highest z wins; ties are broken by ascending sigma, intensity, x
    and y. The cell height is ``max(0, point_z - ground_z)`` and the cell sigma
    is ``sqrt(point_sigma**2 + ground_sigma**2)``.

    Returns a tuple of ``(ix, iy, height, sigma, count)`` tuples for cells
    containing at least one point, sorted lexicographically by ``(ix, iy)``;
    an empty points input returns ``()``.
    """
    if isinstance(cell_size, bool) or not isinstance(cell_size, _NUMERIC_TYPES):
        raise TypeError("cell_size must be a non-bool int or float")
    if isinstance(cell_size, float) and not math.isfinite(cell_size):
        raise ValueError("cell_size must be finite")

    try:
        ground_iter = iter(ground)
    except TypeError:
        raise TypeError("ground must be an iterable of ground cells") from None
    try:
        point_iter = iter(points)
    except TypeError:
        raise TypeError("points must be an iterable of points") from None

    cells: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    groups: dict[tuple[int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dcell = Decimal(str(cell_size))
        if dcell <= 0:
            raise ValueError("cell_size must be positive")

        for cell in ground_iter:
            if not isinstance(cell, (tuple, list)) or len(cell) != 5:
                raise TypeError("each ground cell must be a tuple or list of 5 items "
                                "(ix, iy, z, sigma, count)")
            ix = _as_index(cell[0])
            iy = _as_index(cell[1])
            dz = _as_decimal(cell[2])
            dsigma = _as_decimal(cell[3])
            if dsigma <= 0:
                raise ValueError("ground sigma must be positive")
            if _as_index(cell[4]) <= 0:
                raise ValueError("ground count must be positive")
            if (ix, iy) in cells:
                raise ValueError("duplicate ground cell index")
            cells[(ix, iy)] = (dz, dsigma)

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

            key = (int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR)),
                   int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR)))
            if key not in cells:
                raise ValueError("no ground cell for point cell")

            entry = groups.get(key)
            if entry is None:
                entry = [0, None]
                groups[key] = entry
            entry[0] += 1
            # rank by highest z, ties by ascending sigma, intensity, x, y
            candidate = (dz, dsigma, dintensity, dx, dy)
            best = entry[1]
            if best is None or dz > best[0] or (dz == best[0] and candidate[1:] < best[1:]):
                entry[1] = candidate

        result = []
        for (ix, iy), (count, candidate) in groups.items():
            ground_z, ground_sigma = cells[(ix, iy)]
            height = candidate[0] - ground_z
            if height < 0:
                height = Decimal(0)
            sigma = (candidate[1] * candidate[1]
                     + ground_sigma * ground_sigma).sqrt()
            result.append((
                ix, iy,
                _quantize(height), _quantize(sigma),
                count,
            ))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)
