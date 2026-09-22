"""Grid-cell canopy height model (CHM) of LiDAR points over a ground surface."""

from __future__ import annotations

import math
from collections.abc import Iterable
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_NUMERIC_TYPES = (int, float)


def _as_decimal(value: object, what: str) -> Decimal:
    """Validate a scalar parameter and convert it via ``Decimal(str(value))``."""
    if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
        raise TypeError(f"{what} must be non-bool int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{what} must be finite")
    return Decimal(str(value))


def _as_index(value: object, name: str) -> int:
    """Validate a non-bool integer index/count field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool int")
    return value


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    result = float(value.quantize(_QUANTUM))
    return 0.0 if result == 0.0 else result


def build_chm(points: Iterable[tuple | list],
              ground: Iterable[tuple | list],
              cell_size: int | float = 1.0) -> tuple:
    """Build a 2D canopy height model from points and a ground surface.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list; each
    ground cell is a 5-item ``(ix, iy, z, sigma, count)`` tuple/list. Point
    coordinates divided by ``cell_size`` and floored toward negative infinity
    give the cell index ``(ix, iy)`` which must match a ground cell.

    Within a cell the point with the highest z is kept; exact z ties are
    resolved by ascending ``(sigma, intensity, x, y)``. The output height is
    ``max(0, point_z - ground_z)`` and the output sigma is
    ``sqrt(point_sigma**2 + ground_sigma**2)``.

    Returns a tuple of ``(ix, iy, height, sigma, count)`` tuples, where
    ``count`` is the number of points in the cell, sorted lexicographically by
    ``(ix, iy)``; an empty ``points`` input returns ``()``.
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

    surface: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    groups: dict[tuple[int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dcell = Decimal(str(cell_size))
        if dcell <= 0:
            raise ValueError("cell_size must be positive")

        for cell in ground_iter:
            if not isinstance(cell, (tuple, list)) or len(cell) != 5:
                raise TypeError("each ground cell must be a tuple or list of 5 "
                                "items (ix, iy, z, sigma, count)")

            ix = _as_index(cell[0], "ground ix")
            iy = _as_index(cell[1], "ground iy")
            ground_z = _as_decimal(cell[2], "ground z")
            ground_sigma = _as_decimal(cell[3], "ground sigma")
            if ground_sigma <= 0:
                raise ValueError("ground sigma must be positive")
            ground_count = _as_index(cell[4], "ground count")
            if ground_count <= 0:
                raise ValueError("ground count must be positive")

            key = (ix, iy)
            if key in surface:
                raise ValueError("duplicate ground cell index")
            surface[key] = (ground_z, ground_sigma)

        for point in point_iter:
            if not isinstance(point, (tuple, list)) or len(point) != 5:
                raise TypeError("each point must be a tuple or list of 5 items "
                                "(x, y, z, intensity, sigma)")

            dx = _as_decimal(point[0], "point coordinates, intensity and sigma")
            dy = _as_decimal(point[1], "point coordinates, intensity and sigma")
            dz = _as_decimal(point[2], "point coordinates, intensity and sigma")
            dintensity = _as_decimal(point[3], "point coordinates, intensity and sigma")
            dsigma = _as_decimal(point[4], "point coordinates, intensity and sigma")
            if dsigma <= 0:
                raise ValueError("sigma must be positive")

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))
            key = (ix, iy)
            if key not in surface:
                raise ValueError("point cell has no matching ground cell")

            candidate = (dz, dsigma, dintensity, dx, dy)
            entry = groups.get(key)
            if entry is None:
                groups[key] = [1, candidate]
            else:
                entry[0] += 1
                best = entry[1]
                if candidate[0] > best[0] or (
                        candidate[0] == best[0] and candidate[1:] < best[1:]):
                    entry[1] = candidate

        result = []
        for (ix, iy), (count, best) in groups.items():
            point_z, point_sigma = best[0], best[1]
            ground_z, ground_sigma = surface[(ix, iy)]

            height = point_z - ground_z
            if height < 0:
                height = Decimal(0)
            sigma = (point_sigma * point_sigma
                     + ground_sigma * ground_sigma).sqrt()

            result.append((ix, iy, _quantize(height), _quantize(sigma), count))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)
