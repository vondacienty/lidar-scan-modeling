"""Voxel-level fusion of LiDAR scan points."""

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


def fuse_voxels(points: Iterable[tuple | list], voxel_size: int | float = 1.0) -> tuple:
    """Fuse points sharing a voxel with inverse-variance weighting.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Coordinates
    divided by ``voxel_size`` and floored toward negative infinity give the voxel
    index. Within a voxel, fields are weighted by ``w = 1 / sigma**2`` and the
    fused sigma is ``sqrt(1 / sum(w))``.

    Returns a tuple of ``(ix, iy, iz, x, y, z, intensity, sigma, count)`` tuples
    sorted lexicographically by ``(ix, iy, iz)``; an empty input returns ``()``.
    """
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, _NUMERIC_TYPES):
        raise TypeError("voxel_size must be a non-bool int or float")
    if isinstance(voxel_size, float) and not math.isfinite(voxel_size):
        raise ValueError("voxel_size must be finite")

    try:
        point_iter = iter(points)
    except TypeError:
        raise TypeError("points must be an iterable of points") from None

    groups: dict[tuple[int, int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dvoxel = Decimal(str(voxel_size))
        if dvoxel <= 0:
            raise ValueError("voxel_size must be positive")

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

            ix = int((dx / dvoxel).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dvoxel).to_integral_value(rounding=ROUND_FLOOR))
            iz = int((dz / dvoxel).to_integral_value(rounding=ROUND_FLOOR))

            weight = Decimal(1) / (dsigma * dsigma)

            entry = groups.get((ix, iy, iz))
            if entry is None:
                entry = [0, [], [], [], [], []]
                groups[(ix, iy, iz)] = entry
            entry[0] += 1
            entry[1].append(weight)
            entry[2].append(weight * dx)
            entry[3].append(weight * dy)
            entry[4].append(weight * dz)
            entry[5].append(weight * dintensity)

        result = []
        for (ix, iy, iz), (count, weights, wx, wy, wz, wi) in groups.items():
            sum_w = _sorted_sum(weights)
            x = _sorted_sum(wx) / sum_w
            y = _sorted_sum(wy) / sum_w
            z = _sorted_sum(wz) / sum_w
            intensity = _sorted_sum(wi) / sum_w
            sigma = (Decimal(1) / sum_w).sqrt()
            result.append((
                ix, iy, iz,
                _quantize(x), _quantize(y), _quantize(z),
                _quantize(intensity), _quantize(sigma),
                count,
            ))

    result.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(result)
