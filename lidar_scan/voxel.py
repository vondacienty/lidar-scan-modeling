"""Voxel level fusion of LiDAR points."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from typing import Iterable

_PRECISION = 50
_QUANTUM = Decimal("0.000001")


def _as_decimal(value, name: str) -> Decimal:
    """Validate a scalar argument and convert it via ``Decimal(str(value))``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float (bool is not allowed)")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Decimal(str(value))


def _sorted_sum(terms: list[Decimal]) -> Decimal:
    """Sum terms in ascending Decimal value order for reproducible results."""
    total = Decimal(0)
    for term in sorted(terms):
        total += term
    return total


def _to_float(value: Decimal) -> float:
    """Quantize to six decimal places, convert to float, normalize -0.0."""
    result = float(value.quantize(_QUANTUM))
    if result == 0.0:
        result = 0.0
    return result


def fuse_voxels(points: Iterable[tuple | list], voxel_size: int | float = 1.0) -> tuple:
    """Fuse points belonging to the same voxel with inverse-variance weights.

    Each point is a 5-item tuple/list ``(x, y, z, intensity, sigma)``. Voxel
    indices are the coordinates divided by ``voxel_size`` and floored toward
    negative infinity. Inside a voxel, weighted means of x/y/z/intensity are
    computed with ``w = 1 / sigma**2`` and the fused sigma is
    ``sqrt(1 / sum(w))``.

    All arithmetic runs through :class:`decimal.Decimal` (precision 50,
    ROUND_HALF_EVEN) so input ordering cannot affect the result. Returns a
    tuple of ``(ix, iy, iz, x, y, z, intensity, sigma, count)`` tuples sorted
    lexicographically by voxel index; an empty input yields ``()``.
    """
    size = _as_decimal(voxel_size, "voxel_size")
    if size <= 0:
        raise ValueError("voxel_size must be greater than zero")

    try:
        iterator = iter(points)
    except TypeError as exc:
        raise TypeError("points must be an iterable of (x, y, z, intensity, sigma)") from exc

    groups: dict[tuple[int, int, int], list[tuple[Decimal, ...]]] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for point in iterator:
            if not isinstance(point, (tuple, list)) or len(point) != 5:
                raise TypeError("each point must be a tuple or list of 5 items")

            x, y, z, intensity, sigma = (_as_decimal(v, f"point field {i}") for i, v in enumerate(point))
            if sigma <= 0:
                raise ValueError("sigma must be greater than zero")

            ix = int((x / size).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((y / size).to_integral_value(rounding=ROUND_FLOOR))
            iz = int((z / size).to_integral_value(rounding=ROUND_FLOOR))

            weight = Decimal(1) / (sigma * sigma)
            groups.setdefault((ix, iy, iz), []).append((weight, x, y, z, intensity))

        fused = []
        for (ix, iy, iz), members in groups.items():
            weights = [member[0] for member in members]
            sum_w = _sorted_sum(weights)

            means = []
            for dim in range(1, 5):
                terms = [member[0] * member[dim] for member in members]
                means.append(_sorted_sum(terms) / sum_w)

            sigma_out = (Decimal(1) / sum_w).sqrt()

            fused.append(
                (
                    ix,
                    iy,
                    iz,
                    _to_float(means[0]),
                    _to_float(means[1]),
                    _to_float(means[2]),
                    _to_float(means[3]),
                    _to_float(sigma_out),
                    len(members),
                )
            )

    fused.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(fused)
