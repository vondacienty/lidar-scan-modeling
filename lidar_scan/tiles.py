"""Tile-level indexing of LiDAR scan points."""

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


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    result = float(value.quantize(_QUANTUM))
    return 0.0 if result == 0.0 else result


def build_tile_index(points: Iterable[tuple | list],
                     cell_size: int | float = 1.0,
                     tile_cells: int = 256) -> tuple:
    """Group points into tiles of ``tile_cells`` x ``tile_cells`` cells.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Cell
    indices are ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)`` and
    each tile covers ``tile_cells`` cells per axis, so ``tx = ix // tile_cells``
    and ``ty = iy // tile_cells``.

    Returns a tuple of ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)``
    tuples sorted lexicographically by ``(tx, ty)``, where ``(ix0, iy0)`` and
    ``(ix1, iy1)`` are the inclusive cell-index bounds of the tile and
    ``zmin``/``zmax``/``count`` summarize its points; an empty input returns
    ``()``.
    """
    if isinstance(cell_size, bool) or not isinstance(cell_size, _NUMERIC_TYPES):
        raise TypeError("cell_size must be a non-bool int or float")
    if isinstance(cell_size, float) and not math.isfinite(cell_size):
        raise ValueError("cell_size must be finite")
    if isinstance(tile_cells, bool) or not isinstance(tile_cells, int):
        raise TypeError("tile_cells must be a non-bool int")
    if tile_cells <= 0:
        raise ValueError("tile_cells must be positive")

    # Single pass over ``points``: iter() is called exactly once and any
    # exception raised here or during iteration propagates unchanged.
    point_iter = iter(points)

    tiles: dict[tuple[int, int], list] = {}

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
            _as_decimal(point[3])
            dsigma = _as_decimal(point[4])
            if dsigma <= 0:
                raise ValueError("sigma must be positive")

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))
            tx = ix // tile_cells
            ty = iy // tile_cells

            entry = tiles.get((tx, ty))
            if entry is None:
                entry = [dz, dz, 0]
                tiles[(tx, ty)] = entry
            else:
                if dz < entry[0]:
                    entry[0] = dz
                if dz > entry[1]:
                    entry[1] = dz
            entry[2] += 1

        result = []
        for (tx, ty), (zmin, zmax, count) in tiles.items():
            ix0 = tx * tile_cells
            iy0 = ty * tile_cells
            result.append((
                tx, ty,
                ix0, iy0,
                ix0 + tile_cells - 1, iy0 + tile_cells - 1,
                _quantize(zmin), _quantize(zmax),
                count,
            ))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)
