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


def build_tile_pyramid(points: Iterable[tuple | list],
                       cell_size: int | float = 1.0,
                       tile_cells: int = 256,
                       levels: int = 3) -> tuple:
    """Group points into tile pyramids across ``levels`` aggregation levels.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Cell
    indices are ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)``.
    At level ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``; level 0 matches :func:`build_tile_index`.

    Returns a ``levels``-tuple (level order) of tuples of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` tuples sorted
    lexicographically by ``(tx, ty)``, where ``(ix0, iy0)`` and ``(ix1, iy1)``
    are the inclusive cell-index bounds of the tile and
    ``zmin``/``zmax``/``count`` summarize its points; an empty input returns a
    ``levels``-tuple of empty tuples.
    """
    if isinstance(cell_size, bool) or not isinstance(cell_size, _NUMERIC_TYPES):
        raise TypeError("cell_size must be a non-bool int or float")
    if isinstance(cell_size, float) and not math.isfinite(cell_size):
        raise ValueError("cell_size must be finite")
    if isinstance(tile_cells, bool) or not isinstance(tile_cells, int):
        raise TypeError("tile_cells must be a non-bool int")
    if tile_cells <= 0:
        raise ValueError("tile_cells must be positive")
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise TypeError("levels must be a non-bool int")
    if levels <= 0:
        raise ValueError("levels must be positive")

    # Single pass over ``points``: iter() is called exactly once, len() is
    # never called on it, and every level is aggregated simultaneously.
    point_iter = iter(points)

    widths = [tile_cells * 2 ** level for level in range(levels)]
    tiles_per_level: list[dict[tuple[int, int], list]] = [
        {} for _ in range(levels)
    ]

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

            for level, width in enumerate(widths):
                key = (ix // width, iy // width)
                tiles = tiles_per_level[level]
                entry = tiles.get(key)
                if entry is None:
                    tiles[key] = [dz, dz, 1]
                else:
                    if dz < entry[0]:
                        entry[0] = dz
                    if dz > entry[1]:
                        entry[1] = dz
                    entry[2] += 1

        pyramid = []
        for width, tiles in zip(widths, tiles_per_level):
            level_tiles = []
            for (tx, ty), (zmin, zmax, count) in tiles.items():
                ix0 = tx * width
                iy0 = ty * width
                level_tiles.append((
                    tx, ty,
                    ix0, iy0,
                    ix0 + width - 1, iy0 + width - 1,
                    _quantize(zmin), _quantize(zmax),
                    count,
                ))
            level_tiles.sort(key=lambda item: (item[0], item[1]))
            pyramid.append(tuple(level_tiles))

    return tuple(pyramid)


def _validate_pyramid(pyramid) -> None:
    """Validate the structure of a tile pyramid, raising ``ValueError``."""
    for level_tiles in pyramid:
        if not isinstance(level_tiles, tuple):
            raise ValueError("each pyramid level must be a tuple")
        prev_key = None
        for tile in level_tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError("each tile must be a 9-tuple "
                                 "(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)")
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and count must be "
                                     "non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("zmin and zmax must be finite floats")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("each pyramid level must be sorted by (tx, ty) "
                                 "with no duplicate coordinates")
            prev_key = key


def query_tile_pyramid(pyramid: tuple, level: int, tx: int, ty: int) -> tuple | None:
    """Look up the tile ``(tx, ty)`` at ``level`` of a tile pyramid.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    lexicographically by ``(tx, ty)``. ``level``, ``tx`` and ``ty`` must be
    non-bool ints and ``level`` must satisfy ``0 <= level < len(pyramid)``.

    Returns the stored 9-tuple for the exact ``(tx, ty)`` match at that level,
    or ``None`` if no such tile exists. The input is never modified.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("tx", tx), ("ty", ty)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")

    _validate_pyramid(pyramid)

    for tile in pyramid[level]:
        if tile[0] == tx and tile[1] == ty:
            return tile
    return None


def query_tile_region(pyramid: tuple, level: int,
                      tx_min: int, ty_min: int,
                      tx_max: int, ty_max: int) -> tuple:
    """Select all tiles within a ``(tx, ty)`` rectangle at ``level``.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    lexicographically by ``(tx, ty)``. ``level`` and the four bounds must be
    non-bool ints; ``level`` must satisfy ``0 <= level < len(pyramid)`` and
    the bounds must satisfy ``tx_min <= tx_max`` and ``ty_min <= ty_max``.

    Returns a tuple of the stored 9-tuples at that level with
    ``tx_min <= tx <= tx_max`` and ``ty_min <= ty <= ty_max``, preserving the
    level's existing order; an empty region match returns ``()``. The input is
    never modified.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("tx_min", tx_min), ("ty_min", ty_min),
                        ("tx_max", tx_max), ("ty_max", ty_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if tx_min > tx_max or ty_min > ty_max:
        raise ValueError("region bounds must satisfy tx_min <= tx_max and "
                         "ty_min <= ty_max")

    _validate_pyramid(pyramid)

    return tuple(
        tile for tile in pyramid[level]
        if tx_min <= tile[0] <= tx_max and ty_min <= tile[1] <= ty_max
    )
