"""Tile-level indexing of LiDAR scan points."""

from __future__ import annotations

import json
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


def _sorted_sum(terms: list) -> Decimal:
    """Add Decimal terms in ascending numeric order.

    Accumulating in sorted order makes the result independent of the order in
    which the points arrived.
    """
    total = Decimal(0)
    for term in sorted(terms):
        total += term
    return total


def build_tile_pyramid_stats(points: Iterable[tuple | list],
                             cell_size: int | float = 1.0,
                             tile_cells: int = 256,
                             levels: int = 3) -> tuple:
    """Group points into tile pyramids with inverse-variance z statistics.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Cell
    indices are ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)``.
    At level ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``.

    Per-tile statistics use weights ``w = 1 / sigma ** 2``:
    ``zmean = sum(w * z) / sum(w)`` and ``zsigma = sqrt(1 / sum(w))``; the
    summations are Decimal sums (precision 50, ``ROUND_HALF_EVEN``) of the
    terms sorted in ascending order, so the results do not depend on input
    order.

    Returns a ``levels``-tuple (level order) of tuples of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    tuples sorted lexicographically by ``(tx, ty)``, where ``(ix0, iy0)`` and
    ``(ix1, iy1)`` are the inclusive cell-index bounds of the tile,
    ``zmin``/``zmax``/``zmean``/``zsigma`` are quantized to six decimal places
    as floats (negative zero normalized) and ``count`` summarizes the tile's
    points; an empty input returns a ``levels``-tuple of empty tuples.
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
    # Each entry is [zmin, zmax, count, weights, weighted z values]; the
    # per-point Decimal summands are kept and sorted before accumulation.
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

            weight = Decimal(1) / (dsigma * dsigma)
            weighted_z = weight * dz

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))

            for level, width in enumerate(widths):
                key = (ix // width, iy // width)
                tiles = tiles_per_level[level]
                entry = tiles.get(key)
                if entry is None:
                    tiles[key] = [dz, dz, 1, [weight], [weighted_z]]
                else:
                    if dz < entry[0]:
                        entry[0] = dz
                    if dz > entry[1]:
                        entry[1] = dz
                    entry[2] += 1
                    entry[3].append(weight)
                    entry[4].append(weighted_z)

        pyramid = []
        for width, tiles in zip(widths, tiles_per_level):
            level_tiles = []
            for (tx, ty), entry in tiles.items():
                zmin, zmax, count, weights, weighted_zs = entry
                sum_w = _sorted_sum(weights)
                zmean = _sorted_sum(weighted_zs) / sum_w
                zsigma = (Decimal(1) / sum_w).sqrt()
                ix0 = tx * width
                iy0 = ty * width
                level_tiles.append((
                    tx, ty,
                    ix0, iy0,
                    ix0 + width - 1, iy0 + width - 1,
                    _quantize(zmin), _quantize(zmax),
                    _quantize(zmean), _quantize(zsigma),
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


def _validate_tile_region_stats(tiles) -> None:
    """Validate a flat tuple of tile-stat 11-tuples, raising ``ValueError``.

    Each tile must be an 11-tuple
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` where
    the first six fields and ``count`` are non-bool ints, the four statistics
    are finite floats, ``zsigma`` is positive and the tiles are sorted
    strictly by ``(tx, ty)`` with no duplicates.
    """
    prev_key = None
    for tile in tiles:
        if not isinstance(tile, tuple) or len(tile) != 11:
            raise ValueError(
                "each tile must be an 11-tuple "
                "(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)"
            )
        for value in tile[0:6] + (tile[10],):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and count must be "
                                 "non-bool ints")
        for value in tile[6:10]:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("zmin, zmax, zmean and zsigma must be finite "
                                 "floats")
        if tile[9] <= 0:
            raise ValueError("zsigma must be positive")
        key = (tile[0], tile[1])
        if prev_key is not None and key <= prev_key:
            raise ValueError("tiles must be sorted by (tx, ty) with no "
                             "duplicate coordinates")
        prev_key = key


def _validate_pyramid_stats(pyramid) -> None:
    """Validate the structure of a tile pyramid with z statistics.

    Every level must be a tuple of 11-tuples as checked by
    :func:`_validate_tile_region_stats`.
    """
    for level_tiles in pyramid:
        if not isinstance(level_tiles, tuple):
            raise ValueError("each pyramid level must be a tuple")
        _validate_tile_region_stats(level_tiles)


def query_tile_pyramid_stats(pyramid: tuple, level: int,
                             tx: int, ty: int) -> tuple | None:
    """Look up the tile ``(tx, ty)`` at ``level`` of a tile pyramid with stats.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    11-tuples sorted lexicographically by ``(tx, ty)``, where the first six
    fields and ``count`` are non-bool ints, the four statistics are finite
    floats and ``zsigma`` is positive. ``level``, ``tx`` and ``ty`` must be
    non-bool ints and ``level`` must satisfy ``0 <= level < len(pyramid)``.

    Returns the stored 11-tuple for the exact ``(tx, ty)`` match at that
    level, or ``None`` if no such tile exists. The input is never modified or
    reordered.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("tx", tx), ("ty", ty)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")

    _validate_pyramid_stats(pyramid)

    for tile in pyramid[level]:
        if tile[0] == tx and tile[1] == ty:
            return tile
    return None


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


def query_tile_region_stats(pyramid: tuple, level: int,
                            tx_min: int, ty_min: int,
                            tx_max: int, ty_max: int) -> tuple:
    """Select all tiles within a ``(tx, ty)`` rectangle at a stats-pyramid level.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    11-tuples sorted lexicographically by ``(tx, ty)``, where the first six
    fields and ``count`` are non-bool ints, the four statistics are finite
    floats and ``zsigma`` is positive. ``level`` and the four bounds must be
    non-bool ints; ``level`` must satisfy ``0 <= level < len(pyramid)`` and
    the bounds must satisfy ``tx_min <= tx_max`` and ``ty_min <= ty_max``.

    Returns a tuple of the stored 11-tuples at that level with
    ``tx_min <= tx <= tx_max`` and ``ty_min <= ty <= ty_max``, preserving the
    level's existing order; an empty region match returns ``()``. The input is
    never modified or reordered.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
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

    _validate_pyramid_stats(pyramid)

    return tuple(
        tile for tile in pyramid[level]
        if tx_min <= tile[0] <= tx_max and ty_min <= tile[1] <= ty_max
    )


def query_tile_window(pyramid: tuple, level: int,
                      ix_min: int, iy_min: int,
                      ix_max: int, iy_max: int) -> tuple:
    """Select all tiles whose cell-index window intersects a rectangle.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    lexicographically by ``(tx, ty)`` with no duplicates, where the first six
    fields and ``count`` are non-bool ints, ``zmin``/``zmax`` are finite
    floats and ``ix0 <= ix1``/``iy0 <= iy1``. ``level`` and the four window
    bounds must be non-bool ints; ``level`` must satisfy
    ``0 <= level < len(pyramid)`` and the bounds must satisfy
    ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    A tile matches when its closed cell-index intervals intersect the window:
    ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and tile.iy1 >= iy_min and
    tile.iy0 <= iy_max``. Returns a tuple of the stored 9-tuples at that
    level, preserving the level's existing order; an empty match returns
    ``()``. The input is never modified.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates, fields or cell bounds
        are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("ix_min", ix_min), ("iy_min", iy_min),
                        ("ix_max", ix_max), ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")

    _validate_pyramid(pyramid)

    for tile in pyramid:
        for entry in tile:
            if entry[4] < entry[2] or entry[5] < entry[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")

    return tuple(
        tile for tile in pyramid[level]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max)
    )


def query_tile_window_stats(pyramid: tuple, level: int,
                            ix_min: int, iy_min: int,
                            ix_max: int, iy_max: int) -> tuple:
    """Select all stats tiles whose cell-index window intersects a rectangle.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    11-tuples sorted lexicographically by ``(tx, ty)``, where the first six
    fields and ``count`` are non-bool ints, the four statistics are finite
    floats and ``zsigma`` is positive. ``level`` and the four window bounds
    must be non-bool ints; ``level`` must satisfy ``0 <= level < len(pyramid)``
    and the bounds must satisfy ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    A tile matches when its closed cell-index intervals intersect the window:
    ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and tile.iy1 >= iy_min and
    tile.iy0 <= iy_max``. Returns a tuple of the stored 11-tuples at that
    level, preserving the level's existing order; an empty match returns
    ``()``. The input is never modified or reordered.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("ix_min", ix_min), ("iy_min", iy_min),
                        ("ix_max", ix_max), ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")

    _validate_pyramid_stats(pyramid)

    return tuple(
        tile for tile in pyramid[level]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max)
    )


def aggregate_tile_region_stats(pyramid: tuple, level: int,
                                tx_min: int, ty_min: int,
                                tx_max: int, ty_max: int) -> tuple | None:
    """Aggregate the z statistics of tiles in a ``(tx, ty)`` rectangle.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    sorted lexicographically by ``(tx, ty)``, where the first six fields and
    ``count`` are non-bool ints, the four statistics are finite floats and
    ``zsigma`` is positive. ``level`` and the four bounds must be non-bool
    ints; ``level`` must satisfy ``0 <= level < len(pyramid)`` and the bounds
    must satisfy ``tx_min <= tx_max`` and ``ty_min <= ty_max``.

    Tiles at ``level`` with ``tx_min <= tx <= tx_max`` and
    ``ty_min <= ty <= ty_max`` (closed intervals) are aggregated: ``zmin`` is
    the minimum tile ``zmin``, ``zmax`` the maximum tile ``zmax`` and
    ``count`` the sum of the tile counts. The z statistics use weights
    ``w = 1 / zsigma ** 2``: ``zmean = sum(w * zmean) / sum(w)`` and
    ``zsigma = sqrt(1 / sum(w))``; the summations are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) with the summands accumulated in ascending Decimal
    order, so the result does not depend on tile order. The four statistics
    are quantized to six decimal places as floats (negative zero normalized).

    Returns ``(zmin, zmax, zmean, zsigma, count)`` for the matched tiles, or
    ``None`` when no tile lies within the rectangle. The input is never
    modified or reordered.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
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

    _validate_pyramid_stats(pyramid)

    matches = [
        tile for tile in pyramid[level]
        if tx_min <= tile[0] <= tx_max and ty_min <= tile[1] <= ty_max
    ]
    if not matches:
        return None

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zmin = Decimal(str(matches[0][6]))
        zmax = Decimal(str(matches[0][7]))
        count = 0
        weights: list[Decimal] = []
        weighted_means: list[Decimal] = []
        for tile in matches:
            tile_zmin = Decimal(str(tile[6]))
            tile_zmax = Decimal(str(tile[7]))
            if tile_zmin < zmin:
                zmin = tile_zmin
            if tile_zmax > zmax:
                zmax = tile_zmax
            count += tile[10]
            dzsigma = Decimal(str(tile[9]))
            weight = Decimal(1) / (dzsigma * dzsigma)
            weights.append(weight)
            weighted_means.append(weight * Decimal(str(tile[8])))

        sum_w = _sorted_sum(weights)
        zmean = _sorted_sum(weighted_means) / sum_w
        zsigma = (Decimal(1) / sum_w).sqrt()

    return (_quantize(zmin), _quantize(zmax),
            _quantize(zmean), _quantize(zsigma), count)


def aggregate_tile_window_stats(pyramid: tuple, level: int,
                                ix_min: int, iy_min: int,
                                ix_max: int, iy_max: int) -> tuple | None:
    """Aggregate the z statistics of tiles intersecting a cell-index window.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    sorted lexicographically by ``(tx, ty)``, where the first six fields and
    ``count`` are non-bool ints, the four statistics are finite floats and
    ``zsigma`` is positive. ``level`` and the four window bounds must be
    non-bool ints; ``level`` must satisfy ``0 <= level < len(pyramid)`` and
    the bounds must satisfy ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    Tiles at ``level`` whose closed cell-index intervals intersect the window
    (``tile.ix1 >= ix_min and tile.ix0 <= ix_max and tile.iy1 >= iy_min and
    tile.iy0 <= iy_max``) are aggregated in the level's existing order:
    ``zmin`` is the minimum tile ``zmin``, ``zmax`` the maximum tile ``zmax``
    and ``count`` the sum of the tile counts. The z statistics use weights
    ``w = 1 / zsigma ** 2``: ``zmean = sum(w * zmean) / sum(w)`` and
    ``zsigma = sqrt(1 / sum(w))``; the summations are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) with the summands accumulated in ascending Decimal
    order, so the result does not depend on tile order. The four statistics
    are quantized to six decimal places as floats (negative zero normalized).

    Returns ``(zmin, zmax, zmean, zsigma, count)`` for the matched tiles, or
    ``None`` when no tile intersects the window. The input is never modified
    or reordered.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("ix_min", ix_min), ("iy_min", iy_min),
                        ("ix_max", ix_max), ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")

    _validate_pyramid_stats(pyramid)

    matches = [
        tile for tile in pyramid[level]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max)
    ]
    if not matches:
        return None

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zmin = Decimal(str(matches[0][6]))
        zmax = Decimal(str(matches[0][7]))
        count = 0
        weights: list[Decimal] = []
        weighted_means: list[Decimal] = []
        for tile in matches:
            tile_zmin = Decimal(str(tile[6]))
            tile_zmax = Decimal(str(tile[7]))
            if tile_zmin < zmin:
                zmin = tile_zmin
            if tile_zmax > zmax:
                zmax = tile_zmax
            count += tile[10]
            dzsigma = Decimal(str(tile[9]))
            weight = Decimal(1) / (dzsigma * dzsigma)
            weights.append(weight)
            weighted_means.append(weight * Decimal(str(tile[8])))

        sum_w = _sorted_sum(weights)
        zmean = _sorted_sum(weighted_means) / sum_w
        zsigma = (Decimal(1) / sum_w).sqrt()

    return (_quantize(zmin), _quantize(zmax),
            _quantize(zmean), _quantize(zsigma), count)


def assess_tile_region_stats(estimate: tuple, reference: tuple) -> tuple:
    """Assess estimated tile-region statistics against a reference, tile by tile.

    Both inputs are tuples of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` sorted
    strictly by ``(tx, ty)`` with no duplicates, where ``tx``, ``ty``, the four
    cell bounds and ``count`` are non-bool ints and ``zmin``, ``zmax``,
    ``zmean`` and ``zsigma`` are finite floats with ``zsigma > 0``. The two
    inputs must contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds (their ``count`` values may differ);
    otherwise ``ValueError`` is raised.

    Returns a tuple, in the inputs' order, of
    ``(tx, ty, bias, abs_error, combined_sigma, z_score)`` tuples where
    ``bias = estimate.zmean - reference.zmean``, ``abs_error = abs(bias)``,
    ``combined_sigma = sqrt(estimate.zsigma**2 + reference.zsigma**2)`` and
    ``z_score = bias / combined_sigma``. The metrics are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) quantized to six decimal places as floats (negative
    zero normalized). Two empty inputs return ``()``. The inputs are never
    modified.

    :raises TypeError: ``estimate`` or ``reference`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields, coordinate
        sets or cell bounds of either input are bad.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")

    _validate_tile_region_stats(estimate)
    _validate_tile_region_stats(reference)

    if len(estimate) != len(reference):
        raise ValueError("estimate and reference must contain the same "
                         "(tx, ty) tile coordinates")
    for est_tile, ref_tile in zip(estimate, reference):
        if est_tile[0:2] != ref_tile[0:2]:
            raise ValueError("estimate and reference must contain the same "
                             "(tx, ty) tile coordinates")
        if est_tile[2:6] != ref_tile[2:6]:
            raise ValueError("tiles with the same (tx, ty) must agree on "
                             "(ix0, iy0, ix1, iy1)")

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for est_tile, ref_tile in zip(estimate, reference):
            est_zmean = Decimal(str(est_tile[8]))
            ref_zmean = Decimal(str(ref_tile[8]))
            est_zsigma = Decimal(str(est_tile[9]))
            ref_zsigma = Decimal(str(ref_tile[9]))
            bias = est_zmean - ref_zmean
            combined = (est_zsigma * est_zsigma
                        + ref_zsigma * ref_zsigma).sqrt()
            result.append((
                est_tile[0], est_tile[1],
                _quantize(bias), _quantize(abs(bias)),
                _quantize(combined), _quantize(bias / combined),
            ))

    return tuple(result)


def assess_tile_window_stats(estimate: tuple, reference: tuple) -> tuple:
    """Assess estimated tile-window statistics against a reference, tile by tile.

    Both inputs are tuples of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` sorted
    strictly by ``(tx, ty)`` with no duplicates, where ``tx``, ``ty``, the four
    cell bounds and ``count`` are non-bool ints, the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``zmin``, ``zmax``, ``zmean`` and
    ``zsigma`` are finite floats with ``zsigma > 0``. The two inputs must
    contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds (their ``count`` values may differ);
    otherwise ``ValueError`` is raised.

    Returns a tuple, in the inputs' order, of
    ``(tx, ty, bias, abs_error, combined_sigma, z_score)`` tuples where
    ``bias = estimate.zmean - reference.zmean``, ``abs_error = abs(bias)``,
    ``combined_sigma = sqrt(estimate.zsigma**2 + reference.zsigma**2)`` and
    ``z_score = bias / combined_sigma``. The metrics are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) quantized to six decimal places as floats (negative
    zero normalized). Two empty inputs return ``()``. The inputs are never
    modified.

    :raises TypeError: ``estimate`` or ``reference`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, coordinate sets or shared cell bounds of either input are bad.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")

    _validate_tile_region_stats(estimate)
    _validate_tile_region_stats(reference)

    for tiles in (estimate, reference):
        for tile in tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")

    if len(estimate) != len(reference):
        raise ValueError("estimate and reference must contain the same "
                         "(tx, ty) tile coordinates")
    for est_tile, ref_tile in zip(estimate, reference):
        if est_tile[0:2] != ref_tile[0:2]:
            raise ValueError("estimate and reference must contain the same "
                             "(tx, ty) tile coordinates")
        if est_tile[2:6] != ref_tile[2:6]:
            raise ValueError("tiles with the same (tx, ty) must agree on "
                             "(ix0, iy0, ix1, iy1)")

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for est_tile, ref_tile in zip(estimate, reference):
            est_zmean = Decimal(str(est_tile[8]))
            ref_zmean = Decimal(str(ref_tile[8]))
            est_zsigma = Decimal(str(est_tile[9]))
            ref_zsigma = Decimal(str(ref_tile[9]))
            bias = est_zmean - ref_zmean
            combined = (est_zsigma * est_zsigma
                        + ref_zsigma * ref_zsigma).sqrt()
            result.append((
                est_tile[0], est_tile[1],
                _quantize(bias), _quantize(abs(bias)),
                _quantize(combined), _quantize(bias / combined),
            ))

    return tuple(result)


def assess_tile_pyramid_stats(estimate: tuple, reference: tuple) -> tuple:
    """Assess estimated tile-pyramid statistics against a reference, tile by tile.

    Both inputs are outer tuples as produced by
    :func:`build_tile_pyramid_stats` with the same number of levels: each level
    is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` sorted
    strictly by ``(tx, ty)`` with no duplicates, where ``tx``, ``ty``, the four
    cell bounds and ``count`` are non-bool ints, the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``zmin``, ``zmax``, ``zmean`` and
    ``zsigma`` are finite floats with ``zsigma > 0``. At every level the two
    inputs must contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds (their ``count`` values may differ);
    otherwise ``ValueError`` is raised.

    Returns a tuple, in level order, of tuples (in each level's original
    order) of
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` 11-tuples where
    ``bias = estimate.zmean - reference.zmean``, ``abs_error = abs(bias)``,
    ``combined_sigma = sqrt(estimate.zsigma**2 + reference.zsigma**2)``,
    ``z_score = bias / combined_sigma`` and
    ``count_delta = estimate.count - reference.count``. The four metrics are
    Decimal computations (precision 50, ``ROUND_HALF_EVEN``, each input
    converted via ``Decimal(str(v))``) quantized to six decimal places as
    floats (negative zero normalized). Empty levels are preserved; two empty
    inputs return ``()``. The inputs are never modified.

    :raises TypeError: ``estimate`` or ``reference`` is not a tuple.
    :raises ValueError: the level counts differ or the structure, ordering,
        duplicates, fields, cell bounds, coordinate sets or shared cell bounds
        of either input are bad.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")

    _validate_pyramid_stats(estimate)
    _validate_pyramid_stats(reference)

    for pyramid in (estimate, reference):
        for level_tiles in pyramid:
            for tile in level_tiles:
                if tile[4] < tile[2] or tile[5] < tile[3]:
                    raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                     "iy0 <= iy1")

    if len(estimate) != len(reference):
        raise ValueError("estimate and reference must have the same number "
                         "of levels")
    for est_level, ref_level in zip(estimate, reference):
        if len(est_level) != len(ref_level):
            raise ValueError("estimate and reference must contain the same "
                             "(tx, ty) tile coordinates at every level")
        for est_tile, ref_tile in zip(est_level, ref_level):
            if est_tile[0:2] != ref_tile[0:2]:
                raise ValueError("estimate and reference must contain the "
                                 "same (tx, ty) tile coordinates at every "
                                 "level")
            if est_tile[2:6] != ref_tile[2:6]:
                raise ValueError("tiles with the same (tx, ty) must agree on "
                                 "(ix0, iy0, ix1, iy1)")

    pyramid = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for est_level, ref_level in zip(estimate, reference):
            level_result = []
            for est_tile, ref_tile in zip(est_level, ref_level):
                est_zmean = Decimal(str(est_tile[8]))
                ref_zmean = Decimal(str(ref_tile[8]))
                est_zsigma = Decimal(str(est_tile[9]))
                ref_zsigma = Decimal(str(ref_tile[9]))
                bias = est_zmean - ref_zmean
                combined = (est_zsigma * est_zsigma
                            + ref_zsigma * ref_zsigma).sqrt()
                level_result.append((
                    est_tile[0], est_tile[1],
                    est_tile[2], est_tile[3], est_tile[4], est_tile[5],
                    _quantize(bias), _quantize(abs(bias)),
                    _quantize(combined), _quantize(bias / combined),
                    est_tile[10] - ref_tile[10],
                ))
            pyramid.append(tuple(level_result))

    return tuple(pyramid)


def _format_z(value: float) -> str:
    """Format a finite float with exactly six decimals (``-0`` normalized)."""
    text = format(value, ".6f")
    return "0.000000" if text == "-0.000000" else text


def encode_tile_pyramid(pyramid: tuple) -> str:
    """Serialize a tile pyramid (as produced by :func:`build_tile_pyramid`).

    ``pyramid`` must be the outer tuple: each level is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` with the first six
    fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats, and
    tiles sorted by ``(tx, ty)`` without duplicates.

    Returns a canonical compact JSON string whose sole top-level key is
    ``levels``: an array (in level order) of arrays of tile arrays in the
    tiles' stored order. Integers are decimal; ``zmin``/``zmax`` use exactly
    six decimal places (negative zero written as ``0.000000``). The output has
    no whitespace, ASCII is not escaped and ``NaN``/``Infinity`` never appear.

    :raises TypeError: ``pyramid`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates or fields are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    _validate_pyramid(pyramid)

    parts = ['{"levels":[']
    for level_index, level_tiles in enumerate(pyramid):
        if level_index:
            parts.append(",")
        parts.append("[")
        for tile_index, tile in enumerate(level_tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(zmin),
                                   _format_z(zmax), str(count))))
            parts.append("]")
        parts.append("]")
    parts.append(']}')
    return "".join(parts)


def _format_region_text(level, tx_min, ty_min, tx_max, ty_max, tiles) -> str:
    """Build the canonical compact JSON text of a region document."""
    parts = ["{\"level\":", str(level),
             ",\"tx_min\":", str(tx_min),
             ",\"ty_min\":", str(ty_min),
             ",\"tx_max\":", str(tx_max),
             ",\"ty_max\":", str(ty_max),
             ",\"tiles\":["]
    first = True
    for tile in tiles:
        if not first:
            parts.append(",")
        first = False
        (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
        parts.append("[")
        parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                               str(ix1), str(iy1), _format_z(zmin),
                               _format_z(zmax), str(count))))
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_tile_region(pyramid: tuple, level: int,
                       tx_min: int, ty_min: int,
                       tx_max: int, ty_max: int) -> str:
    """Serialize the tiles of a ``(tx, ty)`` rectangle at one pyramid level.

    ``pyramid`` must be the outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` with the first six
    fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats, and
    tiles sorted by ``(tx, ty)`` without duplicates. ``level`` and the four
    bounds must be non-bool ints; ``level`` must satisfy
    ``0 <= level < len(pyramid)`` and the bounds must satisfy
    ``tx_min <= tx_max`` and ``ty_min <= ty_max``.

    Returns a canonical compact JSON string whose top-level keys, in fixed
    order, are ``level``, ``tx_min``, ``ty_min``, ``tx_max``, ``ty_max`` and
    ``tiles``. The first five values echo the arguments; ``tiles`` is the
    array of nine-value tiles at ``level`` with
    ``tx_min <= tx <= tx_max`` and ``ty_min <= ty <= ty_max``, in the level's
    stored order (``[]`` when nothing matches). Integers are decimal;
    ``zmin``/``zmax`` use exactly six decimal places (negative zero written as
    ``0.000000``). The output has no whitespace, ASCII is not escaped and
    ``NaN``/``Infinity`` never appear. The input is never modified.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
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

    tiles = (
        tile for tile in pyramid[level]
        if tx_min <= tile[0] <= tx_max and ty_min <= tile[1] <= ty_max
    )
    return _format_region_text(
        level, tx_min, ty_min, tx_max, ty_max, tiles)


def _format_window_text(level, ix_min, iy_min, ix_max, iy_max, tiles) -> str:
    """Build the canonical compact JSON text of a window document."""
    parts = ["{\"level\":", str(level),
             ",\"ix_min\":", str(ix_min),
             ",\"iy_min\":", str(iy_min),
             ",\"ix_max\":", str(ix_max),
             ",\"iy_max\":", str(iy_max),
             ",\"tiles\":["]
    first = True
    for tile in tiles:
        if not first:
            parts.append(",")
        first = False
        (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
        parts.append("[")
        parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                               str(ix1), str(iy1), _format_z(zmin),
                               _format_z(zmax), str(count))))
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_tile_window(pyramid: tuple, level: int,
                       ix_min: int, iy_min: int,
                       ix_max: int, iy_max: int) -> str:
    """Serialize the tiles intersecting a cell-index window at one pyramid level.

    ``pyramid`` must be the outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` with the first six
    fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1``, and tiles sorted by ``(tx, ty)`` without
    duplicates. ``level`` and the four window bounds must be non-bool ints;
    ``level`` must satisfy ``0 <= level < len(pyramid)`` and the bounds must
    satisfy ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    Returns a canonical compact JSON string whose top-level keys, in fixed
    order, are ``level``, ``ix_min``, ``iy_min``, ``ix_max``, ``iy_max`` and
    ``tiles``. The first five values echo the arguments; ``tiles`` is the
    array of nine-value tiles at ``level`` whose closed cell-index intervals
    intersect the window (``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``), in the level's stored order
    (``[]`` when nothing matches). Integers are decimal; ``zmin``/``zmax`` use
    exactly six decimal places (negative zero written as ``0.000000``). The
    output has no whitespace, ASCII is not escaped and ``NaN``/``Infinity``
    never appear. The input is never modified.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates, fields or cell bounds
        are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("ix_min", ix_min), ("iy_min", iy_min),
                        ("ix_max", ix_max), ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")

    _validate_pyramid(pyramid)

    for level_tiles in pyramid:
        for entry in level_tiles:
            if entry[4] < entry[2] or entry[5] < entry[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")

    tiles = (
        tile for tile in pyramid[level]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max)
    )
    return _format_window_text(
        level, ix_min, iy_min, ix_max, iy_max, tiles)


def decode_tile_window(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_window`.

    The document must be the compact encoder output: top-level keys exactly
    ``level``, ``ix_min``, ``iy_min``, ``ix_max``, ``iy_max`` and ``tiles`` in
    that order; the first five values non-bool ints with ``level >= 0`` and
    ``ix_min <= ix_max``/``iy_min <= iy_max``; ``tiles`` an array of strict
    9-tuples ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` whose indices
    and ``count`` are non-bool ints, whose ``zmin``/``zmax`` are finite floats
    with ``ix0 <= ix1``/``iy0 <= iy1``, whose ``(tx, ty)`` coordinates are
    strictly increasing with no duplicates and whose closed cell-index
    intervals intersect the window (``ix1 >= ix_min and ix0 <= ix_max and
    iy1 >= iy_min and iy0 <= iy_max``), and whose spelling is exactly
    canonical (integers in decimal, ``zmin``/``zmax`` with six decimals,
    negative zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns ``(level, ix_min, iy_min, ix_max, iy_max, tiles_tuple)`` where
    ``tiles_tuple`` is a tuple of 9-tuples in the document's order. The input
    text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, bounds, ordering,
        duplicates, window intersection, non-finite values, inverted tile
        bounds, numeric formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    expected_keys = ("level", "ix_min", "iy_min", "ix_max", "iy_max", "tiles")
    if (not isinstance(document, dict)
            or tuple(document) != expected_keys):
        raise ValueError(
            "top-level value must be an object with exactly the keys "
            "'level', 'ix_min', 'iy_min', 'ix_max', 'iy_max', 'tiles' "
            "in that order")

    level, ix_min, iy_min, ix_max, iy_max = (
        document[name] for name in expected_keys[:5])
    raw_tiles = document["tiles"]

    for name, value in (("level", level), ("ix_min", ix_min),
                        ("iy_min", iy_min), ("ix_max", ix_max),
                        ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a non-bool int")
    if level < 0:
        raise ValueError("level must be non-negative")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")
    if not isinstance(raw_tiles, list):
        raise ValueError("'tiles' must be an array")

    tiles: list[tuple] = []
    prev_key = None
    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, list) or len(raw_tile) != 9:
            raise ValueError("each tile must be an array of nine values")
        tile = tuple(raw_tile)
        for value in tile[0:6] + (tile[8],):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "tx, ty, ix0, iy0, ix1, iy1 and count must be non-bool "
                    "ints")
        for value in tile[6:8]:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("zmin and zmax must be finite floats")
        if tile[4] < tile[2] or tile[5] < tile[3]:
            raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                             "iy0 <= iy1")
        tx, ty = tile[0], tile[1]
        if not (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max):
            raise ValueError("each tile's cell-index interval must intersect "
                             "the window")
        key = (tx, ty)
        if prev_key is not None and key <= prev_key:
            raise ValueError("tiles must be sorted by (tx, ty) with no "
                             "duplicate coordinates")
        prev_key = key
        tiles.append(tile)
    tiles_tuple = tuple(tiles)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal z formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_window_text(level, ix_min, iy_min, ix_max, iy_max,
                           tiles_tuple) != text:
        raise ValueError("JSON text is not the canonical window encoding")
    return (level, ix_min, iy_min, ix_max, iy_max, tiles_tuple)


def decode_tile_region(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_region`.

    The document must be the compact encoder output: top-level keys exactly
    ``level``, ``tx_min``, ``ty_min``, ``tx_max``, ``ty_max`` and ``tiles`` in
    that order; the first five values non-bool ints with ``level >= 0`` and
    ``tx_min <= tx_max``/``ty_min <= ty_max``; ``tiles`` an array of strict
    9-tuples ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` whose indices
    and ``count`` are non-bool ints, whose ``zmin``/``zmax`` are finite floats,
    whose ``(tx, ty)`` coordinates are strictly increasing with no duplicates
    and lie within the echoed rectangle, and whose spelling is exactly
    canonical (integers in decimal, ``zmin``/``zmax`` with six decimals,
    negative zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns ``(level, tx_min, ty_min, tx_max, ty_max, tiles_tuple)`` where
    ``tiles_tuple`` is a tuple of 9-tuples in the document's order. The input
    text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, bounds, ordering,
        duplicates, range, non-finite values, numeric formatting or canonical
        re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    expected_keys = ("level", "tx_min", "ty_min", "tx_max", "ty_max", "tiles")
    if (not isinstance(document, dict)
            or tuple(document) != expected_keys):
        raise ValueError(
            "top-level value must be an object with exactly the keys "
            "'level', 'tx_min', 'ty_min', 'tx_max', 'ty_max', 'tiles' "
            "in that order")

    level, tx_min, ty_min, tx_max, ty_max = (
        document[name] for name in expected_keys[:5])
    raw_tiles = document["tiles"]

    for name, value in (("level", level), ("tx_min", tx_min),
                        ("ty_min", ty_min), ("tx_max", tx_max),
                        ("ty_max", ty_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a non-bool int")
    if level < 0:
        raise ValueError("level must be non-negative")
    if tx_min > tx_max or ty_min > ty_max:
        raise ValueError("region bounds must satisfy tx_min <= tx_max and "
                         "ty_min <= ty_max")
    if not isinstance(raw_tiles, list):
        raise ValueError("'tiles' must be an array")

    tiles: list[tuple] = []
    prev_key = None
    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, list) or len(raw_tile) != 9:
            raise ValueError("each tile must be an array of nine values")
        tile = tuple(raw_tile)
        for value in tile[0:6] + (tile[8],):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "tx, ty, ix0, iy0, ix1, iy1 and count must be non-bool "
                    "ints")
        for value in tile[6:8]:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("zmin and zmax must be finite floats")
        tx, ty = tile[0], tile[1]
        if not (tx_min <= tx <= tx_max and ty_min <= ty <= ty_max):
            raise ValueError("each tile must lie within the region rectangle")
        key = (tx, ty)
        if prev_key is not None and key <= prev_key:
            raise ValueError("tiles must be sorted by (tx, ty) with no "
                             "duplicate coordinates")
        prev_key = key
        tiles.append(tile)
    tiles_tuple = tuple(tiles)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal z formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_region_text(level, tx_min, ty_min, tx_max, ty_max,
                           tiles_tuple) != text:
        raise ValueError("JSON text is not the canonical region encoding")
    return (level, tx_min, ty_min, tx_max, ty_max, tiles_tuple)


def _format_region_stats_text(level, tx_min, ty_min, tx_max, ty_max,
                              tiles) -> str:
    """Build the canonical compact JSON text of a stats region document."""
    parts = ["{\"level\":", str(level),
             ",\"tx_min\":", str(tx_min),
             ",\"ty_min\":", str(ty_min),
             ",\"tx_max\":", str(tx_max),
             ",\"ty_max\":", str(ty_max),
             ",\"tiles\":["]
    first = True
    for tile in tiles:
        if not first:
            parts.append(",")
        first = False
        (tx, ty, ix0, iy0, ix1, iy1,
         zmin, zmax, zmean, zsigma, count) = tile
        parts.append("[")
        parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                               str(ix1), str(iy1), _format_z(zmin),
                               _format_z(zmax), _format_z(zmean),
                               _format_z(zsigma), str(count))))
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_tile_region_stats(pyramid: tuple, level: int,
                             tx_min: int, ty_min: int,
                             tx_max: int, ty_max: int) -> str:
    """Serialize the tiles of a ``(tx, ty)`` rectangle at a stats-pyramid level.

    ``pyramid`` must be the outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` with
    the first six fields and ``count`` non-bool ints, the four statistics
    finite floats (``zsigma`` positive), and tiles sorted by ``(tx, ty)``
    without duplicates. ``level`` and the four bounds must be non-bool ints;
    ``level`` must satisfy ``0 <= level < len(pyramid)`` and the bounds must
    satisfy ``tx_min <= tx_max`` and ``ty_min <= ty_max``.

    Returns a canonical compact JSON string whose top-level keys, in fixed
    order, are ``level``, ``tx_min``, ``ty_min``, ``tx_max``, ``ty_max`` and
    ``tiles``. The first five values echo the arguments; ``tiles`` is the
    array of eleven-value tiles at ``level`` with
    ``tx_min <= tx <= tx_max`` and ``ty_min <= ty <= ty_max``, in the level's
    stored order (``[]`` when nothing matches). Integers are decimal; the four
    statistics use exactly six decimal places (negative zero written as
    ``0.000000``). The output has no whitespace, ASCII is not escaped and
    ``NaN``/``Infinity`` never appear. The input is never modified.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
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

    _validate_pyramid_stats(pyramid)

    tiles = (
        tile for tile in pyramid[level]
        if tx_min <= tile[0] <= tx_max and ty_min <= tile[1] <= ty_max
    )
    return _format_region_stats_text(
        level, tx_min, ty_min, tx_max, ty_max, tiles)


def _format_window_stats_text(level, ix_min, iy_min, ix_max, iy_max,
                              tiles) -> str:
    """Build the canonical compact JSON text of a stats window document."""
    parts = ["{\"level\":", str(level),
             ",\"ix_min\":", str(ix_min),
             ",\"iy_min\":", str(iy_min),
             ",\"ix_max\":", str(ix_max),
             ",\"iy_max\":", str(iy_max),
             ",\"tiles\":["]
    first = True
    for tile in tiles:
        if not first:
            parts.append(",")
        first = False
        (tx, ty, ix0, iy0, ix1, iy1,
         zmin, zmax, zmean, zsigma, count) = tile
        parts.append("[")
        parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                               str(ix1), str(iy1), _format_z(zmin),
                               _format_z(zmax), _format_z(zmean),
                               _format_z(zsigma), str(count))))
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_tile_window_stats(pyramid: tuple, level: int,
                             ix_min: int, iy_min: int,
                             ix_max: int, iy_max: int) -> str:
    """Serialize the stats tiles intersecting a cell-index window at a level.

    ``pyramid`` must be the outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` with
    the first six fields and ``count`` non-bool ints, the four statistics
    finite floats (``zsigma`` positive), and tiles sorted by ``(tx, ty)``
    without duplicates. ``level`` and the four window bounds must be non-bool
    ints; ``level`` must satisfy ``0 <= level < len(pyramid)`` and the bounds
    must satisfy ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    Returns a canonical compact JSON string whose top-level keys, in fixed
    order, are ``level``, ``ix_min``, ``iy_min``, ``ix_max``, ``iy_max`` and
    ``tiles``. The first five values echo the arguments; ``tiles`` is the
    array of eleven-value tiles at ``level`` whose closed cell-index intervals
    intersect the window (``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``), in the level's stored order
    (``[]`` when nothing matches). Integers are decimal; the four statistics
    use exactly six decimal places (negative zero written as ``0.000000``).
    The output has no whitespace, ASCII is not escaped and
    ``NaN``/``Infinity`` never appear. The input is never modified.

    :raises TypeError: ``pyramid`` is not a tuple or ``level``/a bound is not
        a non-bool int.
    :raises ValueError: ``level`` is out of range, the bounds are inverted or
        the pyramid's structure, ordering, duplicates or fields are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    for name, value in (("level", level), ("ix_min", ix_min), ("iy_min", iy_min),
                        ("ix_max", ix_max), ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(pyramid):
        raise ValueError("level out of range")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")

    _validate_pyramid_stats(pyramid)

    tiles = (
        tile for tile in pyramid[level]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max)
    )
    return _format_window_stats_text(
        level, ix_min, iy_min, ix_max, iy_max, tiles)


def decode_tile_window_stats(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_window_stats`.

    The document must be the compact encoder output: top-level keys exactly
    ``level``, ``ix_min``, ``iy_min``, ``ix_max``, ``iy_max`` and ``tiles`` in
    that order; the first five values non-bool ints with ``level >= 0`` and
    ``ix_min <= ix_max``/``iy_min <= iy_max``; ``tiles`` an array of strict
    11-tuples ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma,
    count)`` whose indices and ``count`` are non-bool ints, whose four
    statistics are finite floats with ``zsigma > 0``, whose ``(tx, ty)``
    coordinates are strictly increasing with no duplicates and whose closed
    cell-index intervals intersect the window (``ix1 >= ix_min and
    ix0 <= ix_max and iy1 >= iy_min and iy0 <= iy_max``), and whose spelling is
    exactly canonical (integers in decimal, statistics with six decimals,
    negative zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns ``(level, ix_min, iy_min, ix_max, iy_max, tiles_tuple)`` where
    ``tiles_tuple`` is a tuple of 11-tuples in the document's order. The input
    text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, bounds, ordering,
        duplicates, window intersection, non-finite values, non-positive
        ``zsigma``, numeric formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    expected_keys = ("level", "ix_min", "iy_min", "ix_max", "iy_max", "tiles")
    if (not isinstance(document, dict)
            or tuple(document) != expected_keys):
        raise ValueError(
            "top-level value must be an object with exactly the keys "
            "'level', 'ix_min', 'iy_min', 'ix_max', 'iy_max', 'tiles' "
            "in that order")

    level, ix_min, iy_min, ix_max, iy_max = (
        document[name] for name in expected_keys[:5])
    raw_tiles = document["tiles"]

    for name, value in (("level", level), ("ix_min", ix_min),
                        ("iy_min", iy_min), ("ix_max", ix_max),
                        ("iy_max", iy_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a non-bool int")
    if level < 0:
        raise ValueError("level must be non-negative")
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("window bounds must satisfy ix_min <= ix_max and "
                         "iy_min <= iy_max")
    if not isinstance(raw_tiles, list):
        raise ValueError("'tiles' must be an array")

    tiles: list[tuple] = []
    prev_key = None
    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, list) or len(raw_tile) != 11:
            raise ValueError("each tile must be an array of eleven values")
        tile = tuple(raw_tile)
        for value in tile[0:6] + (tile[10],):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "tx, ty, ix0, iy0, ix1, iy1 and count must be non-bool "
                    "ints")
        for value in tile[6:10]:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("zmin, zmax, zmean and zsigma must be finite "
                                 "floats")
        if tile[9] <= 0:
            raise ValueError("zsigma must be positive")
        tx, ty = tile[0], tile[1]
        if not (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max):
            raise ValueError("each tile's cell-index interval must intersect "
                             "the window")
        key = (tx, ty)
        if prev_key is not None and key <= prev_key:
            raise ValueError("tiles must be sorted by (tx, ty) with no "
                             "duplicate coordinates")
        prev_key = key
        tiles.append(tile)
    tiles_tuple = tuple(tiles)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal statistic formatting, leading zeros,
    # -0, exponents and any other non-canonical spelling.
    if _format_window_stats_text(level, ix_min, iy_min, ix_max, iy_max,
                                 tiles_tuple) != text:
        raise ValueError("JSON text is not the canonical window-stats "
                         "encoding")
    return (level, ix_min, iy_min, ix_max, iy_max, tiles_tuple)


def decode_tile_region_stats(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_region_stats`.

    The document must be the compact encoder output: top-level keys exactly
    ``level``, ``tx_min``, ``ty_min``, ``tx_max``, ``ty_max`` and ``tiles`` in
    that order; the first five values non-bool ints with ``level >= 0`` and
    ``tx_min <= tx_max``/``ty_min <= ty_max``; ``tiles`` an array of strict
    11-tuples ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma,
    count)`` whose indices and ``count`` are non-bool ints, whose four
    statistics are finite floats with ``zsigma > 0``, whose ``(tx, ty)``
    coordinates are strictly increasing with no duplicates and lie within the
    echoed rectangle, and whose spelling is exactly canonical (integers in
    decimal, statistics with six decimals, negative zero as ``0.000000``, no
    whitespace or extra keys, no ``NaN``/``Infinity``).

    Returns ``(level, tx_min, ty_min, tx_max, ty_max, tiles_tuple)`` where
    ``tiles_tuple`` is a tuple of 11-tuples in the document's order. The input
    text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, bounds, ordering,
        duplicates, range, non-finite values, non-positive ``zsigma``, numeric
        formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    expected_keys = ("level", "tx_min", "ty_min", "tx_max", "ty_max", "tiles")
    if (not isinstance(document, dict)
            or tuple(document) != expected_keys):
        raise ValueError(
            "top-level value must be an object with exactly the keys "
            "'level', 'tx_min', 'ty_min', 'tx_max', 'ty_max', 'tiles' "
            "in that order")

    level, tx_min, ty_min, tx_max, ty_max = (
        document[name] for name in expected_keys[:5])
    raw_tiles = document["tiles"]

    for name, value in (("level", level), ("tx_min", tx_min),
                        ("ty_min", ty_min), ("tx_max", tx_max),
                        ("ty_max", ty_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a non-bool int")
    if level < 0:
        raise ValueError("level must be non-negative")
    if tx_min > tx_max or ty_min > ty_max:
        raise ValueError("region bounds must satisfy tx_min <= tx_max and "
                         "ty_min <= ty_max")
    if not isinstance(raw_tiles, list):
        raise ValueError("'tiles' must be an array")

    tiles: list[tuple] = []
    prev_key = None
    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, list) or len(raw_tile) != 11:
            raise ValueError("each tile must be an array of eleven values")
        tile = tuple(raw_tile)
        for value in tile[0:6] + (tile[10],):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "tx, ty, ix0, iy0, ix1, iy1 and count must be non-bool "
                    "ints")
        for value in tile[6:10]:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("zmin, zmax, zmean and zsigma must be finite "
                                 "floats")
        if tile[9] <= 0:
            raise ValueError("zsigma must be positive")
        tx, ty = tile[0], tile[1]
        if not (tx_min <= tx <= tx_max and ty_min <= ty <= ty_max):
            raise ValueError("each tile must lie within the region rectangle")
        key = (tx, ty)
        if prev_key is not None and key <= prev_key:
            raise ValueError("tiles must be sorted by (tx, ty) with no "
                             "duplicate coordinates")
        prev_key = key
        tiles.append(tile)
    tiles_tuple = tuple(tiles)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal statistic formatting, leading zeros,
    # -0, exponents and any other non-canonical spelling.
    if _format_region_stats_text(level, tx_min, ty_min, tx_max, ty_max,
                                 tiles_tuple) != text:
        raise ValueError("JSON text is not the canonical region-stats "
                         "encoding")
    return (level, tx_min, ty_min, tx_max, ty_max, tiles_tuple)


def _reject_constant(raw: str):
    """Reject JSON non-finite literals (``NaN``/``Infinity``) at parse time."""
    raise ValueError(f"non-finite JSON literal is not allowed: {raw}")


def decode_tile_pyramid(text: str) -> tuple:
    """Deserialize a canonical JSON document produced by :func:`encode_tile_pyramid`.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, top-level keys/types, level/tile
        shape, numeric formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"levels"}:
        raise ValueError("top-level value must be an object with only 'levels'")
    raw_levels = document["levels"]
    if not isinstance(raw_levels, list):
        raise ValueError("'levels' must be an array")

    levels: list[tuple] = []
    for raw_tiles in raw_levels:
        if not isinstance(raw_tiles, list):
            raise ValueError("each level must be an array of tiles")
        tiles: list[tuple] = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 9:
                raise ValueError("each tile must be an array of nine values")
            tiles.append(tuple(raw_tile))
        levels.append(tuple(tiles))
    pyramid = tuple(levels)

    # Structural rules: tuple levels, field types/finiteness, (tx, ty) sort
    # order and no duplicates.
    _validate_pyramid(pyramid)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal z formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if encode_tile_pyramid(pyramid) != text:
        raise ValueError("JSON text is not the canonical pyramid encoding")
    return pyramid


def encode_tile_pyramid_stats(pyramid: tuple) -> str:
    """Serialize a tile pyramid with z statistics.

    ``pyramid`` must be the outer tuple as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` with
    the first six fields and ``count`` non-bool ints, the four statistics
    finite floats (``zsigma`` positive), and tiles sorted by ``(tx, ty)``
    without duplicates.

    Returns a canonical compact JSON string whose sole top-level key is
    ``levels``: an array (in level order) of arrays of tile arrays in the
    tiles' stored order. Integers are decimal; the four statistics use exactly
    six decimal places (negative zero written as ``0.000000``). The output has
    no whitespace, ASCII is not escaped and ``NaN``/``Infinity`` never appear.

    :raises TypeError: ``pyramid`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates or fields are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    _validate_pyramid_stats(pyramid)

    parts = ['{"levels":[']
    for level_index, level_tiles in enumerate(pyramid):
        if level_index:
            parts.append(",")
        parts.append("[")
        for tile_index, tile in enumerate(level_tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1,
             zmin, zmax, zmean, zsigma, count) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(zmin),
                                   _format_z(zmax), _format_z(zmean),
                                   _format_z(zsigma), str(count))))
            parts.append("]")
        parts.append("]")
    parts.append(']}')
    return "".join(parts)


def _validate_pyramid_assessment(assessment) -> None:
    """Validate the outer tuple returned by :func:`assess_tile_pyramid_stats`.

    Every level must be a tuple of strict 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)``: the first six fields and ``count_delta`` are non-bool ints
    with ``ix0 <= ix1`` and ``iy0 <= iy1``; the four metrics are finite floats
    with ``combined_sigma > 0``; and the tiles are sorted strictly by
    ``(tx, ty)`` with no duplicates.
    """
    for level_tiles in assessment:
        if not isinstance(level_tiles, tuple):
            raise ValueError("each assessment level must be a tuple")
        prev_key = None
        for tile in level_tiles:
            if not isinstance(tile, tuple) or len(tile) != 11:
                raise ValueError(
                    "each tile must be an 11-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, "
                    "combined_sigma, z_score, count_delta)"
                )
            for value in tile[0:6] + (tile[10],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "count_delta must be non-bool ints")
            for value in tile[6:10]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("bias, abs_error, combined_sigma and "
                                     "z_score must be finite floats")
            if tile[8] <= 0:
                raise ValueError("combined_sigma must be positive")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("each assessment level must be sorted by "
                                 "(tx, ty) with no duplicate coordinates")
            prev_key = key


def encode_tile_pyramid_assessment(assessment: tuple) -> str:
    """Serialize the result of :func:`assess_tile_pyramid_stats`.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` with tiles sorted strictly by ``(tx, ty)`` and no
    duplicates, where the first six fields and ``count_delta`` are non-bool
    ints satisfying ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``bias``,
    ``abs_error``, ``combined_sigma`` and ``z_score`` are finite floats with
    ``combined_sigma > 0``.

    Returns a canonical compact JSON string whose sole top-level key is
    ``levels``: an array (in level order) of arrays of tile arrays in the
    tiles' input order. Integers are decimal; the four metrics use exactly six
    decimal places (negative zero written as ``0.000000``). The output has no
    whitespace, ASCII is not escaped and ``NaN``/``Infinity`` never appear. An
    empty assessment encodes as ``{"levels":[]}``. The input is never modified.

    :raises TypeError: ``assessment`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields or cell
        bounds are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    _validate_pyramid_assessment(assessment)

    parts = ['{"levels":[']
    for level_index, level_tiles in enumerate(assessment):
        if level_index:
            parts.append(",")
        parts.append("[")
        for tile_index, tile in enumerate(level_tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1,
             bias, abs_error, combined_sigma, z_score,
             count_delta) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(bias),
                                   _format_z(abs_error),
                                   _format_z(combined_sigma),
                                   _format_z(z_score), str(count_delta))))
            parts.append("]")
        parts.append("]")
    parts.append(']}')
    return "".join(parts)


def decode_tile_pyramid_assessment(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_pyramid_assessment`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``levels``; ``levels`` an array (in level order) of
    arrays of strict 11-item tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta]`` whose first six fields and ``count_delta`` are non-bool
    ints with ``ix0 <= ix1`` and ``iy0 <= iy1``, whose four metrics are
    finite floats with ``combined_sigma > 0``, and whose ``(tx, ty)``
    coordinates are strictly increasing with no duplicates within each
    level, and whose spelling is exactly canonical (integers in decimal,
    metrics with six decimals, negative zero as ``0.000000``, no whitespace
    or extra keys, no ``NaN``/``Infinity``).

    Returns the outer level-order tuple of tuples of 11-tuples in the
    document's order; the empty document ``{"levels":[]}`` returns ``()``.
    The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, top-level keys/types, level/tile
        shape, field types, bounds, ordering, duplicates, non-finite values,
        non-positive ``combined_sigma``, numeric formatting or canonical
        re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"levels"}:
        raise ValueError("top-level value must be an object with only 'levels'")
    raw_levels = document["levels"]
    if not isinstance(raw_levels, list):
        raise ValueError("'levels' must be an array")

    levels: list[tuple] = []
    for raw_tiles in raw_levels:
        if not isinstance(raw_tiles, list):
            raise ValueError("each level must be an array of tiles")
        tiles: list[tuple] = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 11:
                raise ValueError("each tile must be an array of eleven values")
            tiles.append(tuple(raw_tile))
        levels.append(tuple(tiles))
    assessment = tuple(levels)

    # Structural rules: tuple levels, field types/finiteness, positive
    # combined_sigma, cell bounds, (tx, ty) sort order and no duplicates.
    _validate_pyramid_assessment(assessment)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, leading zeros,
    # -0, exponents and any other non-canonical spelling.
    if encode_tile_pyramid_assessment(assessment) != text:
        raise ValueError("JSON text is not the canonical pyramid-assessment "
                         "encoding")
    return assessment


def query_tile_pyramid_assessment(assessment: tuple, level: int,
                                  tx: int, ty: int) -> tuple | None:
    """Look up the tile ``(tx, ty)`` at ``level`` of a pyramid assessment.

    ``assessment`` must be an outer tuple as returned by
    :func:`assess_tile_pyramid_stats`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` 11-tuples sorted lexicographically by ``(tx, ty)`` with no
    duplicates, where the first six fields and ``count_delta`` are non-bool
    ints satisfying ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``bias``,
    ``abs_error``, ``combined_sigma`` and ``z_score`` are finite floats with
    ``combined_sigma > 0``. ``level``, ``tx`` and ``ty`` must be non-bool ints
    and ``level`` must satisfy ``0 <= level < len(assessment)``.

    Returns the stored 11-tuple for the exact ``(tx, ty)`` match at that
    level, or ``None`` if no such tile exists. The input is never modified or
    reordered.

    :raises TypeError: ``assessment`` is not a tuple or ``level``/``tx``/``ty``
        is not a non-bool int.
    :raises ValueError: ``level`` is out of range or the assessment's
        structure, ordering, duplicates, fields, cell bounds or finiteness are
        bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    for name, value in (("level", level), ("tx", tx), ("ty", ty)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(assessment):
        raise ValueError("level out of range")

    _validate_pyramid_assessment(assessment)

    for tile in assessment[level]:
        if tile[0] == tx and tile[1] == ty:
            return tile
    return None


def decode_tile_pyramid_stats(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_pyramid_stats`.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, top-level keys/types, level/tile
        shape, field types, ordering, duplicates, non-finite values,
        non-positive ``zsigma``, numeric formatting or canonical re-encoding
        does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"levels"}:
        raise ValueError("top-level value must be an object with only 'levels'")
    raw_levels = document["levels"]
    if not isinstance(raw_levels, list):
        raise ValueError("'levels' must be an array")

    levels: list[tuple] = []
    for raw_tiles in raw_levels:
        if not isinstance(raw_tiles, list):
            raise ValueError("each level must be an array of tiles")
        tiles: list[tuple] = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 11:
                raise ValueError("each tile must be an array of eleven values")
            tiles.append(tuple(raw_tile))
        levels.append(tuple(tiles))
    pyramid = tuple(levels)

    # Structural rules: tuple levels, field types/finiteness, positive
    # zsigma, (tx, ty) sort order and no duplicates.
    _validate_pyramid_stats(pyramid)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal statistic formatting, leading zeros,
    # -0, exponents and any other non-canonical spelling.
    if encode_tile_pyramid_stats(pyramid) != text:
        raise ValueError("JSON text is not the canonical pyramid-stats encoding")
    return pyramid


def merge_tile_pyramids(pyramids: tuple) -> tuple:
    """Merge tile pyramids produced by :func:`build_tile_pyramid`.

    ``pyramids`` must be a tuple of outer pyramids (it may be empty, in which
    case ``()`` is returned); every member must have the same number of levels
    (possibly zero). Each level must be a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples, with the
    first six fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite
    floats, and tiles strictly sorted by ``(tx, ty)`` without duplicates.

    Tiles are unioned per level and per ``(tx, ty)``. Tiles sharing a
    coordinate must also share their ``(ix0, iy0, ix1, iy1)`` bounds; the
    merged tile keeps the geometry of the first input at that coordinate,
    takes the minimum ``zmin`` and maximum ``zmax``, and sums ``count``.

    Returns a pyramid with the same levels in the same order, each level's
    tiles sorted by ``(tx, ty)`` (empty levels are ``()``). The result does
    not depend on the order of the inputs and the inputs are never modified.

    :raises TypeError: ``pyramids`` is not a tuple.
    :raises ValueError: a member is not a valid pyramid, the level counts
        differ, or same-coordinate tiles disagree on their cell bounds.
    """
    if not isinstance(pyramids, tuple):
        raise TypeError("pyramids must be a tuple")
    if not pyramids:
        return ()

    level_count = None
    for pyramid in pyramids:
        if not isinstance(pyramid, tuple):
            raise ValueError("each pyramid must be a tuple of levels")
        _validate_pyramid(pyramid)
        if level_count is None:
            level_count = len(pyramid)
        elif len(pyramid) != level_count:
            raise ValueError("all pyramids must have the same number of levels")

    merged_levels = []
    for level in range(level_count):
        # key -> [bounds, zmin, zmax, count]; bounds come from the first
        # pyramid contributing the coordinate.
        combined: dict[tuple[int, int], list] = {}
        for pyramid in pyramids:
            for tile in pyramid[level]:
                (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
                key = (tx, ty)
                bounds = (ix0, iy0, ix1, iy1)
                entry = combined.get(key)
                if entry is None:
                    combined[key] = [bounds, zmin, zmax, count]
                else:
                    if entry[0] != bounds:
                        raise ValueError(
                            "tiles with the same (tx, ty) must agree on "
                            "(ix0, iy0, ix1, iy1)"
                        )
                    if zmin < entry[1]:
                        entry[1] = zmin
                    if zmax > entry[2]:
                        entry[2] = zmax
                    entry[3] += count

        level_tiles = []
        for (tx, ty), (bounds, zmin, zmax, count) in sorted(combined.items()):
            ix0, iy0, ix1, iy1 = bounds
            level_tiles.append(
                (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)
            )
        merged_levels.append(tuple(level_tiles))

    return tuple(merged_levels)


def merge_tile_pyramid_stats(pyramids: tuple) -> tuple:
    """Merge tile pyramids with z statistics, combining inverse-variance stats.

    ``pyramids`` must be a tuple of outer pyramids as produced by
    :func:`build_tile_pyramid_stats` (it may be empty, in which case ``()`` is
    returned); every member must have the same number of levels (possibly
    zero). Each level must be a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    11-tuples, with the first six fields and ``count`` non-bool ints, the four
    statistics finite floats, ``zsigma`` positive, and tiles strictly sorted by
    ``(tx, ty)`` without duplicates.

    Tiles are unioned per level and per ``(tx, ty)``. Tiles sharing a
    coordinate must also share their ``(ix0, iy0, ix1, iy1)`` bounds; the
    merged tile keeps that geometry, takes the minimum ``zmin`` and maximum
    ``zmax``, and sums ``count``. The z statistics use weights
    ``w = 1 / zsigma ** 2``: ``zmean = sum(w * zmean) / sum(w)`` and
    ``zsigma = sqrt(1 / sum(w))``; the summations are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) with the summands accumulated in ascending Decimal
    order, so the results do not depend on input order. The four statistics
    are quantized to six decimal places as floats (negative zero normalized).

    Returns a pyramid with the same levels in the same order, each level's
    tiles sorted by ``(tx, ty)`` (empty levels are ``()``). The result does
    not depend on the order of the inputs and the inputs are never modified.

    :raises TypeError: ``pyramids`` is not a tuple.
    :raises ValueError: a member is not a valid pyramid, the level counts
        differ, or same-coordinate tiles disagree on their cell bounds.
    """
    if not isinstance(pyramids, tuple):
        raise TypeError("pyramids must be a tuple")
    if not pyramids:
        return ()

    level_count = None
    for pyramid in pyramids:
        if not isinstance(pyramid, tuple):
            raise ValueError("each pyramid must be a tuple of levels")
        _validate_pyramid_stats(pyramid)
        if level_count is None:
            level_count = len(pyramid)
        elif len(pyramid) != level_count:
            raise ValueError("all pyramids must have the same number of levels")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        merged_levels = []
        for level in range(level_count):
            # key -> [bounds, zmin, zmax, count, weights, weighted means];
            # bounds come from the first pyramid contributing the coordinate
            # and the per-tile Decimal summands are sorted before accumulation.
            combined: dict[tuple[int, int], list] = {}
            for pyramid in pyramids:
                for tile in pyramid[level]:
                    (tx, ty, ix0, iy0, ix1, iy1,
                     zmin, zmax, zmean, zsigma, count) = tile
                    key = (tx, ty)
                    bounds = (ix0, iy0, ix1, iy1)
                    dzsigma = Decimal(str(zsigma))
                    weight = Decimal(1) / (dzsigma * dzsigma)
                    weighted_mean = weight * Decimal(str(zmean))
                    entry = combined.get(key)
                    if entry is None:
                        combined[key] = [bounds, zmin, zmax, count,
                                         [weight], [weighted_mean]]
                    else:
                        if entry[0] != bounds:
                            raise ValueError(
                                "tiles with the same (tx, ty) must agree on "
                                "(ix0, iy0, ix1, iy1)"
                            )
                        if zmin < entry[1]:
                            entry[1] = zmin
                        if zmax > entry[2]:
                            entry[2] = zmax
                        entry[3] += count
                        entry[4].append(weight)
                        entry[5].append(weighted_mean)

            level_tiles = []
            for (tx, ty), entry in sorted(combined.items()):
                (bounds, zmin, zmax, count,
                 weights, weighted_means) = entry
                sum_w = _sorted_sum(weights)
                merged_zmean = _sorted_sum(weighted_means) / sum_w
                merged_zsigma = (Decimal(1) / sum_w).sqrt()
                ix0, iy0, ix1, iy1 = bounds
                level_tiles.append((
                    tx, ty, ix0, iy0, ix1, iy1,
                    _quantize(Decimal(str(zmin))),
                    _quantize(Decimal(str(zmax))),
                    _quantize(merged_zmean),
                    _quantize(merged_zsigma),
                    count,
                ))
            merged_levels.append(tuple(level_tiles))

    return tuple(merged_levels)


def update_tile_pyramid_stats(pyramid: tuple,
                              points: Iterable[tuple | list],
                              cell_size: int | float = 1.0,
                              tile_cells: int = 256,
                              levels: int = 3) -> tuple:
    """Add points to a tile pyramid with inverse-variance z statistics.

    Equivalent to
    ``merge_tile_pyramid_stats((pyramid, build_tile_pyramid_stats(points, cell_size, tile_cells, levels)))``
    but ``points`` is consumed in a single pass: the new points are aggregated
    into a pyramid exactly as in :func:`build_tile_pyramid_stats` and the
    combination is then performed as in :func:`merge_tile_pyramid_stats`.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid_stats` with exactly ``levels`` levels: each level
    is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)``
    11-tuples sorted lexicographically by ``(tx, ty)``, where the first six
    fields and ``count`` are non-bool ints, the four statistics are finite
    floats and ``zsigma`` is positive.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list of
    finite non-bool ints/floats with ``sigma > 0``; cell indices are
    ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)`` and at level
    ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``.

    Per-tile statistics use weights ``w = 1 / sigma ** 2`` for the new points
    and ``w = 1 / zsigma ** 2`` for the existing tiles:
    ``zmean = sum(w * z) / sum(w)`` and ``zsigma = sqrt(1 / sum(w))``; the
    summations are Decimal sums (precision 50, ``ROUND_HALF_EVEN``) of the
    terms sorted in ascending order, so the results do not depend on input
    order.

    Returns the updated pyramid (a new tuple; the input pyramid and its tiles
    are never modified), with the same structure, ordering and fields as
    :func:`build_tile_pyramid_stats`. An empty ``points`` iterable returns the
    original ``pyramid`` unchanged.

    :raises TypeError: ``pyramid`` is not a tuple, ``points`` is not iterable,
        ``cell_size``/``tile_cells``/``levels`` have the wrong type, or a
        point's container/length/fields have the wrong type.
    :raises ValueError: a parameter is non-finite or non-positive, the number
        of levels does not match ``pyramid``, the pyramid's structure,
        ordering, duplicates or fields are bad, a point field is non-finite or
        ``sigma <= 0``, or same-coordinate tiles disagree on their cell
        bounds.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
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
    _validate_pyramid_stats(pyramid)
    if len(pyramid) != levels:
        raise ValueError("all pyramids must have the same number of levels")

    # Single pass over ``points``: iter() is called exactly once, len() is
    # never called on it, and every level is aggregated simultaneously. The
    # new points are accumulated exactly as in build_tile_pyramid_stats and
    # the resulting pyramid is then combined with ``pyramid`` via
    # merge_tile_pyramid_stats, guaranteeing identical results.
    point_iter = iter(points)

    widths = [tile_cells * 2 ** level for level in range(levels)]
    # Each entry is [zmin, zmax, count, weights, weighted z values]; the
    # per-point Decimal summands are kept and sorted before accumulation.
    tiles_per_level: list[dict[tuple[int, int], list]] = [
        {} for _ in range(levels)
    ]

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        dcell = Decimal(str(cell_size))
        if dcell <= 0:
            raise ValueError("cell_size must be positive")

        saw_point = False
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

            weight = Decimal(1) / (dsigma * dsigma)
            weighted_z = weight * dz

            ix = int((dx / dcell).to_integral_value(rounding=ROUND_FLOOR))
            iy = int((dy / dcell).to_integral_value(rounding=ROUND_FLOOR))

            for level, width in enumerate(widths):
                key = (ix // width, iy // width)
                tiles = tiles_per_level[level]
                entry = tiles.get(key)
                if entry is None:
                    tiles[key] = [dz, dz, 1, [weight], [weighted_z]]
                else:
                    if dz < entry[0]:
                        entry[0] = dz
                    if dz > entry[1]:
                        entry[1] = dz
                    entry[2] += 1
                    entry[3].append(weight)
                    entry[4].append(weighted_z)
            saw_point = True

        if not saw_point:
            return pyramid

        new_pyramid = []
        for width, tiles in zip(widths, tiles_per_level):
            level_tiles = []
            for (tx, ty), entry in tiles.items():
                zmin, zmax, count, weights, weighted_zs = entry
                sum_w = _sorted_sum(weights)
                zmean = _sorted_sum(weighted_zs) / sum_w
                zsigma = (Decimal(1) / sum_w).sqrt()
                ix0 = tx * width
                iy0 = ty * width
                level_tiles.append((
                    tx, ty,
                    ix0, iy0,
                    ix0 + width - 1, iy0 + width - 1,
                    _quantize(zmin), _quantize(zmax),
                    _quantize(zmean), _quantize(zsigma),
                    count,
                ))
            level_tiles.sort(key=lambda item: (item[0], item[1]))
            new_pyramid.append(tuple(level_tiles))

    return merge_tile_pyramid_stats((pyramid, tuple(new_pyramid)))
