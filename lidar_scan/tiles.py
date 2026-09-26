"""Tile-level indexing of LiDAR scan points."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from decimal import (Decimal, InvalidOperation, ROUND_FLOOR,
                     ROUND_HALF_EVEN, localcontext)

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_MICRO = Decimal(10) ** 6
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


def window_tiles(points: Iterable[tuple | list],
                 windows: tuple,
                 cell_size: int | float = 1.0,
                 tile_cells: int = 256,
                 levels: int = 3) -> tuple:
    """Aggregate points into pyramid tiles intersecting cell-index windows.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list of
    finite non-bool ints/floats with ``sigma > 0``. Cell indices are
    ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)``. At level
    ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``; the per-tile summary holds the minimum/maximum ``z`` and
    the point count.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < levels``.

    ``points`` is consumed in a single pass (``iter()`` is called exactly
    once) while every level is aggregated simultaneously. For each window, a
    tile matches when its closed cell-index intervals intersect the window:
    ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and tile.iy1 >= iy_min and
    tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)``
    9-tuples sorted lexicographically by ``(tx, ty)``; a window with no
    matching tile gets an empty ``tiles`` tuple and an empty ``windows``
    tuple returns ``()``. ``zmin``/``zmax`` are Decimal computations
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) quantized to six
    decimal places as floats (negative zero normalized).

    :raises TypeError: ``points`` is not iterable, ``windows`` is not a tuple,
        a window's container/length/field types are bad, a point's
        container/length/fields have the wrong type, or
        ``cell_size``/``tile_cells``/``levels`` have the wrong type.
    :raises ValueError: a parameter or point field is non-finite,
        ``cell_size``/``tile_cells``/``levels``/``sigma`` is non-positive,
        ``level`` is out of range or the window bounds are inverted.
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

    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")
    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")
    for window in windows:
        level, ix_min, iy_min, ix_max, iy_max = window
        if level < 0 or level >= levels:
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")

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

        results = []
        for level, ix_min, iy_min, ix_max, iy_max in windows:
            width = widths[level]
            matched = []
            for (tx, ty), (zmin, zmax, count) in tiles_per_level[level].items():
                ix0 = tx * width
                iy0 = ty * width
                ix1 = ix0 + width - 1
                iy1 = iy0 + width - 1
                if (ix1 >= ix_min and ix0 <= ix_max
                        and iy1 >= iy_min and iy0 <= iy_max):
                    matched.append((
                        tx, ty,
                        ix0, iy0, ix1, iy1,
                        _quantize(zmin), _quantize(zmax),
                        count,
                    ))
            matched.sort(key=lambda item: (item[0], item[1]))
            results.append((level, ix_min, iy_min, ix_max, iy_max,
                            tuple(matched)))

    return tuple(results)


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


def query_tile_pyramid_assessment(assessment: tuple, level: int,
                                  tx: int, ty: int) -> tuple | None:
    """Look up the tile ``(tx, ty)`` at ``level`` of a pyramid assessment.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_stats`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` 11-tuples sorted strictly by ``(tx, ty)`` with no
    duplicates, where the first six fields and ``count_delta`` are non-bool
    ints with ``ix0 <= ix1`` and ``iy0 <= iy1``, and the four metrics are
    finite floats with ``combined_sigma > 0``. ``level``, ``tx`` and ``ty``
    must be non-bool ints and ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    Returns the stored 11-tuple for the exact ``(tx, ty)`` match at that
    level, or ``None`` if no such tile exists. The input is never modified or
    reordered.

    :raises TypeError: ``assessment`` is not a tuple or ``level``/``tx``/
        ``ty`` is not a non-bool int.
    :raises ValueError: ``level`` is out of range or the assessment's
        structure, ordering, duplicates, fields, cell bounds or finiteness
        are bad.
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


def query_tile_pyramid_delta(assessment: tuple, level: int,
                             tx: int, ty: int) -> tuple | None:
    """Look up the tile ``(tx, ty)`` at ``level`` of a pyramid delta assessment.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_deltas`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields
    and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1``,
    and ``dzmin``/``dzmax`` are finite floats. ``level``, ``tx`` and ``ty``
    must be non-bool ints and ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    Returns the stored 9-tuple for the exact ``(tx, ty)`` match at that
    level, or ``None`` if no such tile exists. The input is never modified or
    reordered and repeated calls return identical results.

    :raises TypeError: ``assessment`` is not a tuple or ``level``/``tx``/
        ``ty`` is not a non-bool int.
    :raises ValueError: ``level`` is out of range or the assessment's
        structure, ordering, duplicates, fields, cell bounds or finiteness
        are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    for name, value in (("level", level), ("tx", tx), ("ty", ty)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int")
    if level < 0 or level >= len(assessment):
        raise ValueError("level out of range")

    _validate_pyramid_deltas(assessment)

    for tile in assessment[level]:
        if tile[0] == tx and tile[1] == ty:
            return tile
    return None


def query_tile_pyramid_delta_windows(assessment: tuple, windows: tuple) -> tuple:
    """Select delta tiles intersecting cell-index windows across levels.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_deltas`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields
    and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1``,
    and ``dzmin``/``dzmax`` are finite floats.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    For each window, a tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the stored 9-tuples at that level, preserving the level's
    existing order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The input is never
    modified and repeated calls return identical results.

    :raises TypeError: ``assessment``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: ``level`` is out of range, the window bounds are
        inverted, or the assessment's structure, ordering, duplicates, fields,
        cell bounds or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid_deltas(assessment)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(assessment):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in assessment[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def aggregate_tile_pyramid_delta_windows(assessment: tuple,
                                         windows: tuple) -> tuple:
    """Aggregate delta tiles intersecting cell-index windows across levels.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_deltas`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields
    and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1``,
    and ``dzmin``/``dzmax`` are finite floats.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    For each window, a tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``. Matching tiles are aggregated
    in the level's existing order into
    ``summary = (min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin`` is the minimum tile ``dzmin``, ``max_dzmax`` the maximum
    tile ``dzmax`` (the extrema compared via ``Decimal(str(v))`` at precision
    50 with ``ROUND_HALF_EVEN`` and quantized to six decimal places as floats,
    negative zero normalized), ``sum_dcount`` is the exact int sum of the
    tile ``dcount`` values and ``match_count`` is the int number of matching
    tiles.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` tuples; a window with
    no matching tile gets ``summary = None`` and an empty ``windows`` tuple
    returns ``()``. The input is never modified and repeated calls return
    identical results.

    :raises TypeError: ``assessment``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: ``level`` is out of range, the window bounds are
        inverted, or the assessment's structure, ordering, duplicates, fields,
        cell bounds or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid_deltas(assessment)

    results = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, ix_min, iy_min, ix_max, iy_max in windows:
            if level < 0 or level >= len(assessment):
                raise ValueError("level out of range")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("window bounds must satisfy ix_min <= ix_max "
                                 "and iy_min <= iy_max")

            min_dzmin = None
            max_dzmax = None
            sum_dcount = 0
            match_count = 0
            for tile in assessment[level]:
                if not (tile[4] >= ix_min and tile[2] <= ix_max
                        and tile[5] >= iy_min and tile[3] <= iy_max):
                    continue
                ddzmin = Decimal(str(tile[6]))
                ddzmax = Decimal(str(tile[7]))
                if min_dzmin is None or ddzmin < min_dzmin:
                    min_dzmin = ddzmin
                if max_dzmax is None or ddzmax > max_dzmax:
                    max_dzmax = ddzmax
                sum_dcount += tile[8]
                match_count += 1

            if match_count:
                summary = (_quantize(min_dzmin), _quantize(max_dzmax),
                           sum_dcount, match_count)
            else:
                summary = None
            results.append((level, ix_min, iy_min, ix_max, iy_max, summary))

    return tuple(results)


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


def query_tile_pyramid_windows(pyramid: tuple, windows: tuple) -> tuple:
    """Select tiles intersecting cell-index windows across pyramid levels.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields
    and ``count`` are non-bool ints, ``zmin``/``zmax`` are finite floats and
    ``ix0 <= ix1``/``iy0 <= iy1``.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(pyramid)``.

    For each window, a tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the stored 9-tuples at that level, preserving the level's
    existing order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The input is never
    modified and repeated calls return identical results.

    :raises TypeError: ``pyramid``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: ``level`` is out of range, the window bounds are
        inverted, or the pyramid's structure, ordering, duplicates, fields
        or cell bounds are bad.
    """
    if not isinstance(pyramid, tuple):
        raise TypeError("pyramid must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid(pyramid)

    for level_tiles in pyramid:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(pyramid):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in pyramid[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


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


def assess_tile_pyramid_deltas(estimate: tuple, reference: tuple) -> tuple:
    """Compute per-tile z/count deltas of an estimated pyramid vs a reference.

    Both inputs are outer tuples as produced by :func:`build_tile_pyramid`
    with the same number of levels: each level is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` sorted strictly by
    ``(tx, ty)`` with no duplicates, where the first six fields and ``count``
    are non-bool ints, ``zmin``/``zmax`` are finite floats with
    ``zmin <= zmax`` and the bounds satisfy ``ix0 <= ix1`` and
    ``iy0 <= iy1``. At every level the two inputs must contain the same
    ``(tx, ty)`` coordinates with identical ``(ix0, iy0, ix1, iy1)`` bounds
    (their ``count`` values may differ); otherwise ``ValueError`` is raised.

    Returns a tuple, in level order, of tuples (in each level's original
    order) of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples where
    ``dzmin = estimate.zmin - reference.zmin``,
    ``dzmax = estimate.zmax - reference.zmax`` and
    ``dcount = estimate.count - reference.count``. The two float deltas are
    Decimal computations (precision 50, ``ROUND_HALF_EVEN``, each input
    converted via ``Decimal(str(v))``) quantized to six decimal places as
    floats (negative zero normalized); ``dcount`` is an exact int. Empty
    levels are preserved; two empty inputs return ``()``. The inputs are never
    modified and the result does not depend on the order in which matching
    tiles are presented beyond the levels' stored order.

    :raises TypeError: ``estimate`` or ``reference`` is not a tuple.
    :raises ValueError: the level counts differ or the structure, ordering,
        duplicates, fields, cell bounds, z bounds, coordinate sets or shared
        cell bounds of either input are bad.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")

    _validate_pyramid(estimate)
    _validate_pyramid(reference)

    for pyramid in (estimate, reference):
        for level_tiles in pyramid:
            for tile in level_tiles:
                if tile[4] < tile[2] or tile[5] < tile[3]:
                    raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                     "iy0 <= iy1")
                if tile[6] > tile[7]:
                    raise ValueError("zmin must be <= zmax")

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
                dzmin = Decimal(str(est_tile[6])) - Decimal(str(ref_tile[6]))
                dzmax = Decimal(str(est_tile[7])) - Decimal(str(ref_tile[7]))
                level_result.append((
                    est_tile[0], est_tile[1],
                    est_tile[2], est_tile[3], est_tile[4], est_tile[5],
                    _quantize(dzmin), _quantize(dzmax),
                    est_tile[8] - ref_tile[8],
                ))
            pyramid.append(tuple(level_result))

    return tuple(pyramid)


def assess_tile_pyramid_windows(estimate: tuple, reference: tuple,
                                windows: tuple) -> tuple:
    """Assess matched tiles inside cell-index windows across pyramid levels.

    ``estimate`` and ``reference`` must be outer tuples as produced by
    :func:`build_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints, the bounds satisfy ``ix0 <= ix1`` and
    ``iy0 <= iy1``, and the four statistics are finite floats with
    ``zsigma > 0``.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(estimate)`` and ``0 <= level < len(reference)``.

    For each window, tiles are matched when the two inputs have a tile with the
    same ``(tx, ty)`` coordinates and identical ``(ix0, iy0, ix1, iy1)``
    bounds whose closed cell-index intervals intersect the window
    (``tile.ix1 >= ix_min and tile.ix0 <= ix_max and tile.iy1 >= iy_min and
    tile.iy0 <= iy_max``). Tiles present on only one side are not matches; a
    window with no match gets an empty ``tiles`` tuple. Matched tiles are
    returned in the estimate level's order as
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` where ``bias = estimate.zmean - reference.zmean``,
    ``abs_error = abs(bias)``,
    ``combined_sigma = sqrt(estimate.zsigma**2 + reference.zsigma**2)``,
    ``z_score = bias / combined_sigma`` and
    ``count_delta = estimate.count - reference.count``. The four metrics are
    Decimal computations (precision 50, ``ROUND_HALF_EVEN``, each input
    converted via ``Decimal(str(v))``) quantized to six decimal places as
    floats (negative zero normalized).

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples. The inputs are
    never modified.

    :raises TypeError: ``estimate``/``reference``/``windows`` is not a tuple or
        a window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds or finiteness of either input are bad, same-coordinate tiles
        disagree on their cell bounds, ``level`` is out of range or the
        window bounds are inverted.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid_stats(estimate)
    _validate_pyramid_stats(reference)

    for pyramid in (estimate, reference):
        for level_tiles in pyramid:
            for tile in level_tiles:
                if tile[4] < tile[2] or tile[5] < tile[3]:
                    raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                     "iy0 <= iy1")

    results = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for window in windows:
            level, ix_min, iy_min, ix_max, iy_max = window
            if (level < 0 or level >= len(estimate)
                    or level >= len(reference)):
                raise ValueError("level out of range")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("window bounds must satisfy ix_min <= ix_max "
                                 "and iy_min <= iy_max")

            ref_by_key = {}
            for ref_tile in reference[level]:
                if (ref_tile[4] >= ix_min and ref_tile[2] <= ix_max
                        and ref_tile[5] >= iy_min and ref_tile[3] <= iy_max):
                    ref_by_key[(ref_tile[0], ref_tile[1])] = ref_tile

            matched = []
            for est_tile in estimate[level]:
                if not (est_tile[4] >= ix_min and est_tile[2] <= ix_max
                        and est_tile[5] >= iy_min and est_tile[3] <= iy_max):
                    continue
                ref_tile = ref_by_key.get((est_tile[0], est_tile[1]))
                if ref_tile is None:
                    continue
                if est_tile[2:6] != ref_tile[2:6]:
                    raise ValueError(
                        "tiles with the same (tx, ty) must agree on "
                        "(ix0, iy0, ix1, iy1)"
                    )

                est_zmean = Decimal(str(est_tile[8]))
                ref_zmean = Decimal(str(ref_tile[8]))
                est_zsigma = Decimal(str(est_tile[9]))
                ref_zsigma = Decimal(str(ref_tile[9]))
                bias = est_zmean - ref_zmean
                combined = (est_zsigma * est_zsigma
                            + ref_zsigma * ref_zsigma).sqrt()
                matched.append((
                    est_tile[0], est_tile[1],
                    est_tile[2], est_tile[3], est_tile[4], est_tile[5],
                    _quantize(bias), _quantize(abs(bias)),
                    _quantize(combined), _quantize(bias / combined),
                    est_tile[10] - ref_tile[10],
                ))

            results.append((level, ix_min, iy_min, ix_max, iy_max,
                            tuple(matched)))

    return tuple(results)


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


def summarize_tile_pyramid_assessment(assessment: tuple) -> tuple:
    """Summarize each level of a :func:`assess_tile_pyramid_stats` result.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_stats`: each level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, bias, abs_error, combined_sigma, z_score,
    count_delta)`` with ``(tx, ty)`` strictly increasing and with no
    duplicates, where the first six fields and ``count_delta`` are non-bool
    ints satisfying ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``bias``,
    ``abs_error``, ``combined_sigma`` and ``z_score`` are finite floats with
    ``combined_sigma > 0`` and ``abs_error >= 0``.

    Returns a tuple, in level order, of
    ``(level, bias_min, bias_max, abs_error_mean, z_score_rms,
    count_delta_sum)`` tuples where ``level`` is the level index,
    ``bias_min``/``bias_max`` are the level's minimum and maximum ``bias``,
    ``abs_error_mean`` is the mean ``abs_error``,
    ``z_score_rms`` is ``sqrt(mean(z_score**2))`` and ``count_delta_sum`` is
    the exact int sum of the ``count_delta`` values. The four float metrics
    are Decimal computations (each input converted via ``Decimal(str(v))``,
    precision 50, ``ROUND_HALF_EVEN``) with summation terms added in ascending
    Decimal order (so the results do not depend on tile presentation order
    beyond the level each tile belongs to), quantized to six decimal places as
    floats (negative zero normalized). An empty level summarizes as
    ``(level, None, None, None, None, 0)``; an empty assessment ``()`` returns
    ``()``. The input is never modified.

    :raises TypeError: ``assessment`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, finiteness, ``combined_sigma`` positivity or ``abs_error``
        non-negativity are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if len(assessment) == 0:
        return ()

    _validate_pyramid_assessment(assessment)
    for level_tiles in assessment:
        for tile in level_tiles:
            if tile[7] < 0:
                raise ValueError("abs_error must be non-negative")

    summaries = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, level_tiles in enumerate(assessment):
            if not level_tiles:
                summaries.append((level, None, None, None, None, 0))
                continue
            biases = [Decimal(str(tile[6])) for tile in level_tiles]
            abs_error_terms = [Decimal(str(tile[7])) for tile in level_tiles]
            z_score_terms = [Decimal(str(tile[9])) for tile in level_tiles]
            count_terms = [Decimal(tile[10]) for tile in level_tiles]
            z_squared_terms = [z * z for z in z_score_terms]
            tile_count = Decimal(len(level_tiles))

            abs_error_mean = _sorted_sum(abs_error_terms) / tile_count
            z_score_rms = (_sorted_sum(z_squared_terms)
                           / tile_count).sqrt()
            summaries.append((
                level,
                _quantize(min(biases)),
                _quantize(max(biases)),
                _quantize(abs_error_mean),
                _quantize(z_score_rms),
                int(_sorted_sum(count_terms)),
            ))

    return tuple(summaries)


def summarize_tile_pyramid_windows(estimate: tuple, reference: tuple,
                                   windows: tuple) -> tuple:
    """Summarize matched tiles inside cell-index windows across pyramid levels.

    ``estimate`` and ``reference`` must be outer tuples as produced by
    :func:`build_tile_pyramid_stats` with the same number of levels: each
    level is a tuple of 11-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, zmean, zsigma, count)`` sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints, the bounds satisfy ``ix0 <= ix1`` and
    ``iy0 <= iy1``, and the four statistics are finite floats with
    ``zsigma > 0``. At every level the two inputs must contain the same
    ``(tx, ty)`` coordinates with identical ``(ix0, iy0, ix1, iy1)`` bounds
    (their ``zmean``/``zsigma``/``count`` values may differ).

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(estimate)`` and ``0 <= level < len(reference)``.

    For each window, tiles are matched when the two inputs have a tile with the
    same ``(tx, ty)`` coordinates whose closed cell-index intervals intersect
    the window (``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``); tiles present on only one
    side are not matches. For every matched tile,
    ``b = estimate.zmean - reference.zmean``, ``a = abs(b)``,
    ``q = b / sqrt(estimate.zsigma**2 + reference.zsigma**2)`` and
    ``d = estimate.count - reference.count``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` tuples. A window with
    no matched tile gets ``summary = None``; otherwise ``summary`` is the
    5-tuple ``(min(b), max(b), mean(a), sqrt(mean(q**2)), sum(d))`` where the
    first four values are Decimal computations (each input converted via
    ``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) with the ``a``
    terms and the ``q**2`` terms added in ascending Decimal order (so the
    results do not depend on tile presentation order), quantized to six
    decimal places as floats (negative zero normalized), and ``sum(d)`` is
    the exact int sum of the count deltas. An empty ``windows`` tuple returns
    ``()``. The inputs are never modified.

    :raises TypeError: ``estimate``/``reference``/``windows`` is not a tuple or
        a window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds or finiteness of either input are bad, the level counts or
        tile coordinates differ, same-coordinate tiles disagree on their cell
        bounds, ``level`` is out of range or the window bounds are inverted.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

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
            raise ValueError(
                "estimate and reference must contain the same (tx, ty) tile "
                "coordinates at every level"
            )
        for est_tile, ref_tile in zip(est_level, ref_level):
            if est_tile[0:2] != ref_tile[0:2]:
                raise ValueError(
                    "estimate and reference must contain the same (tx, ty) "
                    "tile coordinates at every level"
                )
            if est_tile[2:6] != ref_tile[2:6]:
                raise ValueError(
                    "tiles with the same (tx, ty) must agree on "
                    "(ix0, iy0, ix1, iy1)"
                )

    results = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for window in windows:
            level, ix_min, iy_min, ix_max, iy_max = window
            if (level < 0 or level >= len(estimate)
                    or level >= len(reference)):
                raise ValueError("level out of range")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("window bounds must satisfy ix_min <= ix_max "
                                 "and iy_min <= iy_max")

            ref_by_key = {}
            for ref_tile in reference[level]:
                if (ref_tile[4] >= ix_min and ref_tile[2] <= ix_max
                        and ref_tile[5] >= iy_min and ref_tile[3] <= iy_max):
                    ref_by_key[(ref_tile[0], ref_tile[1])] = ref_tile

            biases = []
            abs_terms = []
            q_squared_terms = []
            count_delta_sum = 0
            for est_tile in estimate[level]:
                if not (est_tile[4] >= ix_min and est_tile[2] <= ix_max
                        and est_tile[5] >= iy_min and est_tile[3] <= iy_max):
                    continue
                ref_tile = ref_by_key.get((est_tile[0], est_tile[1]))
                if ref_tile is None:
                    continue

                est_zmean = Decimal(str(est_tile[8]))
                ref_zmean = Decimal(str(ref_tile[8]))
                est_zsigma = Decimal(str(est_tile[9]))
                ref_zsigma = Decimal(str(ref_tile[9]))
                bias = est_zmean - ref_zmean
                combined = (est_zsigma * est_zsigma
                            + ref_zsigma * ref_zsigma).sqrt()
                z_score = bias / combined
                biases.append(bias)
                abs_terms.append(abs(bias))
                q_squared_terms.append(z_score * z_score)
                count_delta_sum += est_tile[10] - ref_tile[10]

            if not biases:
                summary = None
            else:
                match_count = Decimal(len(biases))
                abs_mean = _sorted_sum(abs_terms) / match_count
                z_score_rms = (_sorted_sum(q_squared_terms)
                               / match_count).sqrt()
                summary = (
                    _quantize(min(biases)),
                    _quantize(max(biases)),
                    _quantize(abs_mean),
                    _quantize(z_score_rms),
                    count_delta_sum,
                )
            results.append((level, ix_min, iy_min, ix_max, iy_max, summary))

    return tuple(results)


def _validate_pyramid_deltas(assessment) -> None:
    """Validate the outer tuple returned by :func:`assess_tile_pyramid_deltas`.

    Every level must be a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)``: the first six
    fields and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1``; ``dzmin``/``dzmax`` are finite floats; and the tiles are
    sorted strictly by ``(tx, ty)`` with no duplicates.
    """
    for level_tiles in assessment:
        if not isinstance(level_tiles, tuple):
            raise ValueError("each assessment level must be a tuple")
        prev_key = None
        for tile in level_tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "dcount must be non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("dzmin and dzmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("each assessment level must be sorted by "
                                 "(tx, ty) with no duplicate coordinates")
            prev_key = key


def encode_tile_pyramid_deltas(assessment: tuple) -> str:
    """Serialize the result of :func:`assess_tile_pyramid_deltas`.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_tile_pyramid_deltas`: each level is a tuple of strict
    9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` with tiles sorted
    strictly by ``(tx, ty)`` and no duplicates, where the first six fields and
    ``dcount`` are non-bool ints satisfying ``ix0 <= ix1`` and
    ``iy0 <= iy1``, and ``dzmin``/``dzmax`` are finite floats.

    Returns a canonical compact JSON string whose sole top-level key is
    ``levels``: an array (in level order) of arrays of tile arrays in the
    tiles' stored order. Integers are decimal; ``dzmin``/``dzmax`` use exactly
    six decimal places (negative zero written as ``0.000000``). The output has
    no whitespace, ASCII is not escaped and ``NaN``/``Infinity`` never appear.
    An empty assessment encodes as ``{"levels":[]}``. The input is never
    modified.

    :raises TypeError: ``assessment`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields,
        finiteness or cell bounds are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    _validate_pyramid_deltas(assessment)

    parts = ['{"levels":[']
    for level_index, level_tiles in enumerate(assessment):
        if level_index:
            parts.append(",")
        parts.append("[")
        for tile_index, tile in enumerate(level_tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1,
             dzmin, dzmax, dcount) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(dzmin),
                                   _format_z(dzmax), str(dcount))))
            parts.append("]")
        parts.append("]")
    parts.append(']}')
    return "".join(parts)


def decode_tile_pyramid_deltas(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_pyramid_deltas`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``levels``; ``levels`` an array (in level order) of
    arrays of strict 9-item tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount]`` whose first six
    fields and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1``, whose ``dzmin``/``dzmax`` are finite floats, and whose
    ``(tx, ty)`` coordinates are strictly increasing with no duplicates within
    each level, and whose spelling is exactly canonical (integers in decimal,
    deltas with six decimals, negative zero as ``0.000000``, no whitespace or
    extra keys, no ``NaN``/``Infinity``).

    Returns the outer level-order tuple of tuples of 9-tuples in the
    document's order; the empty document ``{"levels":[]}`` returns ``()``.
    The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, top-level keys/types, level/tile
        shape, field types, bounds, ordering, duplicates, non-finite values,
        numeric formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or tuple(document) != ("levels",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'levels'")
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
    assessment = tuple(levels)

    # Structural rules: tuple levels, field types/finiteness, cell bounds,
    # (tx, ty) sort order and no duplicates.
    _validate_pyramid_deltas(assessment)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal delta formatting, leading zeros,
    # -0, exponents and any other non-canonical spelling.
    if encode_tile_pyramid_deltas(assessment) != text:
        raise ValueError("JSON text is not the canonical pyramid-deltas "
                         "encoding")
    return assessment


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


def merge_tile_pyramid_windows(pyramids: tuple, windows: tuple) -> tuple:
    """Merge tile pyramids and select tiles intersecting cell-index windows.

    Equivalent to
    ``query_tile_pyramid_windows(merge_tile_pyramids(pyramids), windows)``.

    ``pyramids`` must be a tuple of outer pyramids as produced by
    :func:`build_tile_pyramid`; every member must have the same number of
    levels. Each level must be a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples, with the
    first six fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite
    floats, and tiles strictly sorted by ``(tx, ty)`` without duplicates.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(merged)`` where ``merged`` is the merged pyramid.

    Tiles are unioned per level and per ``(tx, ty)``. Tiles sharing a
    coordinate must also share their ``(ix0, iy0, ix1, iy1)`` bounds; the
    merged tile keeps the geometry of the first input at that coordinate,
    takes the minimum ``zmin`` and maximum ``zmax``, and sums ``count``. For
    each window, a merged tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the merged pyramid's 9-tuples at that level, sorted
    lexicographically by ``(tx, ty)``; a window with no matching tile gets an
    empty ``tiles`` tuple. An empty ``pyramids`` tuple is only valid with an
    empty ``windows`` tuple and returns ``()``. The result does not depend on
    the order of the inputs and the inputs are never modified.

    :raises TypeError: ``pyramids``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: a member is not a valid pyramid, the level counts
        differ, same-coordinate tiles disagree on their cell bounds, ``level``
        is out of range or the window bounds are inverted.
    """
    if not isinstance(pyramids, tuple):
        raise TypeError("pyramids must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")
    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    merged = merge_tile_pyramids(pyramids)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(merged):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in merged[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def merge_tile_pyramid_deltas(assessments: tuple) -> tuple:
    """Merge pyramid delta assessments produced by :func:`assess_tile_pyramid_deltas`.

    ``assessments`` must be a tuple of outer assessments (it may be empty, in
    which case ``()`` is returned); every member must have the same number of
    levels (possibly zero). Each level must be a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples, with the
    first six fields and ``dcount`` non-bool ints satisfying ``ix0 <= ix1``
    and ``iy0 <= iy1``, ``dzmin``/``dzmax`` finite floats, and tiles strictly
    sorted by ``(tx, ty)`` without duplicates.

    Tiles are unioned per level and per ``(tx, ty)``. Tiles sharing a
    coordinate must also share their ``(ix0, iy0, ix1, iy1)`` bounds; the
    merged tile keeps that geometry, sums the ``dzmin``/``dzmax`` deltas of
    the assessments that contain the coordinate (a missing tile contributes
    zero) and sums ``dcount`` exactly. The float sums are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) with the summands accumulated in ascending Decimal
    order, quantized to six decimal places as floats (negative zero
    normalized).

    Returns an assessment with the same levels in the same order, each level's
    tiles sorted by ``(tx, ty)`` (empty levels are ``()``). The result does
    not depend on the order of the inputs and the inputs are never modified.

    :raises TypeError: ``assessments`` is not a tuple.
    :raises ValueError: a member is not a valid assessment, the level counts
        differ, or same-coordinate tiles disagree on their cell bounds.
    """
    if not isinstance(assessments, tuple):
        raise TypeError("assessments must be a tuple")
    if not assessments:
        return ()

    level_count = None
    for assessment in assessments:
        if not isinstance(assessment, tuple):
            raise ValueError("each assessment must be a tuple of levels")
        _validate_pyramid_deltas(assessment)
        if level_count is None:
            level_count = len(assessment)
        elif len(assessment) != level_count:
            raise ValueError("all assessments must have the same number of "
                             "levels")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        merged_levels = []
        for level in range(level_count):
            # key -> [bounds, dzmin terms, dzmax terms, dcount]; bounds come
            # from the first assessment contributing the coordinate and the
            # per-tile Decimal summands are sorted before accumulation.
            combined: dict[tuple[int, int], list] = {}
            for assessment in assessments:
                for tile in assessment[level]:
                    (tx, ty, ix0, iy0, ix1, iy1,
                     dzmin, dzmax, dcount) = tile
                    key = (tx, ty)
                    bounds = (ix0, iy0, ix1, iy1)
                    entry = combined.get(key)
                    if entry is None:
                        combined[key] = [bounds, [Decimal(str(dzmin))],
                                         [Decimal(str(dzmax))], dcount]
                    else:
                        if entry[0] != bounds:
                            raise ValueError(
                                "tiles with the same (tx, ty) must agree on "
                                "(ix0, iy0, ix1, iy1)"
                            )
                        entry[1].append(Decimal(str(dzmin)))
                        entry[2].append(Decimal(str(dzmax)))
                        entry[3] += dcount

            level_tiles = []
            for (tx, ty), entry in sorted(combined.items()):
                bounds, dzmin_terms, dzmax_terms, dcount = entry
                ix0, iy0, ix1, iy1 = bounds
                level_tiles.append((
                    tx, ty, ix0, iy0, ix1, iy1,
                    _quantize(_sorted_sum(dzmin_terms)),
                    _quantize(_sorted_sum(dzmax_terms)),
                    dcount,
                ))
            merged_levels.append(tuple(level_tiles))

    return tuple(merged_levels)


def merge_tile_pyramid_delta_windows(assessments: tuple, windows: tuple) -> tuple:
    """Merge pyramid delta assessments and select tiles intersecting windows.

    Equivalent to
    ``query_tile_pyramid_delta_windows(merge_tile_pyramid_deltas(assessments), windows)``.

    ``assessments`` must be a tuple of outer assessments as produced by
    :func:`assess_tile_pyramid_deltas`; every member must have the same number
    of levels (possibly zero). Each level must be a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples, with the
    first six fields and ``dcount`` non-bool ints satisfying ``ix0 <= ix1``
    and ``iy0 <= iy1``, ``dzmin``/``dzmax`` finite floats, and tiles strictly
    sorted by ``(tx, ty)`` without duplicates.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(merged)`` where ``merged`` is the merged assessment.

    Tiles are unioned per level and per ``(tx, ty)``. Tiles sharing a
    coordinate must also share their ``(ix0, iy0, ix1, iy1)`` bounds; the
    merged tile keeps that geometry, sums the ``dzmin``/``dzmax`` deltas of
    the assessments that contain the coordinate (a missing tile contributes
    zero) and sums ``dcount`` exactly. The float sums are Decimal computations
    (precision 50, ``ROUND_HALF_EVEN``, each input converted via
    ``Decimal(str(v))``) with the summands accumulated in ascending Decimal
    order, quantized to six decimal places as floats (negative zero
    normalized). For each window, a merged tile matches when its closed
    cell-index intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the merged assessment's 9-tuples at that level, sorted
    lexicographically by ``(tx, ty)``; a window with no matching tile gets an
    empty ``tiles`` tuple. An empty ``assessments`` tuple is only valid with
    an empty ``windows`` tuple and returns ``()``. The result does not depend
    on the order of the inputs and the inputs are never modified.

    :raises TypeError: ``assessments``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: a member is not a valid assessment, the level counts
        differ, same-coordinate tiles disagree on their cell bounds, ``level``
        is out of range or the window bounds are inverted.
    """
    if not isinstance(assessments, tuple):
        raise TypeError("assessments must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")
    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    merged = merge_tile_pyramid_deltas(assessments)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(merged):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in merged[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


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


def update_tile_pyramid(pyramid: tuple,
                        points: Iterable[tuple | list],
                        cell_size: int | float = 1.0,
                        tile_cells: int = 256,
                        levels: int = 3) -> tuple:
    """Add points to a tile pyramid.

    Equivalent to
    ``merge_tile_pyramids((pyramid, build_tile_pyramid(points, cell_size, tile_cells, levels)))``
    but ``points`` is consumed in a single pass: the new points are aggregated
    into a pyramid exactly as in :func:`build_tile_pyramid` and the
    combination is then performed as in :func:`merge_tile_pyramids`.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid` with exactly ``levels`` levels: each level is a
    tuple of ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples
    sorted lexicographically by ``(tx, ty)`` (with no duplicate coordinates),
    where the first six fields and ``count`` are non-bool ints and
    ``zmin``/``zmax`` are finite floats. At level ``l`` every tile's bounds
    must be ``(tx * N, ty * N, (tx + 1) * N - 1, (ty + 1) * N - 1)`` where
    ``N = tile_cells * 2 ** l``.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list of
    finite non-bool ints/floats with ``sigma > 0``; cell indices are
    ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)`` and at level
    ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``.

    Returns the updated pyramid (a new tuple; the input pyramid and its tiles
    are never modified), with the same structure, ordering and fields as
    :func:`build_tile_pyramid`. An empty ``points`` iterable returns the
    original ``pyramid`` unchanged.

    :raises TypeError: ``pyramid`` is not a tuple, ``points`` is not iterable,
        ``cell_size``/``tile_cells``/``levels`` have the wrong type, or a
        point's container/length/fields have the wrong type.
    :raises ValueError: a parameter is non-finite or non-positive, the number
        of levels does not match ``pyramid``, the pyramid's structure,
        ordering, duplicates, fields or cell bounds are bad, a point field is
        non-finite or ``sigma <= 0``, or same-coordinate tiles disagree on
        their cell bounds.
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
    _validate_pyramid(pyramid)
    if len(pyramid) != levels:
        raise ValueError("all pyramids must have the same number of levels")
    for level, level_tiles in enumerate(pyramid):
        width = tile_cells * 2 ** level
        for tile in level_tiles:
            tx, ty, ix0, iy0, ix1, iy1 = tile[0:6]
            if (ix0, iy0, ix1, iy1) != (
                tx * width, ty * width,
                (tx + 1) * width - 1, (ty + 1) * width - 1,
            ):
                raise ValueError(
                    "tile bounds must be (tx * N, ty * N, (tx + 1) * N - 1, "
                    "(ty + 1) * N - 1) with N = tile_cells * 2 ** level"
                )

    # Single pass over ``points``: iter() is called exactly once, len() is
    # never called on it, and every level is aggregated simultaneously. The
    # new points are accumulated exactly as in build_tile_pyramid and the
    # resulting pyramid is then combined with ``pyramid`` via
    # merge_tile_pyramids, guaranteeing identical results.
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
            saw_point = True

        if not saw_point:
            return pyramid

        new_pyramid = []
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
            new_pyramid.append(tuple(level_tiles))

    return merge_tile_pyramids((pyramid, tuple(new_pyramid)))


def update_tile_pyramid_windows(pyramid: tuple,
                                points: Iterable[tuple | list],
                                windows: tuple,
                                cell_size: int | float = 1.0,
                                tile_cells: int = 256,
                                levels: int = 3) -> tuple:
    """Add points to a tile pyramid and select tiles intersecting windows.

    Equivalent to
    ``query_tile_pyramid_windows(update_tile_pyramid(pyramid, points, cell_size, tile_cells, levels), windows)``
    but ``points`` is consumed in a single pass: the new points are aggregated
    into a pyramid exactly as in :func:`build_tile_pyramid`, the combination
    is then performed as in :func:`merge_tile_pyramids` and the merged
    pyramid is queried as in :func:`query_tile_pyramid_windows`.

    ``pyramid`` must be an outer tuple as produced by
    :func:`build_tile_pyramid` with exactly ``levels`` levels: each level is a
    tuple of ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples
    sorted lexicographically by ``(tx, ty)`` (with no duplicate coordinates),
    where the first six fields and ``count`` are non-bool ints and
    ``zmin``/``zmax`` are finite floats. At level ``l`` every tile's bounds
    must be ``(tx * N, ty * N, (tx + 1) * N - 1, (ty + 1) * N - 1)`` where
    ``N = tile_cells * 2 ** l``.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list of
    finite non-bool ints/floats with ``sigma > 0``; cell indices are
    ``ix = floor(x / cell_size)``, ``iy = floor(y / cell_size)`` and at level
    ``l`` (``0 <= l < levels``) each tile covers
    ``N = tile_cells * 2 ** l`` cells per axis, so ``tx = ix // N`` and
    ``ty = iy // N``.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < levels``.

    For each window, a tile of the merged pyramid matches when its closed
    cell-index intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the merged pyramid's stored 9-tuples at that level, sorted
    lexicographically by ``(tx, ty)``; a window with no matching tile gets an
    empty ``tiles`` tuple and an empty ``windows`` tuple returns ``()``. An
    empty ``points`` iterable queries the original ``pyramid`` unchanged. The
    inputs are never modified, ``points`` is consumed with a single
    ``iter()`` call (``len()`` is never called on it) and the result does not
    depend on the order of the points.

    :raises TypeError: ``pyramid``/``windows`` is not a tuple, ``points`` is
        not iterable, a window's container/length/field types are bad, a
        point's container/length/fields have the wrong type, or
        ``cell_size``/``tile_cells``/``levels`` have the wrong type.
    :raises ValueError: a parameter is non-finite or non-positive, the number
        of levels does not match ``pyramid``, the pyramid's structure,
        ordering, duplicates, fields or cell bounds are bad, a point field is
        non-finite or ``sigma <= 0``, same-coordinate tiles disagree on their
        cell bounds, ``level`` is out of range or the window bounds are
        inverted.
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
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")
    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid(pyramid)
    if len(pyramid) != levels:
        raise ValueError("all pyramids must have the same number of levels")
    for level, level_tiles in enumerate(pyramid):
        width = tile_cells * 2 ** level
        for tile in level_tiles:
            tx, ty, ix0, iy0, ix1, iy1 = tile[0:6]
            if (ix0, iy0, ix1, iy1) != (
                tx * width, ty * width,
                (tx + 1) * width - 1, (ty + 1) * width - 1,
            ):
                raise ValueError(
                    "tile bounds must be (tx * N, ty * N, (tx + 1) * N - 1, "
                    "(ty + 1) * N - 1) with N = tile_cells * 2 ** level"
                )

    for window in windows:
        level, ix_min, iy_min, ix_max, iy_max = window
        if level < 0 or level >= levels:
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")

    # Single pass over ``points``: iter() is called exactly once, len() is
    # never called on it, and every level is aggregated simultaneously. The
    # new points are accumulated exactly as in build_tile_pyramid and the
    # resulting pyramid is then combined with ``pyramid`` via
    # merge_tile_pyramids, guaranteeing identical results.
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
            saw_point = True

        if saw_point:
            new_pyramid = []
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
                new_pyramid.append(tuple(level_tiles))

    if saw_point:
        merged = merge_tile_pyramids((pyramid, tuple(new_pyramid)))
    else:
        merged = pyramid

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        matched = tuple(
            tile for tile in merged[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def _validate_tile_pyramid_windows(result) -> None:
    """Validate the tuple returned by :func:`merge_tile_pyramid_windows`.

    Every member must be a 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` whose first five fields
    are non-bool ints with ``ix_min <= ix_max`` and ``iy_min <= iy_max`` and
    whose ``tiles`` is a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)``: the first six fields
    and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1`` and the tiles strictly sorted by
    ``(tx, ty)`` with no duplicates.
    """
    for window in result:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, tiles)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        _, ix_min, iy_min, ix_max, iy_max, tiles = window
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        if not isinstance(tiles, tuple):
            raise ValueError("tiles must be a tuple")
        prev_key = None
        for tile in tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "tx, ty, ix0, iy0, ix1, iy1 and count must be "
                        "non-bool ints"
                    )
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("zmin and zmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("tiles must be sorted by (tx, ty) with no "
                                 "duplicate coordinates")
            prev_key = key


def _format_pyramid_windows_text(result) -> str:
    """Build the canonical compact JSON text of a pyramid-windows document."""
    parts = ['{"windows":[']
    for window_index, window in enumerate(result):
        if window_index:
            parts.append(",")
        level, ix_min, iy_min, ix_max, iy_max, tiles = window
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",[")
        for tile_index, tile in enumerate(tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(zmin),
                                   _format_z(zmax), str(count))))
            parts.append("]")
        parts.append("]]")
    parts.append("]}")
    return "".join(parts)


def encode_tile_pyramid_windows(result: tuple) -> str:
    """Serialize the result of :func:`merge_tile_pyramid_windows`.

    ``result`` must be the tuple returned by
    :func:`merge_tile_pyramid_windows`: a tuple, in windows order, of
    6-tuples ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` whose first
    five fields are non-bool ints with ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``tiles`` is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` with the first six
    fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1`` and the tiles sorted by ``(tx, ty)`` with
    no duplicates.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being an array of nine-value tiles in the tiles' stored order.
    Integers are decimal; ``zmin``/``zmax`` use exactly six decimal places
    (negative zero written as ``0.000000``). The output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    result encodes as ``{"windows":[]}``. The input is never modified.

    :raises TypeError: ``result`` is not a tuple.
    :raises ValueError: the structure, field types, ordering, duplicates,
        cell bounds or non-finite values are bad.
    """
    if not isinstance(result, tuple):
        raise TypeError("result must be a tuple")
    _validate_tile_pyramid_windows(result)
    return _format_pyramid_windows_text(result)


def decode_tile_pyramid_windows(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_pyramid_windows`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, tiles]`` whose first five
    values are non-bool ints with ``ix_min <= ix_max``/``iy_min <= iy_max``
    and whose ``tiles`` is an array of strict nine-value tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count]`` with the first six
    fields and ``count`` non-bool ints, ``zmin``/``zmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1`` and the tiles strictly sorted by
    ``(tx, ty)`` with no duplicates, and whose spelling is exactly canonical
    (integers in decimal, ``zmin``/``zmax`` with six decimals, negative zero
    as ``0.000000``, no whitespace or extra keys, no ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each ``tiles`` field a
    tuple of 9-tuples in the document's order); the empty document
    ``{"windows":[]}`` returns ``()``. The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, lengths, bounds,
        ordering, duplicates, finiteness, numeric formatting or canonical
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

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_tiles = raw_window[5]
        if not isinstance(raw_tiles, list):
            raise ValueError("each window's tiles must be an array")
        tiles = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 9:
                raise ValueError("each tile must be an array of nine values")
            tiles.append(tuple(raw_tile))
        windows.append(tuple(raw_window[:5]) + (tuple(tiles),))
    result = tuple(windows)

    # Structural rules: field types/finiteness, window and tile bounds,
    # (tx, ty) sort order and no duplicates.
    _validate_tile_pyramid_windows(result)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal z formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_pyramid_windows_text(result) != text:
        raise ValueError("JSON text is not the canonical pyramid-windows "
                         "encoding")
    return result


def _validate_pyramid_delta_windows(result) -> None:
    """Validate the tuple returned by
    :func:`query_tile_pyramid_delta_windows`.

    Every member must be a 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` whose first five fields
    are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``tiles`` is a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)``: the first six
    fields and ``dcount`` non-bool ints, ``dzmin``/``dzmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1``, the tiles strictly sorted by ``(tx, ty)``
    with no duplicates, and each tile's closed cell-index interval
    intersecting the window.
    """
    for window in result:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, tiles)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, tiles = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        if not isinstance(tiles, tuple):
            raise ValueError("tiles must be a tuple")
        prev_key = None
        for tile in tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "tx, ty, ix0, iy0, ix1, iy1 and dcount must be "
                        "non-bool ints"
                    )
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("dzmin and dzmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if not (tile[4] >= ix_min and tile[2] <= ix_max
                    and tile[5] >= iy_min and tile[3] <= iy_max):
                raise ValueError("each tile must intersect the window's "
                                 "closed cell-index interval")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("tiles must be sorted by (tx, ty) with no "
                                 "duplicate coordinates")
            prev_key = key


def encode_tile_pyramid_delta_windows(result: tuple) -> str:
    """Serialize the result of :func:`query_tile_pyramid_delta_windows`.

    ``result`` must be the tuple returned by
    :func:`query_tile_pyramid_delta_windows`: a tuple, in windows order, of
    strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` whose first five fields
    are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``tiles`` is a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` with the first six
    fields and ``dcount`` non-bool ints, ``dzmin``/``dzmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1``, the tiles strictly sorted by ``(tx, ty)``
    with no duplicates, and each tile's closed cell-index interval
    intersecting the window.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being an array of nine-value tiles in the tiles' stored order.
    Integers are decimal; ``dzmin``/``dzmax`` use exactly six decimal places
    (negative zero written as ``0.000000``). The output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    result encodes as ``{"windows":[]}``. The input is never modified and
    repeated calls return identical results.

    :raises TypeError: ``result`` is not a tuple.
    :raises ValueError: the structure, field types, ordering, duplicates,
        cell bounds, level range, finiteness or window/tile intersection are
        bad.
    """
    if not isinstance(result, tuple):
        raise TypeError("result must be a tuple")
    _validate_pyramid_delta_windows(result)
    return _format_pyramid_windows_text(result)


def decode_tile_pyramid_delta_windows(text: str) -> tuple:
    """Deserialize canonical JSON produced by
    :func:`encode_tile_pyramid_delta_windows`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, tiles]`` whose first five
    values are non-bool ints with ``level >= 0``,
    ``ix_min <= ix_max``/``iy_min <= iy_max`` and whose ``tiles`` is an array
    of strict nine-value tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount]`` with the first six
    fields and ``dcount`` non-bool ints, ``dzmin``/``dzmax`` finite floats,
    ``ix0 <= ix1``/``iy0 <= iy1``, the tiles strictly sorted by ``(tx, ty)``
    with no duplicates and each tile's closed cell-index interval
    intersecting the window, and whose spelling is exactly canonical
    (integers in decimal, ``dzmin``/``dzmax`` with six decimals, negative
    zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each ``tiles`` field a
    tuple of 9-tuples in the document's order); the empty document
    ``{"windows":[]}`` returns ``()``. The input text is never modified and
    repeated calls return identical results.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, lengths, bounds,
        level range, ordering, duplicates, finiteness, window/tile
        intersection, numeric formatting or canonical re-encoding does not
        match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_tiles = raw_window[5]
        if not isinstance(raw_tiles, list):
            raise ValueError("each window's tiles must be an array")
        tiles = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 9:
                raise ValueError("each tile must be an array of nine values")
            tiles.append(tuple(raw_tile))
        windows.append(tuple(raw_window[:5]) + (tuple(tiles),))
    result = tuple(windows)

    # Structural rules: field types/finiteness, level range, window and tile
    # bounds, window/tile intersection, (tx, ty) sort order, no duplicates.
    _validate_pyramid_delta_windows(result)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal dz formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_pyramid_windows_text(result) != text:
        raise ValueError("JSON text is not the canonical pyramid-delta-windows "
                         "encoding")
    return result


def _validate_delta_summary_result(result) -> None:
    """Validate the tuple returned by
    :func:`aggregate_tile_pyramid_delta_windows`.

    Every member must be a 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``summary`` is either ``None`` or a
    strict 4-tuple
    ``(min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.
    """
    for window in result:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, summary)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, summary = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        if summary is None:
            continue
        if not isinstance(summary, tuple) or len(summary) != 4:
            raise ValueError(
                "summary must be None or a 4-tuple "
                "(min_dzmin, max_dzmax, sum_dcount, match_count)"
            )
        min_dzmin, max_dzmax, sum_dcount, match_count = summary
        for name, value in (("min_dzmin", min_dzmin),
                            ("max_dzmax", max_dzmax)):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if min_dzmin > max_dzmax:
            raise ValueError("summary must satisfy min_dzmin <= max_dzmax")
        for name, value in (("sum_dcount", sum_dcount),
                            ("match_count", match_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        if match_count <= 0:
            raise ValueError("match_count must be > 0")


def _format_delta_summary_text(result) -> str:
    """Build the canonical compact JSON text of a delta-summary document."""
    parts = ['{"windows":[']
    for window_index, (level, ix_min, iy_min, ix_max, iy_max,
                       summary) in enumerate(result):
        if window_index:
            parts.append(",")
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if summary is None:
            parts.append("null")
        else:
            min_dzmin, max_dzmax, sum_dcount, match_count = summary
            parts.append("[")
            parts.append(",".join((_format_z(min_dzmin),
                                   _format_z(max_dzmax),
                                   str(sum_dcount),
                                   str(match_count))))
            parts.append("]")
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_delta_summary(result: tuple) -> str:
    """Serialize the result of
    :func:`aggregate_tile_pyramid_delta_windows`.

    ``result`` must be that function's returned tuple: a tuple, in windows
    order, of strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``summary`` is either ``None`` or a
    strict 4-tuple
    ``(min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being ``null`` or a four-value summary array. Integers are
    decimal; ``min_dzmin``/``max_dzmax`` use exactly six decimal places
    (negative zero written as ``0.000000``). The output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    result encodes as ``{"windows":[]}``. The input is never modified.

    :raises TypeError: ``result`` is not a tuple.
    :raises ValueError: the structure, field types, level range, bounds,
        summary shape, finiteness or ``match_count`` are bad.
    """
    if not isinstance(result, tuple):
        raise TypeError("result must be a tuple")
    _validate_delta_summary_result(result)
    return _format_delta_summary_text(result)


def decode_delta_summary(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_delta_summary`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, summary]`` whose first
    five values are non-bool ints with ``level >= 0``,
    ``ix_min <= ix_max``/``iy_min <= iy_max`` and whose ``summary`` is
    either ``null`` or a strict four-value array
    ``[min_dzmin, max_dzmax, sum_dcount, match_count]`` with the first two
    values finite floats with ``min_dzmin <= max_dzmax`` and the last two
    non-bool ints with ``match_count > 0``, and whose spelling is exactly
    canonical (integers in decimal, the floats with six decimals, negative
    zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each non-null
    ``summary`` a 4-tuple); the empty document ``{"windows":[]}`` returns
    ``()``. The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, lengths, level
        range, bounds, summary shape, finiteness, ``match_count``, numeric
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

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_summary = raw_window[5]
        if raw_summary is not None:
            if not isinstance(raw_summary, list) or len(raw_summary) != 4:
                raise ValueError(
                    "each window's summary must be null or an array of "
                    "four values"
                )
            raw_summary = tuple(raw_summary)
        windows.append(tuple(raw_window[:5]) + (raw_summary,))
    result = tuple(windows)

    # Structural rules: field types/finiteness, level range, window bounds,
    # summary fields and match_count.
    _validate_delta_summary_result(result)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal float formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_delta_summary_text(result) != text:
        raise ValueError("JSON text is not the canonical delta-summary "
                         "encoding")
    return result


def _validate_summary_result(result) -> None:
    """Validate the tuple returned by
    :func:`summarize_tile_pyramid_windows`.

    Every member must be a 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``summary`` is either ``None`` or a
    strict 5-tuple
    ``(bias_min, bias_max, abs_error_mean, z_score_rms, count_delta)`` where
    the first four values are finite floats with ``bias_min <= bias_max`` and
    ``abs_error_mean``/``z_score_rms >= 0`` and ``count_delta`` is a
    non-bool int.
    """
    for window in result:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, summary)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, summary = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        if summary is None:
            continue
        if not isinstance(summary, tuple) or len(summary) != 5:
            raise ValueError(
                "summary must be None or a 5-tuple "
                "(bias_min, bias_max, abs_error_mean, z_score_rms, "
                "count_delta)"
            )
        (bias_min, bias_max, abs_error_mean, z_score_rms,
         count_delta) = summary
        for name, value in (("bias_min", bias_min),
                            ("bias_max", bias_max),
                            ("abs_error_mean", abs_error_mean),
                            ("z_score_rms", z_score_rms)):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if bias_min > bias_max:
            raise ValueError("summary must satisfy bias_min <= bias_max")
        if abs_error_mean < 0.0 or z_score_rms < 0.0:
            raise ValueError("abs_error_mean and z_score_rms must be >= 0")
        if isinstance(count_delta, bool) or not isinstance(count_delta, int):
            raise ValueError("count_delta must be a non-bool int")


def _format_summary_text(result) -> str:
    """Build the canonical compact JSON text of a window-summary document."""
    parts = ['{"windows":[']
    for window_index, (level, ix_min, iy_min, ix_max, iy_max,
                       summary) in enumerate(result):
        if window_index:
            parts.append(",")
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if summary is None:
            parts.append("null")
        else:
            (bias_min, bias_max, abs_error_mean, z_score_rms,
             count_delta) = summary
            parts.append("[")
            parts.append(",".join((_format_z(bias_min),
                                   _format_z(bias_max),
                                   _format_z(abs_error_mean),
                                   _format_z(z_score_rms),
                                   str(count_delta))))
            parts.append("]")
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_summary(result: tuple) -> str:
    """Serialize the result of :func:`summarize_tile_pyramid_windows`.

    ``result`` must be that function's returned tuple: a tuple, in windows
    order, of strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``summary`` is either ``None`` or a
    strict 5-tuple
    ``(bias_min, bias_max, abs_error_mean, z_score_rms, count_delta)`` where
    the first four values are finite floats with ``bias_min <= bias_max`` and
    ``abs_error_mean``/``z_score_rms >= 0`` and ``count_delta`` is a
    non-bool int.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being ``null`` or a five-value summary array. Integers are
    decimal; the four summary floats use exactly six decimal places
    (negative zero written as ``0.000000``). The output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    result encodes as ``{"windows":[]}``. The input is never modified.

    :raises TypeError: ``result`` is not a tuple.
    :raises ValueError: the structure, field types, level range, bounds,
        summary shape, finiteness or ranges are bad.
    """
    if not isinstance(result, tuple):
        raise TypeError("result must be a tuple")
    _validate_summary_result(result)
    return _format_summary_text(result)


def decode_summary(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_summary`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, summary]`` whose first
    five values are non-bool ints with ``level >= 0``,
    ``ix_min <= ix_max``/``iy_min <= iy_max`` and whose ``summary`` is
    either ``null`` or a strict five-value array
    ``[bias_min, bias_max, abs_error_mean, z_score_rms, count_delta]`` with
    the first four values finite floats with ``bias_min <= bias_max`` and
    ``abs_error_mean``/``z_score_rms >= 0`` and the last value a non-bool
    int, and whose spelling is exactly canonical (integers in decimal, the
    floats with six decimals, negative zero as ``0.000000``, no whitespace
    or extra keys, no ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each non-null
    ``summary`` a 5-tuple); the empty document ``{"windows":[]}`` returns
    ``()``. The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, lengths, level
        range, bounds, summary shape, finiteness, ranges, numeric
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

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_summary = raw_window[5]
        if raw_summary is not None:
            if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                raise ValueError(
                    "each window's summary must be null or an array of "
                    "five values"
                )
            raw_summary = tuple(raw_summary)
        windows.append(tuple(raw_window[:5]) + (raw_summary,))
    result = tuple(windows)

    # Structural rules: field types/finiteness, level range, window bounds,
    # summary fields and ranges.
    _validate_summary_result(result)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal float formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_summary_text(result) != text:
        raise ValueError("JSON text is not the canonical window-summary "
                         "encoding")
    return result


def _validate_delta_summary_entries(entries) -> None:
    """Validate an input tuple of delta-summary 6-tuples for
    :func:`assess_delta_summary`, raising ``ValueError``.

    Each member must be a strict 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max``; the five-field keys must be strictly increasing
    with no duplicates. ``summary`` is either ``None`` or a strict 4-tuple
    ``(min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite non-bool ints or floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.
    """
    prev_key = None
    for window in entries:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each entry must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, summary)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, summary = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        key = (level, ix_min, iy_min, ix_max, iy_max)
        if prev_key is not None and key <= prev_key:
            raise ValueError(
                "entries must be sorted strictly by "
                "(level, ix_min, iy_min, ix_max, iy_max) with no duplicate "
                "keys"
            )
        prev_key = key

        if summary is None:
            continue
        if not isinstance(summary, tuple) or len(summary) != 4:
            raise ValueError(
                "summary must be None or a 4-tuple "
                "(min_dzmin, max_dzmax, sum_dcount, match_count)"
            )
        min_dzmin, max_dzmax, sum_dcount, match_count = summary
        for name, value in (("min_dzmin", min_dzmin),
                            ("max_dzmax", max_dzmax)):
            if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
                raise ValueError(f"{name} must be a non-bool int or float")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if Decimal(str(min_dzmin)) > Decimal(str(max_dzmax)):
            raise ValueError("summary must satisfy min_dzmin <= max_dzmax")
        for name, value in (("sum_dcount", sum_dcount),
                            ("match_count", match_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        if match_count <= 0:
            raise ValueError("match_count must be > 0")


def assess_delta_summary(estimate: tuple, reference: tuple) -> tuple:
    """Compute per-window summary deltas of an estimated vs a reference result.

    Both inputs are tuples as returned by
    :func:`aggregate_tile_pyramid_delta_windows`: members are strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose five-field keys are strictly increasing
    with no duplicates. ``summary`` is either ``None`` or a strict 4-tuple
    ``(min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite non-bool ints or floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.

    The two inputs must contain the same five-field keys, and the two
    summaries for each key must agree on being ``None`` versus present;
    otherwise ``ValueError`` is raised.

    Returns a tuple, in ``estimate`` order, of 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, delta)``. When both summaries
    are ``None``, ``delta`` is ``None``; otherwise it is the 4-tuple
    ``(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` with each value
    computed as ``estimate - reference``. The two count deltas are exact
    ints; a minimum/maximum delta is an exact int when both operands are
    ints, and otherwise a Decimal computation
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) quantized to
    six decimal places as a float (negative zero normalized). Two empty
    inputs return ``()``. The inputs are never modified.

    :raises TypeError: ``estimate`` or ``reference`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicate keys, field
        types, level range, bounds, summary shape, finiteness,
        ``match_count``, key sets or None states of either input are bad.
    """
    if not isinstance(estimate, tuple):
        raise TypeError("estimate must be a tuple")
    if not isinstance(reference, tuple):
        raise TypeError("reference must be a tuple")

    _validate_delta_summary_entries(estimate)
    _validate_delta_summary_entries(reference)

    estimate_states = {window[:5]: window[5] is None for window in estimate}
    reference_states = {window[:5]: window[5] is None for window in reference}
    if estimate_states != reference_states:
        raise ValueError("estimate and reference must contain the same keys "
                         "with matching summary None states")
    reference_summaries = {window[:5]: window[5] for window in reference}

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, ix_min, iy_min, ix_max, iy_max, est_summary in estimate:
            key = (level, ix_min, iy_min, ix_max, iy_max)
            if est_summary is None:
                delta = None
            else:
                ref_summary = reference_summaries[key]
                est_min, est_max, est_sum, est_matches = est_summary
                ref_min, ref_max, ref_sum, ref_matches = ref_summary
                if isinstance(est_min, int) and isinstance(ref_min, int):
                    dmin = est_min - ref_min
                else:
                    dmin = _quantize(
                        Decimal(str(est_min)) - Decimal(str(ref_min)))
                if isinstance(est_max, int) and isinstance(ref_max, int):
                    dmax = est_max - ref_max
                else:
                    dmax = _quantize(
                        Decimal(str(est_max)) - Decimal(str(ref_max)))
                delta = (dmin, dmax,
                         est_sum - ref_sum, est_matches - ref_matches)
            result.append((level, ix_min, iy_min, ix_max, iy_max, delta))

    return tuple(result)


def _validate_delta_assessment(assessment) -> None:
    """Validate the tuple returned by :func:`assess_delta_summary`.

    Every member must be a strict 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, delta)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max``; the five-field keys must be strictly increasing
    with no duplicates. ``delta`` is either ``None`` or a strict 4-tuple
    ``(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the first
    two values are finite non-bool ints or floats (in either order) and the
    last two are non-bool ints (of any sign).
    """
    prev_key = None
    for window in assessment:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a strict 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, delta)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, delta = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        key = (level, ix_min, iy_min, ix_max, iy_max)
        if prev_key is not None and key <= prev_key:
            raise ValueError(
                "windows must be sorted strictly by "
                "(level, ix_min, iy_min, ix_max, iy_max) with no duplicate "
                "keys"
            )
        prev_key = key

        if delta is None:
            continue
        if not isinstance(delta, tuple) or len(delta) != 4:
            raise ValueError(
                "delta must be None or a strict 4-tuple "
                "(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)"
            )
        dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count = delta
        for name, value in (("dmin_dzmin", dmin_dzmin),
                            ("dmax_dzmax", dmax_dzmax)):
            if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
                raise ValueError(f"{name} must be a non-bool int or float")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (("dsum_dcount", dsum_dcount),
                            ("dmatch_count", dmatch_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")


def _format_delta_assessment_text(assessment) -> str:
    """Build the canonical compact JSON text of a delta-assessment document."""
    parts = ['{"windows":[']
    for window_index, (level, ix_min, iy_min, ix_max, iy_max,
                       delta) in enumerate(assessment):
        if window_index:
            parts.append(",")
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if delta is None:
            parts.append("null")
        else:
            dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count = delta
            parts.append("[")
            parts.append(",".join((
                str(dmin_dzmin) if isinstance(dmin_dzmin, int)
                else _format_z(dmin_dzmin),
                str(dmax_dzmax) if isinstance(dmax_dzmax, int)
                else _format_z(dmax_dzmax),
                str(dsum_dcount),
                str(dmatch_count))))
            parts.append("]")
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def encode_delta_assessment(assessment: tuple) -> str:
    """Serialize the result of :func:`assess_delta_summary`.

    ``assessment`` must be that function's returned tuple: a tuple, in
    windows order, of strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, delta)`` whose first five
    fields are non-bool ints with ``level >= 0`` and ordered bounds, whose
    five-field keys are strictly increasing with no duplicates, and whose
    ``delta`` is either ``None`` or a strict 4-tuple
    ``(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the first
    two values are finite non-bool ints or floats (in either order) and the
    last two are non-bool ints of any sign.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being ``null`` or a four-value delta array. Integers are decimal;
    float deltas use exactly six decimal places (negative zero written as
    ``0.000000``). The output has no whitespace, ASCII is not escaped and
    ``NaN``/``Infinity`` never appear. An empty assessment encodes as
    ``{"windows":[]}``. The input is never modified.

    :raises TypeError: ``assessment`` is not a tuple.
    :raises ValueError: the structure, field types, level range, bounds,
        key ordering/duplicates, delta shape or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    _validate_delta_assessment(assessment)
    return _format_delta_assessment_text(assessment)


def decode_delta_assessment(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_delta_assessment`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, delta]`` whose first
    five values are non-bool ints with ``level >= 0`` and ordered bounds,
    whose five-field keys are strictly increasing with no duplicates, and
    whose ``delta`` is either ``null`` or a strict four-value array
    ``[dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count]`` with the first
    two values finite non-bool ints or floats (in either order) and the last
    two non-bool ints of any sign, and whose spelling is exactly canonical
    (integers in decimal, float deltas with six decimals, negative zero as
    ``0.000000``, no whitespace or extra keys, no ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each non-null
    ``delta`` a 4-tuple); the empty document ``{"windows":[]}`` returns
    ``()``. The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, structure, types,
        finiteness, level range, bounds, key ordering/duplicates, numeric
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

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_delta = raw_window[5]
        if raw_delta is not None:
            if not isinstance(raw_delta, list) or len(raw_delta) != 4:
                raise ValueError(
                    "each window's delta must be null or an array of "
                    "four values"
                )
            raw_delta = tuple(raw_delta)
        windows.append(tuple(raw_window[:5]) + (raw_delta,))
    assessment = tuple(windows)

    # Structural rules: field types/finiteness, level range, window bounds,
    # strictly increasing keys and delta fields.
    _validate_delta_assessment(assessment)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal float formatting, ints written as
    # floats (or vice versa), leading zeros, -0, exponents and any other
    # non-canonical spelling.
    if _format_delta_assessment_text(assessment) != text:
        raise ValueError("JSON text is not the canonical delta-assessment "
                         "encoding")
    return assessment


def query_delta_summary_windows(assessment: tuple, windows: tuple) -> tuple:
    """Select delta-summary entries intersecting cell-index windows by level.

    ``assessment`` must be the outer tuple returned by
    :func:`assess_delta_summary`: a tuple of strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, delta)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max``; the five-field keys must be strictly increasing
    with no duplicates. ``delta`` is either ``None`` or a strict 4-tuple
    ``(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the first
    two values are finite non-bool ints or floats and the last two are
    non-bool ints.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool
    ints with ``level >= 0``, ``ix_min <= ix_max`` and ``iy_min <= iy_max``.

    For each window, an entry matches when it is at the window's level and
    its closed cell-index intervals intersect the window:
    ``entry.ix_max >= ix_min and entry.ix_min <= ix_max and
    entry.iy_max >= iy_min and entry.iy_min <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, entries)`` tuples where
    ``entries`` is a tuple of the matching stored 6-tuples, preserving the
    assessment's existing order; a window with no matching entry gets an
    empty ``entries`` tuple and an empty ``windows`` tuple returns ``()``.
    The input is never modified and repeated calls return identical results.

    :raises TypeError: ``assessment``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: a window's ``level`` is negative, the window bounds
        are inverted, or the assessment's structure, ordering, duplicates,
        fields, cell bounds, delta shape or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_delta_assessment(assessment)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            entry for entry in assessment
            if (entry[0] == level
                and entry[3] >= ix_min and entry[1] <= ix_max
                and entry[4] >= iy_min and entry[2] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def merge_delta_assessments(assessments: tuple) -> tuple:
    """Merge delta assessments produced by :func:`assess_delta_summary`.

    ``assessments`` must be a tuple of such assessments (it may be empty, in
    which case ``()`` is returned). Each member must be a tuple of strict
    6-tuples ``(level, ix_min, iy_min, ix_max, iy_max, delta)`` whose first
    five fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max``
    and ``iy_min <= iy_max``; the five-field keys must be strictly increasing
    with no duplicates. ``delta`` is either ``None`` or a strict 4-tuple
    ``(dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the first
    two values are finite non-bool ints or floats and the last two are
    non-bool ints.

    The five-field keys are unioned across the assessments; a key missing
    from an assessment contributes nothing. Assessments sharing a key must
    agree on its ``delta`` being ``None`` versus present, otherwise
    ``ValueError`` is raised. When every contributing ``delta`` is ``None``
    the merged ``delta`` is ``None``; otherwise the four fields are summed
    term by term over the contributing deltas. The two count deltas are
    exact ints; a minimum/maximum delta is an exact int sum when every
    contributing term is an int, and otherwise a Decimal computation
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) with the
    summands accumulated in ascending Decimal order, quantized to six
    decimal places as a float (negative zero normalized).

    Returns a tuple of 6-tuples sorted by the five-field key. The result
    does not depend on the order of the inputs and the inputs are never
    modified.

    :raises TypeError: ``assessments`` is not a tuple.
    :raises ValueError: a member is not a valid assessment (structure,
        field types, level range, bounds, key ordering/duplicates, delta
        shape or finiteness) or assessments sharing a key disagree on the
        delta being ``None`` versus present.
    """
    if not isinstance(assessments, tuple):
        raise TypeError("assessments must be a tuple")
    if not assessments:
        return ()

    for assessment in assessments:
        if not isinstance(assessment, tuple):
            raise ValueError("each assessment must be a tuple as returned "
                             "by assess_delta_summary")
        _validate_delta_assessment(assessment)

    # key -> None (all contributing deltas None) or [dmin terms, dmax
    # terms, dsum_dcount, dmatch_count]; the per-field summands are
    # collected so their accumulation order does not depend on input order.
    combined: dict[tuple, list | None] = {}
    for assessment in assessments:
        for window in assessment:
            key = window[:5]
            delta = window[5]
            if key not in combined:
                combined[key] = (None if delta is None
                                 else [[delta[0]], [delta[1]],
                                       delta[2], delta[3]])
                continue
            entry = combined[key]
            if (entry is None) != (delta is None):
                raise ValueError(
                    "assessments sharing a key must agree on the delta "
                    "being None versus present"
                )
            if delta is not None:
                entry[0].append(delta[0])
                entry[1].append(delta[1])
                entry[2] += delta[2]
                entry[3] += delta[3]

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        merged = []
        for key in sorted(combined):
            entry = combined[key]
            if entry is None:
                delta = None
            else:
                dmin_terms, dmax_terms, dsum_dcount, dmatch_count = entry
                if all(isinstance(term, int) for term in dmin_terms):
                    dmin = sum(dmin_terms)
                else:
                    dmin = _quantize(_sorted_sum(
                        [Decimal(str(term)) for term in dmin_terms]))
                if all(isinstance(term, int) for term in dmax_terms):
                    dmax = sum(dmax_terms)
                else:
                    dmax = _quantize(_sorted_sum(
                        [Decimal(str(term)) for term in dmax_terms]))
                delta = (dmin, dmax, dsum_dcount, dmatch_count)
            merged.append(key + (delta,))

    return tuple(merged)


def apply_delta_assessment(base: tuple, delta: tuple) -> tuple:
    """Apply a delta assessment to a base of delta-window summaries.

    ``base`` must be a tuple as returned by
    :func:`aggregate_tile_pyramid_delta_windows`: members are strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose five-field keys are strictly increasing
    with no duplicates. ``summary`` is either ``None`` or a strict 4-tuple
    ``S = (min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite non-bool ints or floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.

    ``delta`` must be a tuple as returned by :func:`assess_delta_summary`:
    members are the same strict 6-tuples with strictly increasing five-field
    keys, and ``delta`` is either ``None`` or a strict 4-tuple
    ``D = (dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the
    first two values are finite non-bool ints or floats (in either order)
    and the last two are non-bool ints of any sign.

    The two inputs must contain the same five-field keys, and the two values
    for each key must agree on being ``None`` versus present; otherwise
    ``ValueError`` is raised.

    Returns a tuple, in ``base`` order, of the same 6-tuples. A ``None``
    value stays ``None``; otherwise the summary is ``S + D`` term by term.
    The two count sums are exact ints; a minimum/maximum sum is an exact int
    when both operands are ints, and otherwise a Decimal computation
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) quantized to
    six decimal places as a float (negative zero normalized). Each resulting
    summary must satisfy ``min_dzmin <= max_dzmax`` and ``match_count > 0``;
    otherwise ``ValueError`` is raised. Two empty inputs return ``()``. The
    inputs are never modified.

    :raises TypeError: ``base`` or ``delta`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicate keys, field
        types, level range, bounds, summary/delta shape, finiteness or
        ``match_count`` of either input are bad, their key sets or None
        states disagree, or a resulting summary violates
        ``min_dzmin <= max_dzmax`` or ``match_count > 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(delta, tuple):
        raise TypeError("delta must be a tuple")

    _validate_delta_summary_entries(base)
    _validate_delta_assessment(delta)

    base_states = {window[:5]: window[5] is None for window in base}
    delta_states = {window[:5]: window[5] is None for window in delta}
    if base_states != delta_states:
        raise ValueError("base and delta must contain the same keys with "
                         "matching None states")
    delta_values = {window[:5]: window[5] for window in delta}

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, ix_min, iy_min, ix_max, iy_max, summary in base:
            key = (level, ix_min, iy_min, ix_max, iy_max)
            if summary is None:
                value = None
            else:
                base_min, base_max, base_sum, base_matches = summary
                d_min, d_max, d_sum, d_matches = delta_values[key]
                if isinstance(base_min, int) and isinstance(d_min, int):
                    new_min = base_min + d_min
                else:
                    new_min = _quantize(
                        Decimal(str(base_min)) + Decimal(str(d_min)))
                if isinstance(base_max, int) and isinstance(d_max, int):
                    new_max = base_max + d_max
                else:
                    new_max = _quantize(
                        Decimal(str(base_max)) + Decimal(str(d_max)))
                new_sum = base_sum + d_sum
                new_matches = base_matches + d_matches

                if isinstance(new_min, int) and isinstance(new_max, int):
                    ordered = new_min <= new_max
                else:
                    ordered = (Decimal(str(new_min))
                               <= Decimal(str(new_max)))
                if not ordered:
                    raise ValueError(
                        "resulting summary must satisfy min_dzmin <= "
                        "max_dzmax"
                    )
                if new_matches <= 0:
                    raise ValueError("resulting match_count must be > 0")

                value = (new_min, new_max, new_sum, new_matches)
            result.append((level, ix_min, iy_min, ix_max, iy_max, value))

    return tuple(result)


def apply_delta_assessments(base: tuple, deltas: tuple) -> tuple:
    """Apply a tuple of delta assessments to a base of delta-window summaries.

    ``base`` must be a tuple as returned by
    :func:`aggregate_tile_pyramid_delta_windows`: members are strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose five-field keys are strictly increasing
    with no duplicates. ``summary`` is either ``None`` or a strict 4-tuple
    ``S = (min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite non-bool ints or floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.

    ``deltas`` must be a tuple of assessments as returned by
    :func:`assess_delta_summary`: each member is a tuple of the same strict
    6-tuples with strictly increasing five-field keys, and ``delta`` is
    either ``None`` or a strict 4-tuple
    ``D = (dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the
    first two values are finite non-bool ints or floats (in either order)
    and the last two are non-bool ints of any sign. Every member assessment
    must contain the same five-field keys as ``base`` and agree with
    ``base`` on each key's value being ``None`` versus present; otherwise
    ``ValueError`` is raised. An empty ``deltas`` returns ``base``.

    Returns a tuple, in ``base`` order, of the same 6-tuples. A ``None``
    value stays ``None``; otherwise the summary is ``S`` plus every
    contributing ``D`` term by term. The two count sums are exact ints; a
    minimum/maximum sum is an exact int when the base value and every
    contributing term are ints, and otherwise a Decimal computation
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) with the
    summands accumulated in ascending Decimal order, quantized to six
    decimal places as a float (negative zero normalized). Each resulting
    summary must satisfy ``min_dzmin <= max_dzmax`` and ``match_count > 0``;
    otherwise ``ValueError`` is raised. The result does not depend on the
    order of ``deltas`` and the inputs are never modified.

    :raises TypeError: ``base`` or ``deltas`` is not a tuple.
    :raises ValueError: a member of ``deltas`` is not a valid assessment,
        or the structure, ordering, duplicate keys, field types, level
        range, bounds, summary/delta shape, finiteness or ``match_count``
        of any input are bad, the key sets or None states disagree, or a
        resulting summary violates ``min_dzmin <= max_dzmax`` or
        ``match_count > 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")

    _validate_delta_summary_entries(base)

    for delta in deltas:
        if not isinstance(delta, tuple):
            raise ValueError("each delta must be a tuple as returned "
                             "by assess_delta_summary")
        _validate_delta_assessment(delta)

    if not deltas:
        return base

    base_states = {window[:5]: window[5] is None for window in base}
    for delta in deltas:
        delta_states = {window[:5]: window[5] is None for window in delta}
        if delta_states != base_states:
            raise ValueError("base and each delta must contain the same "
                             "keys with matching None states")

    # key -> [dmin terms, dmax terms, dsum_dcount, dmatch_count] for keys
    # whose summary is present; the per-field summands are collected so
    # their accumulation order does not depend on the order of ``deltas``.
    contributions: dict[tuple, list] = {}
    for delta in deltas:
        for window in delta:
            value = window[5]
            if value is None:
                continue
            key = window[:5]
            entry = contributions.get(key)
            if entry is None:
                entry = [[], [], 0, 0]
                contributions[key] = entry
            entry[0].append(value[0])
            entry[1].append(value[1])
            entry[2] += value[2]
            entry[3] += value[3]

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, ix_min, iy_min, ix_max, iy_max, summary in base:
            if summary is None:
                result.append((level, ix_min, iy_min, ix_max, iy_max, None))
                continue
            key = (level, ix_min, iy_min, ix_max, iy_max)
            base_min, base_max, base_sum, base_matches = summary
            dmin_terms, dmax_terms, dsum_dcount, dmatch_count = \
                contributions[key]

            min_terms = [base_min] + dmin_terms
            if all(isinstance(term, int) for term in min_terms):
                new_min = sum(min_terms)
            else:
                new_min = _quantize(_sorted_sum(
                    [Decimal(str(term)) for term in min_terms]))
            max_terms = [base_max] + dmax_terms
            if all(isinstance(term, int) for term in max_terms):
                new_max = sum(max_terms)
            else:
                new_max = _quantize(_sorted_sum(
                    [Decimal(str(term)) for term in max_terms]))
            new_sum = base_sum + dsum_dcount
            new_matches = base_matches + dmatch_count

            if isinstance(new_min, int) and isinstance(new_max, int):
                ordered = new_min <= new_max
            else:
                ordered = (Decimal(str(new_min))
                           <= Decimal(str(new_max)))
            if not ordered:
                raise ValueError(
                    "resulting summary must satisfy min_dzmin <= "
                    "max_dzmax"
                )
            if new_matches <= 0:
                raise ValueError("resulting match_count must be > 0")

            result.append((level, ix_min, iy_min, ix_max, iy_max,
                           (new_min, new_max, new_sum, new_matches)))

    return tuple(result)


def rollback_delta_assessments(base: tuple, deltas: tuple) -> tuple:
    """Subtract a tuple of delta assessments from delta-window summaries.

    ``base`` must be a tuple as returned by
    :func:`aggregate_tile_pyramid_delta_windows`: members are strict 6-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max, summary)`` whose first five
    fields are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose five-field keys are strictly increasing
    with no duplicates. ``summary`` is either ``None`` or a strict 4-tuple
    ``S = (min_dzmin, max_dzmax, sum_dcount, match_count)`` where
    ``min_dzmin``/``max_dzmax`` are finite non-bool ints or floats with
    ``min_dzmin <= max_dzmax`` and ``sum_dcount``/``match_count`` are
    non-bool ints with ``match_count > 0``.

    ``deltas`` must be a tuple of assessments as returned by
    :func:`assess_delta_summary`: each member is a tuple of the same strict
    6-tuples with strictly increasing five-field keys, and ``delta`` is
    either ``None`` or a strict 4-tuple
    ``D = (dmin_dzmin, dmax_dzmax, dsum_dcount, dmatch_count)`` where the
    first two values are finite non-bool ints or floats (in either order)
    and the last two are non-bool ints of any sign. Every member assessment
    must contain the same five-field keys as ``base`` and agree with
    ``base`` on each key's value being ``None`` versus present; otherwise
    ``ValueError`` is raised. An empty ``deltas`` returns ``base``.

    Returns a tuple, in ``base`` order, of the same 6-tuples. A ``None``
    value stays ``None``; otherwise the summary is ``S`` minus the sum of
    every contributing ``D``, term by term. The two count differences are
    exact ints; a minimum/maximum difference is an exact int when the base
    value and every contributing term are ints, and otherwise a Decimal
    computation (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``)
    where the contributing terms are summed in ascending Decimal order
    before being subtracted, quantized to six decimal places as a float
    (negative zero normalized). Each resulting summary must satisfy
    ``min_dzmin <= max_dzmax`` and ``match_count > 0``; otherwise
    ``ValueError`` is raised. The result does not depend on the order of
    ``deltas`` and the inputs are never modified.

    :raises TypeError: ``base`` or ``deltas`` is not a tuple.
    :raises ValueError: a member of ``deltas`` is not a valid assessment,
        or the structure, ordering, duplicate keys, field types, level
        range, bounds, summary/delta shape, finiteness or ``match_count``
        of any input are bad, the key sets or None states disagree, or a
        resulting summary violates ``min_dzmin <= max_dzmax`` or
        ``match_count > 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")

    _validate_delta_summary_entries(base)

    for delta in deltas:
        if not isinstance(delta, tuple):
            raise ValueError("each delta must be a tuple as returned "
                             "by assess_delta_summary")
        _validate_delta_assessment(delta)

    if not deltas:
        return base

    base_states = {window[:5]: window[5] is None for window in base}
    for delta in deltas:
        delta_states = {window[:5]: window[5] is None for window in delta}
        if delta_states != base_states:
            raise ValueError("base and each delta must contain the same "
                             "keys with matching None states")

    # key -> [dmin terms, dmax terms, dsum_dcount, dmatch_count] for keys
    # whose summary is present; the per-field terms are collected so their
    # accumulation order does not depend on the order of ``deltas``.
    contributions: dict[tuple, list] = {}
    for delta in deltas:
        for window in delta:
            value = window[5]
            if value is None:
                continue
            key = window[:5]
            entry = contributions.get(key)
            if entry is None:
                entry = [[], [], 0, 0]
                contributions[key] = entry
            entry[0].append(value[0])
            entry[1].append(value[1])
            entry[2] += value[2]
            entry[3] += value[3]

    result = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for level, ix_min, iy_min, ix_max, iy_max, summary in base:
            if summary is None:
                result.append((level, ix_min, iy_min, ix_max, iy_max, None))
                continue
            key = (level, ix_min, iy_min, ix_max, iy_max)
            base_min, base_max, base_sum, base_matches = summary
            dmin_terms, dmax_terms, dsum_dcount, dmatch_count = \
                contributions[key]

            min_terms = [base_min] + dmin_terms
            if all(isinstance(term, int) for term in min_terms):
                new_min = base_min - sum(dmin_terms)
            else:
                new_min = _quantize(
                    Decimal(str(base_min)) - _sorted_sum(
                        [Decimal(str(term)) for term in dmin_terms]))
            max_terms = [base_max] + dmax_terms
            if all(isinstance(term, int) for term in max_terms):
                new_max = base_max - sum(dmax_terms)
            else:
                new_max = _quantize(
                    Decimal(str(base_max)) - _sorted_sum(
                        [Decimal(str(term)) for term in dmax_terms]))
            new_sum = base_sum - dsum_dcount
            new_matches = base_matches - dmatch_count

            if isinstance(new_min, int) and isinstance(new_max, int):
                ordered = new_min <= new_max
            else:
                ordered = (Decimal(str(new_min))
                           <= Decimal(str(new_max)))
            if not ordered:
                raise ValueError(
                    "resulting summary must satisfy min_dzmin <= "
                    "max_dzmax"
                )
            if new_matches <= 0:
                raise ValueError("resulting match_count must be > 0")

            result.append((level, ix_min, iy_min, ix_max, iy_max,
                           (new_min, new_max, new_sum, new_matches)))

    return tuple(result)


def rollback_tile_pyramid_deltas(base: tuple, deltas: tuple) -> tuple:
    """Subtract per-tile pyramid deltas from a base tile pyramid.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax`` are
    finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1`` and
    ``dzmin``/``dzmax`` are finite floats. At every level the two inputs must
    contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds; otherwise ``ValueError`` is raised. An
    empty ``deltas`` returns ``base``.

    Returns a tuple, in level order, of tuples in ``base`` order: the
    geometry is taken unchanged from each base tile and ``zmin``/``zmax``
    equal the base value minus the same-coordinate delta value, ``count`` the
    base count minus the same-coordinate ``dcount``. The float differences
    are Decimal computations (``Decimal(str(v))``, precision 50,
    ``ROUND_HALF_EVEN``); when the base value and the delta value are
    integral the result is kept as an exact int, otherwise it is quantized to
    six decimal places and converted to a float (negative zero normalized).
    Each resulting tile must satisfy ``zmin <= zmax`` and ``count >= 0``;
    otherwise ``ValueError`` is raised. The inputs are never modified.

    :raises TypeError: ``base`` or ``deltas`` is not a tuple.
    :raises ValueError: the level counts differ or the structure, ordering,
        duplicates, fields, cell bounds, z bounds, counts, coordinate sets or
        shared cell bounds of either input are bad, or a resulting tile
        violates ``zmin <= zmax`` or ``count >= 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")

    _validate_pyramid(base)
    _validate_pyramid_deltas(deltas)

    for level_tiles in base:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")

    if not deltas:
        return base

    if len(base) != len(deltas):
        raise ValueError("base and deltas must have the same number of levels")
    for base_level, delta_level in zip(base, deltas):
        if len(base_level) != len(delta_level):
            raise ValueError("base and deltas must contain the same (tx, ty) "
                             "tile coordinates at every level")
        for base_tile, delta_tile in zip(base_level, delta_level):
            if base_tile[0:2] != delta_tile[0:2]:
                raise ValueError("base and deltas must contain the same "
                                 "(tx, ty) tile coordinates at every level")
            if base_tile[2:6] != delta_tile[2:6]:
                raise ValueError(
                    "tiles with the same (tx, ty) must agree on "
                    "(ix0, iy0, ix1, iy1)"
                )

    result_levels = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for base_level, delta_level in zip(base, deltas):
            level_result = []
            for base_tile, delta_tile in zip(base_level, delta_level):
                tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count = base_tile
                dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                    delta_tile[8]

                base_zmin = Decimal(str(zmin))
                base_zmax = Decimal(str(zmax))
                dzmin_term = Decimal(str(dzmin))
                dzmax_term = Decimal(str(dzmax))
                if (base_zmin == base_zmin.to_integral_value()
                        and dzmin_term == dzmin_term.to_integral_value()):
                    new_zmin = int(base_zmin) - int(dzmin_term)
                else:
                    new_zmin = _quantize(base_zmin - dzmin_term)
                if (base_zmax == base_zmax.to_integral_value()
                        and dzmax_term == dzmax_term.to_integral_value()):
                    new_zmax = int(base_zmax) - int(dzmax_term)
                else:
                    new_zmax = _quantize(base_zmax - dzmax_term)
                new_count = count - dcount

                if isinstance(new_zmin, int) and isinstance(new_zmax, int):
                    ordered = new_zmin <= new_zmax
                else:
                    ordered = (Decimal(str(new_zmin))
                               <= Decimal(str(new_zmax)))
                if not ordered:
                    raise ValueError("resulting tile must satisfy zmin <= zmax")
                if new_count < 0:
                    raise ValueError("resulting count must be >= 0")

                level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                     new_zmin, new_zmax, new_count))
            result_levels.append(tuple(level_result))

    return tuple(result_levels)


def apply_tile_pyramid(base: tuple, deltas: tuple) -> tuple:
    """Add per-tile pyramid deltas to a base tile pyramid.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax`` are
    finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1`` and
    ``dzmin``/``dzmax`` are finite floats. At every level the two inputs must
    contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds; otherwise ``ValueError`` is raised. An
    empty ``deltas`` returns ``base``.

    Returns a tuple, in level order, of tuples in ``base`` order: the
    geometry is taken unchanged from each base tile and ``zmin``/``zmax``
    equal the base value plus the same-coordinate delta value, ``count`` the
    base count plus the same-coordinate ``dcount``. The float sums are Decimal
    computations (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``),
    quantized to six decimal places and converted to a finite float
    (integral results are still floats and negative zero is normalized to
    ``0.0``). Each resulting tile must satisfy ``zmin <= zmax`` and
    ``count >= 0``; otherwise ``ValueError`` is raised. The inputs are never
    modified, the result can be passed directly to :func:`query_tile_pyramid`
    and :func:`query_tile_pyramid_windows`, and the result does not depend on
    the order in which matching tiles are presented beyond the levels' stored
    order.

    :raises TypeError: ``base`` or ``deltas`` is not a tuple.
    :raises ValueError: the level counts differ or the structure, ordering,
        duplicates, fields, cell bounds, z bounds, counts, coordinate sets or
        shared cell bounds of either input are bad, or a resulting tile
        violates ``zmin <= zmax`` or ``count >= 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")

    _validate_pyramid(base)
    _validate_pyramid_deltas(deltas)

    for level_tiles in base:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")

    if not deltas:
        return base

    if len(base) != len(deltas):
        raise ValueError("base and deltas must have the same number of levels")
    for base_level, delta_level in zip(base, deltas):
        if len(base_level) != len(delta_level):
            raise ValueError("base and deltas must contain the same (tx, ty) "
                             "tile coordinates at every level")
        for base_tile, delta_tile in zip(base_level, delta_level):
            if base_tile[0:2] != delta_tile[0:2]:
                raise ValueError("base and deltas must contain the same "
                                 "(tx, ty) tile coordinates at every level")
            if base_tile[2:6] != delta_tile[2:6]:
                raise ValueError(
                    "tiles with the same (tx, ty) must agree on "
                    "(ix0, iy0, ix1, iy1)"
                )

    result_levels = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for base_level, delta_level in zip(base, deltas):
            level_result = []
            for base_tile, delta_tile in zip(base_level, delta_level):
                tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count = base_tile
                dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                    delta_tile[8]

                new_zmin = _quantize(
                    Decimal(str(zmin)) + Decimal(str(dzmin)))
                new_zmax = _quantize(
                    Decimal(str(zmax)) + Decimal(str(dzmax)))
                if (not math.isfinite(new_zmin)
                        or not math.isfinite(new_zmax)):
                    raise ValueError("resulting zmin and zmax must be finite")
                new_count = count + dcount

                if new_zmin > new_zmax:
                    raise ValueError("resulting tile must satisfy zmin <= zmax")
                if new_count < 0:
                    raise ValueError("resulting count must be >= 0")

                level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                     new_zmin, new_zmax, new_count))
            result_levels.append(tuple(level_result))

    return tuple(result_levels)


def rollback_tile_pyramid_windows(base: tuple, deltas: tuple,
                                  windows: tuple) -> tuple:
    """Subtract per-tile pyramid deltas and select tiles intersecting windows.

    Equivalent to
    ``query_tile_pyramid_windows(rollback_tile_pyramid_deltas(base, deltas), windows)``:
    every coordinate of ``base`` is rolled back against the same-coordinate
    tile of ``deltas`` exactly as in :func:`rollback_tile_pyramid_deltas`,
    then the rolled-back tiles are filtered per window.

    ``base`` and ``deltas`` must satisfy the full contract of
    :func:`rollback_tile_pyramid_deltas`: outer tuples with the same number of
    levels and, at every level, the same ``(tx, ty)`` coordinates with
    identical ``(ix0, iy0, ix1, iy1)`` bounds; an empty ``deltas`` queries
    ``base`` unchanged.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(base)``.

    For each window, a rolled-back tile matches when its closed cell-index
    intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of rolled-back
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples preserving
    ``base`` order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The rollback
    differences are Decimal computations (``Decimal(str(v))``, precision 50,
    ``ROUND_HALF_EVEN``) kept as exact ints when both operands are integral
    and otherwise quantized to six decimal places as floats (negative zero
    normalized). The inputs are never modified and the result does not depend
    on the order of ``deltas``.

    :raises TypeError: ``base``/``deltas``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, z bounds, counts, level counts or coordinate sets of
        ``base``/``deltas`` are bad, a resulting tile violates
        ``zmin <= zmax`` or ``count >= 0``, ``level`` is out of range, or the
        window bounds are inverted.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    rolled_back = rollback_tile_pyramid_deltas(base, deltas)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(rolled_back):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in rolled_back[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def apply_tile_pyramid_windows(base: tuple, deltas: tuple,
                               windows: tuple) -> tuple:
    """Add per-tile pyramid deltas and select tiles intersecting windows.

    Every coordinate of ``base`` is combined with the same-coordinate tile of
    ``deltas``: ``zmin``/``zmax`` equal the base value plus the
    same-coordinate delta value and ``count`` the base count plus the
    same-coordinate ``dcount``; the combined tiles are then filtered per
    window.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax`` are
    finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``dcount`` are non-bool ints with ``ix0 <= ix1`` and ``iy0 <= iy1`` and
    ``dzmin``/``dzmax`` are finite floats. At every level the two inputs must
    contain the same ``(tx, ty)`` coordinates with identical
    ``(ix0, iy0, ix1, iy1)`` bounds; otherwise ``ValueError`` is raised. An
    empty ``deltas`` queries ``base`` unchanged.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(base)``.

    For each window, a combined tile matches when its closed cell-index
    intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of combined
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples preserving
    ``base`` order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The additions are
    Decimal computations (``Decimal(str(v))``, precision 50,
    ``ROUND_HALF_EVEN``), quantized to six decimal places and converted to
    finite floats (integral results are still floats and negative zero is
    normalized to ``0.0``). Each resulting tile must satisfy
    ``zmin <= zmax`` and ``count >= 0``; otherwise ``ValueError`` is raised. The inputs are never
    modified and the result does not depend on the order of ``deltas``.

    :raises TypeError: ``base``/``deltas``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, z bounds, counts, level counts or coordinate sets of
        ``base``/``deltas`` are bad, a resulting tile violates
        ``zmin <= zmax`` or ``count >= 0``, ``level`` is out of range, or the
        window bounds are inverted.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid(base)
    _validate_pyramid_deltas(deltas)

    for level_tiles in base:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")

    if not deltas:
        applied = base
    else:
        if len(base) != len(deltas):
            raise ValueError("base and deltas must have the same number of "
                             "levels")
        for base_level, delta_level in zip(base, deltas):
            if len(base_level) != len(delta_level):
                raise ValueError("base and deltas must contain the same "
                                 "(tx, ty) tile coordinates at every level")
            for base_tile, delta_tile in zip(base_level, delta_level):
                if base_tile[0:2] != delta_tile[0:2]:
                    raise ValueError("base and deltas must contain the same "
                                     "(tx, ty) tile coordinates at every "
                                     "level")
                if base_tile[2:6] != delta_tile[2:6]:
                    raise ValueError(
                        "tiles with the same (tx, ty) must agree on "
                        "(ix0, iy0, ix1, iy1)"
                    )

        applied_levels = []
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            for base_level, delta_level in zip(base, deltas):
                level_result = []
                for base_tile, delta_tile in zip(base_level, delta_level):
                    tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count = base_tile
                    dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                        delta_tile[8]

                    new_zmin = _quantize(
                        Decimal(str(zmin)) + Decimal(str(dzmin)))
                    new_zmax = _quantize(
                        Decimal(str(zmax)) + Decimal(str(dzmax)))
                    if (not math.isfinite(new_zmin)
                            or not math.isfinite(new_zmax)):
                        raise ValueError(
                            "resulting zmin and zmax must be finite")
                    new_count = count + dcount

                    if new_zmin > new_zmax:
                        raise ValueError(
                            "resulting tile must satisfy zmin <= zmax")
                    if new_count < 0:
                        raise ValueError("resulting count must be >= 0")

                    level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                         new_zmin, new_zmax, new_count))
                applied_levels.append(tuple(level_result))

        applied = tuple(applied_levels)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(applied):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in applied[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def apply_tile_pyramid_deltas(base: tuple, deltas: tuple,
                              windows: tuple) -> tuple:
    """Add reorder-tolerant pyramid deltas and select tiles by windows.

    Every coordinate of ``base`` is combined with the same-coordinate tile of
    ``deltas``: ``zmin``/``zmax`` equal the base value plus the
    same-coordinate delta value and ``count`` the base count plus the
    same-coordinate ``dcount``; the combined tiles are then filtered per
    window.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax`` are
    finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples where the
    first six fields and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1`` and ``dzmin``/``dzmax`` are finite floats. At every level
    the two inputs must contain the same ``(tx, ty)`` coordinates with
    identical ``(ix0, iy0, ix1, iy1)`` bounds, but the delta tiles may be
    presented in any order — matching is by ``(tx, ty)`` coordinate;
    otherwise ``ValueError`` is raised. An empty ``deltas`` queries ``base``
    unchanged.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(base)``.

    For each window, a combined tile matches when its closed cell-index
    intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of combined
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples preserving
    ``base`` order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The additions are
    Decimal computations (``Decimal(str(v))``, precision 50,
    ``ROUND_HALF_EVEN``) with the summands accumulated in ascending Decimal
    order, quantized to six decimal places and converted to finite floats
    (integral results are still floats and negative zero is normalized to
    ``0.0``); ``count`` is summed exactly. Each resulting tile must satisfy
    ``zmin <= zmax`` and ``count >= 0``; otherwise ``ValueError`` is raised.
    The inputs are never modified and the result does not depend on the order
    of ``deltas``.

    :raises TypeError: ``base``/``deltas``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, z bounds, counts, level counts or coordinate sets of
        ``base``/``deltas`` are bad, a resulting tile violates
        ``zmin <= zmax`` or ``count >= 0``, ``level`` is out of range, or the
        window bounds are inverted.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid(base)

    for level_tiles in base:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")

    # The delta tiles may be reordered within each level, so they are
    # validated for structure, fields, bounds and duplicates here without
    # requiring a sorted order.
    for delta_level in deltas:
        if not isinstance(delta_level, tuple):
            raise ValueError("each deltas level must be a tuple")
        seen_keys = set()
        for tile in delta_level:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "dcount must be non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("dzmin and dzmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if key in seen_keys:
                raise ValueError("each deltas level must not contain "
                                 "duplicate (tx, ty) coordinates")
            seen_keys.add(key)

    if not deltas:
        applied = base
    else:
        if len(base) != len(deltas):
            raise ValueError("base and deltas must have the same number of "
                             "levels")

        applied_levels = []
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            for base_level, delta_level in zip(base, deltas):
                delta_by_coord = {}
                for delta_tile in delta_level:
                    delta_by_coord[(delta_tile[0], delta_tile[1])] = \
                        delta_tile
                if len(delta_by_coord) != len(base_level):
                    raise ValueError("base and deltas must contain the same "
                                     "(tx, ty) tile coordinates at every "
                                     "level")

                level_result = []
                for base_tile in base_level:
                    (tx, ty, ix0, iy0, ix1, iy1,
                     zmin, zmax, count) = base_tile
                    delta_tile = delta_by_coord.get((tx, ty))
                    if delta_tile is None:
                        raise ValueError(
                            "base and deltas must contain the same (tx, ty) "
                            "tile coordinates at every level"
                        )
                    if base_tile[2:6] != delta_tile[2:6]:
                        raise ValueError(
                            "tiles with the same (tx, ty) must agree on "
                            "(ix0, iy0, ix1, iy1)"
                        )
                    dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                        delta_tile[8]

                    new_zmin = _quantize(_sorted_sum(
                        [Decimal(str(zmin)), Decimal(str(dzmin))]))
                    new_zmax = _quantize(_sorted_sum(
                        [Decimal(str(zmax)), Decimal(str(dzmax))]))
                    if (not math.isfinite(new_zmin)
                            or not math.isfinite(new_zmax)):
                        raise ValueError(
                            "resulting zmin and zmax must be finite")
                    new_count = count + dcount

                    if new_zmin > new_zmax:
                        raise ValueError(
                            "resulting tile must satisfy zmin <= zmax")
                    if new_count < 0:
                        raise ValueError("resulting count must be >= 0")

                    level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                         new_zmin, new_zmax, new_count))
                applied_levels.append(tuple(level_result))

        applied = tuple(applied_levels)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(applied):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in applied[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def rollback_tile_pyramid_delta_windows(base: tuple, deltas: tuple,
                                        windows: tuple) -> tuple:
    """Subtract reorder-tolerant pyramid deltas and select tiles by windows.

    Every coordinate of ``base`` is rolled back against the same-coordinate
    tile of ``deltas``: ``zmin``/``zmax`` equal the base value minus the
    same-coordinate delta value and ``count`` the base count minus the
    same-coordinate ``dcount``; the rolled-back tiles are then filtered per
    window.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields and
    ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax`` are
    finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples where the
    first six fields and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1`` and ``dzmin``/``dzmax`` are finite floats. At every level
    the two inputs must contain the same ``(tx, ty)`` coordinates with
    identical ``(ix0, iy0, ix1, iy1)`` bounds, but the delta tiles may be
    presented in any order — matching is by ``(tx, ty)`` coordinate;
    otherwise ``ValueError`` is raised. An empty ``deltas`` queries ``base``
    unchanged.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(base)``.

    For each window, a rolled-back tile matches when its closed cell-index
    intervals intersect the window: ``tile.ix1 >= ix_min and
    tile.ix0 <= ix_max and tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of rolled-back
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples preserving
    ``base`` order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The differences are
    Decimal computations (``Decimal(str(v))``, precision 50,
    ``ROUND_HALF_EVEN``), quantized to six decimal places and converted to
    finite floats (integral results are still floats and negative zero is
    normalized to ``0.0``); ``count`` is subtracted exactly. Each resulting
    tile must satisfy ``zmin <= zmax`` and ``count >= 0``; otherwise
    ``ValueError`` is raised. The inputs are never modified and the result
    does not depend on the order of ``deltas``.

    :raises TypeError: ``base``/``deltas``/``windows`` is not a tuple or a
        window's container, length or field types are bad.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, z bounds, counts, level counts or coordinate sets of
        ``base``/``deltas`` are bad, a resulting tile violates
        ``zmin <= zmax`` or ``count >= 0``, ``level`` is out of range, or the
        window bounds are inverted.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_pyramid(base)

    for level_tiles in base:
        for tile in level_tiles:
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")

    # The delta tiles may be reordered within each level, so they are
    # validated for structure, fields, bounds and duplicates here without
    # requiring a sorted order.
    for delta_level in deltas:
        if not isinstance(delta_level, tuple):
            raise ValueError("each deltas level must be a tuple")
        seen_keys = set()
        for tile in delta_level:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "dcount must be non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("dzmin and dzmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if key in seen_keys:
                raise ValueError("each deltas level must not contain "
                                 "duplicate (tx, ty) coordinates")
            seen_keys.add(key)

    if not deltas:
        rolled_back = base
    else:
        if len(base) != len(deltas):
            raise ValueError("base and deltas must have the same number of "
                             "levels")

        rolled_back_levels = []
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            for base_level, delta_level in zip(base, deltas):
                delta_by_coord = {}
                for delta_tile in delta_level:
                    delta_by_coord[(delta_tile[0], delta_tile[1])] = \
                        delta_tile
                if len(delta_by_coord) != len(base_level):
                    raise ValueError("base and deltas must contain the same "
                                     "(tx, ty) tile coordinates at every "
                                     "level")

                level_result = []
                for base_tile in base_level:
                    (tx, ty, ix0, iy0, ix1, iy1,
                     zmin, zmax, count) = base_tile
                    delta_tile = delta_by_coord.get((tx, ty))
                    if delta_tile is None:
                        raise ValueError(
                            "base and deltas must contain the same (tx, ty) "
                            "tile coordinates at every level"
                        )
                    if base_tile[2:6] != delta_tile[2:6]:
                        raise ValueError(
                            "tiles with the same (tx, ty) must agree on "
                            "(ix0, iy0, ix1, iy1)"
                        )
                    dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                        delta_tile[8]

                    new_zmin = _quantize(
                        Decimal(str(zmin)) - Decimal(str(dzmin)))
                    new_zmax = _quantize(
                        Decimal(str(zmax)) - Decimal(str(dzmax)))
                    if (not math.isfinite(new_zmin)
                            or not math.isfinite(new_zmax)):
                        raise ValueError(
                            "resulting zmin and zmax must be finite")
                    new_count = count - dcount

                    if new_zmin > new_zmax:
                        raise ValueError(
                            "resulting tile must satisfy zmin <= zmax")
                    if new_count < 0:
                        raise ValueError("resulting count must be >= 0")

                    level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                         new_zmin, new_zmax, new_count))
                rolled_back_levels.append(tuple(level_result))

        rolled_back = tuple(rolled_back_levels)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(rolled_back):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in rolled_back[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def _validate_rollback_delta_windows_result(result) -> None:
    """Validate the tuple returned by
    :func:`rollback_tile_pyramid_delta_windows`.

    Every member must be a 6-tuple
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` whose first five fields
    are non-bool ints with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max`` and whose ``tiles`` is a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)``: the first six fields
    and ``count`` non-bool ints with ``count >= 0``, ``zmin``/``zmax`` finite
    floats with ``zmin <= zmax``, ``ix0 <= ix1``/``iy0 <= iy1`` and the tiles
    strictly sorted by ``(tx, ty)`` with no duplicates.
    """
    for window in result:
        if not isinstance(window, tuple) or len(window) != 6:
            raise ValueError(
                "each window must be a 6-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max, tiles)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a non-bool int")
        level, ix_min, iy_min, ix_max, iy_max, tiles = window
        if level < 0:
            raise ValueError("level must be >= 0")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        if not isinstance(tiles, tuple):
            raise ValueError("tiles must be a tuple")
        prev_key = None
        for tile in tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "tx, ty, ix0, iy0, ix1, iy1 and count must be "
                        "non-bool ints"
                    )
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("zmin and zmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            if tile[6] > tile[7]:
                raise ValueError("zmin must be <= zmax")
            if tile[8] < 0:
                raise ValueError("count must be >= 0")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("tiles must be sorted by (tx, ty) with no "
                                 "duplicate coordinates")
            prev_key = key


def encode_rollback_tile_pyramid_delta_windows(result: tuple) -> str:
    """Serialize the result of :func:`rollback_tile_pyramid_delta_windows`.

    ``result`` must be the tuple returned by
    :func:`rollback_tile_pyramid_delta_windows`: a tuple, in windows order,
    of strict 6-tuples ``(level, ix_min, iy_min, ix_max, iy_max, tiles)``
    whose first five fields are non-bool ints with ``level >= 0``,
    ``ix_min <= ix_max`` and ``iy_min <= iy_max`` and whose ``tiles`` is a
    tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` with the first six
    fields and ``count`` non-bool ints, ``count >= 0``, ``zmin``/``zmax``
    finite floats with ``zmin <= zmax``, ``ix0 <= ix1``/``iy0 <= iy1`` and
    the tiles strictly sorted by ``(tx, ty)`` with no duplicates.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in windows order) of six-value arrays, the last
    value being an array of nine-value tiles in the tiles' stored order.
    Integers are decimal; ``zmin``/``zmax`` use exactly six decimal places
    (negative zero written as ``0.000000``). The output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    result encodes as ``{"windows":[]}``. The input is never modified and
    repeated calls return identical results.

    :raises TypeError: ``result`` is not a tuple.
    :raises ValueError: the structure, field types, ordering, duplicates,
        cell bounds, level range, z bounds, counts or finiteness are bad.
    """
    if not isinstance(result, tuple):
        raise TypeError("result must be a tuple")
    _validate_rollback_delta_windows_result(result)
    return _format_pyramid_windows_text(result)


def decode_rollback_tile_pyramid_delta_windows(text: str) -> tuple:
    """Deserialize canonical JSON produced by
    :func:`encode_rollback_tile_pyramid_delta_windows`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``windows``; ``windows`` an array of strict six-value
    arrays ``[level, ix_min, iy_min, ix_max, iy_max, tiles]`` whose first five
    values are non-bool ints with ``level >= 0``,
    ``ix_min <= ix_max``/``iy_min <= iy_max`` and whose ``tiles`` is an array
    of strict nine-value tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count]`` with the first six
    fields and ``count`` non-bool ints, ``count >= 0``, ``zmin``/``zmax``
    finite floats with ``zmin <= zmax``, ``ix0 <= ix1``/``iy0 <= iy1`` and
    the tiles strictly sorted by ``(tx, ty)`` with no duplicates, and whose
    spelling is exactly canonical (integers in decimal, ``zmin``/``zmax``
    with six decimals, negative zero as ``0.000000``, no whitespace or extra
    keys, no ``NaN``/``Infinity``).

    Returns the outer windows-order tuple of 6-tuples (each ``tiles`` field a
    tuple of 9-tuples in the document's order); the empty document
    ``{"windows":[]}`` returns ``()``. The input text is never modified and
    repeated calls return identical results.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, key order, types, lengths, bounds,
        level range, z bounds, counts, ordering, duplicates, finiteness,
        numeric formatting or canonical re-encoding does not match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("text is not valid JSON") from exc

    if not isinstance(document, dict) or tuple(document) != ("windows",):
        raise ValueError("top-level value must be an object with exactly the "
                         "key 'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")

    windows: list[tuple] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each window must be an array of six values")
        raw_tiles = raw_window[5]
        if not isinstance(raw_tiles, list):
            raise ValueError("each window's tiles must be an array")
        tiles = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, list) or len(raw_tile) != 9:
                raise ValueError("each tile must be an array of nine values")
            tiles.append(tuple(raw_tile))
        windows.append(tuple(raw_window[:5]) + (tuple(tiles),))
    result = tuple(windows)

    # Structural rules: field types/finiteness, level range, window and tile
    # bounds, z bounds, counts, (tx, ty) sort order, no duplicates.
    _validate_rollback_delta_windows_result(result)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal z formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if _format_pyramid_windows_text(result) != text:
        raise ValueError("JSON text is not the canonical "
                         "rollback-delta-windows encoding")
    return result


def reconcile_tile_pyramid(base: tuple, deltas: tuple,
                           candidate: tuple) -> tuple:
    """Compare a candidate pyramid against base-plus-deltas expectations.

    Every coordinate of ``base`` is combined with the same-coordinate tile of
    ``deltas`` to form the expected tile: expected ``zmin``/``zmax`` equal the
    base value plus the same-coordinate delta value and expected ``count``
    the base count plus the same-coordinate ``dcount``. Each expected tile is
    then compared with the same-coordinate tile of ``candidate`` and the
    per-tile error (candidate minus expected) is returned.

    ``base`` must be an outer tuple as produced by :func:`build_tile_pyramid`:
    each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)`` 9-tuples sorted
    strictly by ``(tx, ty)`` with no duplicates, where the first six fields
    and ``count`` are non-bool ints with ``count >= 0``, ``zmin``/``zmax``
    are finite floats with ``zmin <= zmax`` and the bounds satisfy
    ``ix0 <= ix1`` and ``iy0 <= iy1``.

    ``deltas`` must be an outer tuple as produced by
    :func:`assess_tile_pyramid_deltas` with the same number of levels as
    ``base``: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)`` 9-tuples where the
    first six fields and ``dcount`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1`` and ``dzmin``/``dzmax`` are finite floats. At every level
    the delta tiles must contain the same ``(tx, ty)`` coordinates with
    identical ``(ix0, iy0, ix1, iy1)`` bounds as ``base``, but they may be
    presented in any order — matching is by ``(tx, ty)`` coordinate;
    otherwise ``ValueError`` is raised.

    ``candidate`` must be an outer tuple with the same structure as ``base``
    and the same number of levels; at every level it must contain the same
    ``(tx, ty)`` coordinates with identical ``(ix0, iy0, ix1, iy1)`` bounds
    as ``base``, otherwise ``ValueError`` is raised.

    Returns a tuple, in ``base`` level and tile order, of tuples of
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)`` 9-tuples
    where ``err_zmin``/``err_zmax`` equal the candidate value minus the
    expected value and ``err_count`` the candidate count minus the expected
    count. The expectations are Decimal computations
    (``Decimal(str(v))``, precision 50, ``ROUND_HALF_EVEN``) and the errors
    are quantized to six decimal places and converted to floats (negative
    zero is normalized to ``0.0``); the counts are exact. An expected tile
    with ``zmin > zmax`` or ``count < 0`` raises ``ValueError``. The inputs
    are never modified and the result does not depend on the order of
    ``deltas``.

    :raises TypeError: ``base``/``deltas``/``candidate`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds, z bounds, counts, level counts or coordinate sets of
        ``base``/``deltas``/``candidate`` are bad, or an expected tile
        violates ``zmin <= zmax`` or ``count >= 0``.
    """
    if not isinstance(base, tuple):
        raise TypeError("base must be a tuple")
    if not isinstance(deltas, tuple):
        raise TypeError("deltas must be a tuple")
    if not isinstance(candidate, tuple):
        raise TypeError("candidate must be a tuple")

    _validate_pyramid(base)
    _validate_pyramid(candidate)

    for pyramid in (base, candidate):
        for level_tiles in pyramid:
            for tile in level_tiles:
                if tile[4] < tile[2] or tile[5] < tile[3]:
                    raise ValueError("tile bounds must satisfy ix0 <= ix1 "
                                     "and iy0 <= iy1")
                if tile[6] > tile[7]:
                    raise ValueError("zmin must be <= zmax")
                if tile[8] < 0:
                    raise ValueError("count must be >= 0")

    # The delta tiles may be reordered within each level, so they are
    # validated for structure, fields, bounds and duplicates here without
    # requiring a sorted order.
    for delta_level in deltas:
        if not isinstance(delta_level, tuple):
            raise ValueError("each deltas level must be a tuple")
        seen_keys = set()
        for tile in delta_level:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, dzmin, dzmax, dcount)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "dcount must be non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("dzmin and dzmax must be finite floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if key in seen_keys:
                raise ValueError("each deltas level must not contain "
                                 "duplicate (tx, ty) coordinates")
            seen_keys.add(key)

    if len(base) != len(deltas):
        raise ValueError("base and deltas must have the same number of "
                         "levels")
    if len(base) != len(candidate):
        raise ValueError("base and candidate must have the same number of "
                         "levels")

    reconciled_levels = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for base_level, delta_level, candidate_level in zip(base, deltas,
                                                            candidate):
            delta_by_coord = {}
            for delta_tile in delta_level:
                delta_by_coord[(delta_tile[0], delta_tile[1])] = delta_tile
            if len(delta_by_coord) != len(base_level):
                raise ValueError("base and deltas must contain the same "
                                 "(tx, ty) tile coordinates at every level")
            if len(candidate_level) != len(base_level):
                raise ValueError("base and candidate must contain the same "
                                 "(tx, ty) tile coordinates at every level")

            level_result = []
            for base_tile, candidate_tile in zip(base_level,
                                                 candidate_level):
                (tx, ty, ix0, iy0, ix1, iy1,
                 zmin, zmax, count) = base_tile
                if candidate_tile[0:2] != (tx, ty):
                    raise ValueError(
                        "base and candidate must contain the same (tx, ty) "
                        "tile coordinates at every level"
                    )
                if candidate_tile[2:6] != base_tile[2:6]:
                    raise ValueError(
                        "tiles with the same (tx, ty) must agree on "
                        "(ix0, iy0, ix1, iy1)"
                    )
                delta_tile = delta_by_coord.get((tx, ty))
                if delta_tile is None:
                    raise ValueError(
                        "base and deltas must contain the same (tx, ty) "
                        "tile coordinates at every level"
                    )
                if delta_tile[2:6] != base_tile[2:6]:
                    raise ValueError(
                        "tiles with the same (tx, ty) must agree on "
                        "(ix0, iy0, ix1, iy1)"
                    )
                dzmin, dzmax, dcount = delta_tile[6], delta_tile[7], \
                    delta_tile[8]

                expected_zmin = Decimal(str(zmin)) + Decimal(str(dzmin))
                expected_zmax = Decimal(str(zmax)) + Decimal(str(dzmax))
                expected_count = count + dcount
                if expected_zmin > expected_zmax:
                    raise ValueError(
                        "expected tile must satisfy zmin <= zmax")
                if expected_count < 0:
                    raise ValueError("expected count must be >= 0")

                err_zmin = _quantize(
                    Decimal(str(candidate_tile[6])) - expected_zmin)
                err_zmax = _quantize(
                    Decimal(str(candidate_tile[7])) - expected_zmax)
                err_count = candidate_tile[8] - expected_count

                level_result.append((tx, ty, ix0, iy0, ix1, iy1,
                                     err_zmin, err_zmax, err_count))
            reconciled_levels.append(tuple(level_result))

    return tuple(reconciled_levels)


def _validate_reconciliation(assessment) -> None:
    """Validate the outer tuple returned by :func:`reconcile_tile_pyramid`.

    Every level must be a tuple of strict 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)``: the
    first six fields and ``err_count`` are non-bool ints with
    ``ix0 <= ix1`` and ``iy0 <= iy1``;
    ``err_zmin``/``err_zmax`` are finite floats; and the tiles are sorted
    strictly by ``(tx, ty)`` with no duplicates.
    """
    for level_tiles in assessment:
        if not isinstance(level_tiles, tuple):
            raise ValueError("each reconciliation level must be a tuple")
        prev_key = None
        for tile in level_tiles:
            if not isinstance(tile, tuple) or len(tile) != 9:
                raise ValueError(
                    "each tile must be a 9-tuple "
                    "(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, "
                    "err_count)"
                )
            for value in tile[0:6] + (tile[8],):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("tx, ty, ix0, iy0, ix1, iy1 and "
                                     "err_count must be non-bool ints")
            for value in tile[6:8]:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("err_zmin and err_zmax must be finite "
                                     "floats")
            if tile[4] < tile[2] or tile[5] < tile[3]:
                raise ValueError("tile bounds must satisfy ix0 <= ix1 and "
                                 "iy0 <= iy1")
            key = (tile[0], tile[1])
            if prev_key is not None and key <= prev_key:
                raise ValueError("each reconciliation level must be sorted "
                                 "by (tx, ty) with no duplicate coordinates")
            prev_key = key


def encode_tile_reconciliation(assessment: tuple) -> str:
    """Serialize the result of :func:`reconcile_tile_pyramid`.

    ``assessment`` must be the outer tuple returned by
    :func:`reconcile_tile_pyramid`: each level is a tuple of 9-tuples
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)`` with
    tiles sorted strictly by ``(tx, ty)`` and no duplicates, where the
    first six fields and ``err_count`` are non-bool ints satisfying
    ``ix0 <= ix1`` and ``iy0 <= iy1``, and ``err_zmin``/``err_zmax`` are
    finite floats.

    Returns a canonical compact JSON string whose sole top-level key is
    ``levels``: an array (in level order) of arrays of tile arrays in the
    tiles' input order. Integers are decimal; ``err_zmin``/``err_zmax`` use
    exactly six decimal places (negative zero written as ``0.000000``). The
    output has no whitespace, ASCII is not escaped and ``NaN``/``Infinity``
    never appear. An empty assessment encodes as ``{"levels":[]}``. The
    input is never modified.

    :raises TypeError: ``assessment`` is not a tuple.
    :raises ValueError: the structure, ordering, duplicates, fields, cell
        bounds or error values are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    _validate_reconciliation(assessment)

    parts = ['{"levels":[']
    for level_index, level_tiles in enumerate(assessment):
        if level_index:
            parts.append(",")
        parts.append("[")
        for tile_index, tile in enumerate(level_tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1,
             err_zmin, err_zmax, err_count) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _format_z(err_zmin),
                                   _format_z(err_zmax), str(err_count))))
            parts.append("]")
        parts.append("]")
    parts.append(']}')
    return "".join(parts)


def decode_tile_reconciliation(text: str) -> tuple:
    """Deserialize canonical JSON produced by :func:`encode_tile_reconciliation`.

    The document must be the compact encoder output: an object whose sole
    top-level key is ``levels``; ``levels`` an array (in level order) of
    arrays of strict 9-item tiles
    ``[tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count]`` whose
    first six fields and ``err_count`` are non-bool ints with
    ``ix0 <= ix1`` and ``iy0 <= iy1``, whose ``err_zmin``/``err_zmax`` are
    finite floats, and whose ``(tx, ty)`` coordinates are strictly
    increasing with no duplicates within each level, and whose spelling is
    exactly canonical (integers in decimal, errors with six decimals,
    negative zero as ``0.000000``, no whitespace or extra keys, no
    ``NaN``/``Infinity``).

    Returns the outer level-order tuple of tuples of 9-tuples in the
    document's order; the empty document ``{"levels":[]}`` returns ``()``.
    The input text is never modified.

    :raises TypeError: ``text`` is not a ``str``.
    :raises ValueError: the JSON syntax, top-level keys/types, level/tile
        shape, field types, bounds, ordering, duplicates, non-finite
        values, numeric formatting or canonical re-encoding does not match.
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
    assessment = tuple(levels)

    # Structural rules: tuple levels, field types/finiteness, cell bounds,
    # (tx, ty) sort order and no duplicates.
    _validate_reconciliation(assessment)

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal error formatting, leading zeros, -0,
    # exponents and any other non-canonical spelling.
    if encode_tile_reconciliation(assessment) != text:
        raise ValueError("JSON text is not the canonical reconciliation "
                         "encoding")
    return assessment


def query_tile_reconciliation_windows(assessment: tuple,
                                      windows: tuple) -> tuple:
    """Select reconciliation tiles intersecting cell-index windows across levels.

    ``assessment`` must be the outer tuple returned by
    :func:`reconcile_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)`` 9-tuples
    sorted strictly by ``(tx, ty)`` with no duplicates, where the first six
    fields and ``err_count`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1``, and ``err_zmin``/``err_zmax`` are finite floats.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    For each window, a tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a tuple, in ``windows`` order, of
    ``(level, ix_min, iy_min, ix_max, iy_max, tiles)`` tuples where ``tiles``
    is a tuple of the stored 9-tuples at that level, preserving the level's
    existing order; a window with no matching tile gets an empty ``tiles``
    tuple and an empty ``windows`` tuple returns ``()``. The input is never
    modified and repeated calls return identical results.

    :raises TypeError: ``assessment``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: ``level`` is out of range, the window bounds are
        inverted, or the assessment's structure, ordering, duplicates, fields,
        cell bounds or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_reconciliation(assessment)

    results = []
    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0 or level >= len(assessment):
            raise ValueError("level out of range")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")
        matched = tuple(
            tile for tile in assessment[level]
            if (tile[4] >= ix_min and tile[2] <= ix_max
                and tile[5] >= iy_min and tile[3] <= iy_max)
        )
        results.append((level, ix_min, iy_min, ix_max, iy_max, matched))

    return tuple(results)


def encode_reconciliation_windows(assessment: tuple, windows: tuple) -> str:
    """Encode windowed reconciliation summaries as canonical compact JSON.

    ``assessment`` must be the outer tuple returned by
    :func:`reconcile_tile_pyramid`: each level is a tuple of
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)`` 9-tuples
    sorted strictly by ``(tx, ty)`` with no duplicates, where the first six
    fields and ``err_count`` are non-bool ints with ``ix0 <= ix1`` and
    ``iy0 <= iy1``, and ``err_zmin``/``err_zmax`` are finite floats.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max``; ``level`` must satisfy
    ``0 <= level < len(assessment)``.

    For each window, a tile matches when its closed cell-index intervals
    intersect the window: ``tile.ix1 >= ix_min and tile.ix0 <= ix_max and
    tile.iy1 >= iy_min and tile.iy0 <= iy_max``.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in ``windows`` order) of six-value arrays
    ``[level, ix_min, iy_min, ix_max, iy_max, summary]``. A window with no
    matching tile gets ``summary = null``; otherwise ``summary`` is the
    five-value array
    ``[err_zmin_min, err_zmax_max, z_rmse, abs_count_sum, match_count]``
    where ``err_zmin_min``/``err_zmax_max`` are the extremes of the matched
    tiles' ``err_zmin``/``err_zmax``,
    ``z_rmse = sqrt(sum(err_zmin**2 + err_zmax**2) / (2 * match_count))``,
    ``abs_count_sum`` is the exact int sum of ``abs(err_count)`` and
    ``match_count`` is the number of matched tiles. The float values are
    Decimal computations (each input converted via ``Decimal(str(v))``,
    precision 50, ``ROUND_HALF_EVEN``) with the square terms added in
    ascending Decimal order, quantized to six decimal places. Integers are
    decimal; floats use exactly six decimal places (negative zero written as
    ``0.000000``). The output has no whitespace, ASCII is not escaped and
    ``NaN``/``Infinity`` never appear. An empty ``windows`` tuple encodes as
    ``{"windows":[]}``. The inputs are never modified and repeated calls
    return byte-identical strings.

    :raises TypeError: ``assessment``/``windows`` is not a tuple or a window's
        container, length or field types are bad.
    :raises ValueError: ``level`` is out of range, the window bounds are
        inverted, or the assessment's structure, ordering, duplicates, fields,
        cell bounds or finiteness are bad.
    """
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    _validate_reconciliation(assessment)

    parts = ['{"windows":[']
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for window_index, (level, ix_min, iy_min, ix_max,
                           iy_max) in enumerate(windows):
            if level < 0 or level >= len(assessment):
                raise ValueError("level out of range")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("window bounds must satisfy "
                                 "ix_min <= ix_max and iy_min <= iy_max")

            err_zmins = []
            err_zmaxs = []
            square_terms = []
            abs_count_sum = 0
            match_count = 0
            for tile in assessment[level]:
                if (tile[4] >= ix_min and tile[2] <= ix_max
                        and tile[5] >= iy_min and tile[3] <= iy_max):
                    err_zmin = Decimal(str(tile[6]))
                    err_zmax = Decimal(str(tile[7]))
                    err_zmins.append(err_zmin)
                    err_zmaxs.append(err_zmax)
                    square_terms.append(err_zmin * err_zmin)
                    square_terms.append(err_zmax * err_zmax)
                    abs_count_sum += abs(tile[8])
                    match_count += 1

            if window_index:
                parts.append(",")
            parts.append("[")
            parts.append(",".join((str(level), str(ix_min), str(iy_min),
                                   str(ix_max), str(iy_max))))
            parts.append(",")
            if not match_count:
                parts.append("null")
            else:
                z_rmse = (_sorted_sum(square_terms)
                          / Decimal(2 * match_count)).sqrt()
                parts.append("[")
                parts.append(",".join((_format_z(_quantize(min(err_zmins))),
                                       _format_z(_quantize(max(err_zmaxs))),
                                       _format_z(_quantize(z_rmse)),
                                       str(abs_count_sum),
                                       str(match_count))))
                parts.append("]")
            parts.append("]")
    parts.append("]}")
    return "".join(parts)


def _format_decimal6(value: Decimal) -> str:
    """Format a Decimal with exactly six decimals (``-0`` normalized)."""
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        text = format(value.quantize(_QUANTUM), "f")
    return "0.000000" if text == "-0.000000" else text


def _decode_reconciliation_state(text: str, windows: tuple) -> list:
    """Parse and validate a canonical reconciliation accumulation state.

    Returns a list aligned with ``windows`` whose entries are ``None`` or
    ``[emin, emax, qsum, asum, n]`` where ``emin``/``emax`` are Decimals and
    the other fields are ints.

    :raises ValueError: the JSON syntax or shape is bad, the text is not the
        canonical encoding, or the window keys do not match ``windows`` in
        order.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("state is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"windows"}:
        raise ValueError("state top-level value must be an object with only "
                         "'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list) or len(raw_windows) != len(windows):
        raise ValueError("state must contain one entry per window in the "
                         "same order")

    entries = []
    for raw_window, expected in zip(raw_windows, windows):
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each state window must be an array of six "
                             "values")
        key = raw_window[:5]
        for value in key:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("state window keys must be integers")
        if tuple(key) != tuple(expected):
            raise ValueError("state window keys must match windows in order")

        raw_summary = raw_window[5]
        if raw_summary is None:
            entries.append(None)
            continue
        if not isinstance(raw_summary, list) or len(raw_summary) != 5:
            raise ValueError("each state summary must be null or an array "
                             "of five values")
        emin, emax = raw_summary[0], raw_summary[1]
        qsum, asum, match_count = raw_summary[2:5]
        for value in (emin, emax):
            if isinstance(value, bool) or not isinstance(value,
                                                         (int, Decimal)):
                raise ValueError("state emin/emax must be numbers")
        for value in (qsum, asum, match_count):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("state qsum, asum and n must be integers")
        if match_count < 1 or qsum < 0 or asum < 0:
            raise ValueError("state summary requires n >= 1, qsum >= 0 and "
                             "asum >= 0")
        entries.append([Decimal(emin), Decimal(emax),
                        qsum, asum, match_count])

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal error formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    try:
        canonical = _format_reconciliation_state(entries, windows)
    except InvalidOperation as exc:
        raise ValueError("state is not the canonical reconciliation "
                         "accumulation encoding") from exc
    if canonical != text:
        raise ValueError("state is not the canonical reconciliation "
                         "accumulation encoding")
    return entries


def _format_reconciliation_state(entries: list, windows: tuple) -> str:
    """Serialize accumulation entries aligned with ``windows`` to JSON."""
    parts = ['{"windows":[']
    for index, (window, entry) in enumerate(zip(windows, entries)):
        if index:
            parts.append(",")
        level, ix_min, iy_min, ix_max, iy_max = window
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if entry is None:
            parts.append("null")
        else:
            emin, emax, qsum, asum, match_count = entry
            parts.append("[")
            parts.append(",".join((_format_decimal6(emin),
                                   _format_decimal6(emax),
                                   str(qsum), str(asum), str(match_count))))
            parts.append("]")
        parts.append("]")
    parts.append("]}")
    return "".join(parts)


def accumulate_reconciliation(state, assessment: tuple,
                              windows: tuple) -> str:
    """Accumulate windowed reconciliation error summaries across batches.

    ``assessment`` and ``windows`` follow exactly the validation, exceptions
    and closed-interval intersection contract of
    :func:`query_tile_reconciliation_windows`: ``assessment`` is the outer
    reconciliation tuple of levels of
    ``(tx, ty, ix0, iy0, ix1, iy1, err_zmin, err_zmax, err_count)`` 9-tuples
    and each window is a ``(level, ix_min, iy_min, ix_max, iy_max)``
    5-tuple; a tile matches when its cell-index intervals intersect the
    window.

    ``state`` is either ``None`` (the first batch) or a previous return
    value of this function. The state must be canonical JSON and its window
    keys must equal ``windows`` in order.

    For each window the summary is ``S = [emin, emax, qsum, asum, n]`` where
    each matched error ``v`` is converted via ``Decimal(str(v))`` and, at
    precision 50 with ``ROUND_HALF_EVEN``, quantized to six decimal places
    and multiplied by ``10 ** 6`` to give the integer micro-unit ``u``;
    ``emin`` is the minimum quantized ``err_zmin``, ``emax`` the maximum
    quantized ``err_zmax``, ``qsum`` accumulates ``u ** 2`` for both errors,
    ``asum`` is the sum of ``abs(err_count)`` and ``n`` is the match count.
    Accumulation merges the extremes and the three integer totals with the
    prior state; a window with no matches leaves its prior entry unchanged,
    and ``None`` initializes every window with ``null``.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in ``windows`` order) whose values are the five
    window fields followed by ``S`` (``null`` when nothing has matched).
    Integers are decimal; ``emin``/``emax`` use exactly six decimal places
    (negative zero written as ``0.000000``); the output has no whitespace,
    ASCII is not escaped and ``NaN``/``Infinity`` never appear. An empty
    ``windows`` tuple returns ``{"windows":[]}``. Splitting one batch into
    several batches or presenting them in another order gives byte-identical
    results. The inputs are never modified.

    :raises TypeError: ``state`` is neither ``None`` nor a ``str``, or
        ``assessment``/``windows`` is not a tuple (including bad window
        container, length or field types).
    :raises ValueError: ``state`` is not canonical or its window keys do not
        match ``windows`` in order, ``level`` is out of range, the window
        bounds are inverted, or the assessment's structure, ordering,
        duplicates, fields, cell bounds or finiteness are bad.
    """
    if state is not None and not isinstance(state, str):
        raise TypeError("state must be None or a str")
    if not isinstance(assessment, tuple):
        raise TypeError("assessment must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    if state is None:
        entries = [None] * len(windows)
    else:
        entries = _decode_reconciliation_state(state, windows)

    _validate_reconciliation(assessment)

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for index, (level, ix_min, iy_min, ix_max,
                    iy_max) in enumerate(windows):
            if level < 0 or level >= len(assessment):
                raise ValueError("level out of range")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("window bounds must satisfy ix_min <= ix_max "
                                 "and iy_min <= iy_max")

            err_zmins = []
            err_zmaxs = []
            qsum = 0
            asum = 0
            match_count = 0
            for tile in assessment[level]:
                if (tile[4] >= ix_min and tile[2] <= ix_max
                        and tile[5] >= iy_min and tile[3] <= iy_max):
                    err_zmin = Decimal(str(tile[6])).quantize(_QUANTUM)
                    err_zmax = Decimal(str(tile[7])).quantize(_QUANTUM)
                    err_zmins.append(err_zmin)
                    err_zmaxs.append(err_zmax)
                    u_min = int(err_zmin * _MICRO)
                    u_max = int(err_zmax * _MICRO)
                    qsum += u_min * u_min + u_max * u_max
                    asum += abs(tile[8])
                    match_count += 1

            if not match_count:
                continue

            emin = min(err_zmins)
            emax = max(err_zmaxs)
            previous = entries[index]
            if previous is None:
                entries[index] = [emin, emax, qsum, asum, match_count]
            else:
                previous[0] = min(previous[0], emin)
                previous[1] = max(previous[1], emax)
                previous[2] += qsum
                previous[3] += asum
                previous[4] += match_count

    return _format_reconciliation_state(entries, windows)


def merge_reconciliation_states(states: tuple, windows: tuple) -> str:
    """Merge canonical reconciliation accumulation states.

    ``states`` must be a tuple of ``str`` values, each a return value of
    :func:`accumulate_reconciliation` for the same ``windows``. ``windows``
    must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max`` and ``level >= 0``.

    Each state must be the canonical compact JSON whose sole top-level key is
    ``windows`` with one entry per window in ``windows`` order: the five window
    fields followed by ``S``, where ``S`` is either ``null`` or
    ``[emin, emax, qsum, asum, n]`` with ``emin``/``emax`` fixed six-decimal
    finite numbers (negative zero written as ``0.000000``) and ``qsum``,
    ``asum`` and ``n`` non-bool non-negative decimal integers with ``n > 0``;
    re-encoding the parsed state must reproduce it byte for byte.

    Entries are merged per window: a ``null`` summary contributes nothing, so
    a window whose summaries are all ``null`` stays ``null``; otherwise
    ``emin`` is the minimum ``emin``, ``emax`` the maximum ``emax`` and
    ``qsum``, ``asum`` and ``n`` are the exact integer sums. An empty
    ``states`` tuple yields ``null`` for every window.

    Returns a canonical compact JSON string in the same format as the input
    states; an empty ``windows`` tuple returns ``{"windows":[]}``. The result
    does not depend on the order or grouping of ``states`` and the inputs are
    never modified.

    :raises TypeError: ``states``/``windows`` is not a tuple, a state member
        is not a ``str``, or a window's container, length or field types are
        bad.
    :raises ValueError: ``level`` is negative, the window bounds are inverted,
        or a state's JSON, structure, values, window keys or canonical
        formatting are bad.
    """
    if not isinstance(states, tuple):
        raise TypeError("states must be a tuple")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    for state in states:
        if not isinstance(state, str):
            raise TypeError("each state must be a str")

    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0:
            raise ValueError("level must be non-negative")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")

    merged = [None] * len(windows)
    for state in states:
        entries = _decode_reconciliation_state(state, windows)
        for index, entry in enumerate(entries):
            if entry is None:
                continue
            current = merged[index]
            if current is None:
                # Copy so the decoded state's list is never mutated.
                merged[index] = list(entry)
            else:
                current[0] = min(current[0], entry[0])
                current[1] = max(current[1], entry[1])
                current[2] += entry[2]
                current[3] += entry[3]
                current[4] += entry[4]

    return _format_reconciliation_state(merged, windows)


def finalize_reconciliation_state(state: str, windows: tuple) -> str:
    """Finalize a merged reconciliation accumulation state as a report.

    ``state`` must be canonical JSON in the
    :func:`merge_reconciliation_states` format: the compact document whose
    sole top-level key is ``windows`` with one entry per window in
    ``windows`` order, each entry being the five window fields followed by
    either ``null`` or ``[emin, emax, qsum, asum, n]`` with
    ``emin``/``emax`` fixed six-decimal finite numbers (negative zero written
    as ``0.000000``) and ``qsum``, ``asum`` and ``n`` non-bool decimal
    integers with ``qsum >= 0``, ``asum >= 0`` and ``n > 0``; re-encoding the
    parsed state must reproduce it byte for byte.

    ``windows`` must be a tuple of 5-tuples
    ``(level, ix_min, iy_min, ix_max, iy_max)`` whose fields are non-bool ints
    with ``ix_min <= ix_max`` and ``iy_min <= iy_max`` and ``level >= 0``.

    Returns a canonical compact JSON string whose sole top-level key is
    ``windows``: an array (in ``windows`` order) of six-value arrays
    ``[level, ix_min, iy_min, ix_max, iy_max, report]``. A window whose
    summary is ``null`` gets ``report = null``; otherwise ``report`` is the
    five-value array ``[emin, emax, rmse, amean, n]`` where
    ``rmse = sqrt(qsum / (2 * n)) / 10 ** 6`` and ``amean = asum / n``. Every
    value except ``n`` is a Decimal computation (precision 50,
    ``ROUND_HALF_EVEN``) written with exactly six decimal places (negative
    zero written as ``0.000000``); ``n`` is a decimal integer. The output has
    no whitespace, ASCII is not escaped and ``NaN``/``Infinity`` never
    appear. An empty ``windows`` tuple returns ``{"windows":[]}``. The inputs
    are never modified and repeated calls return byte-identical strings.

    :raises TypeError: ``state`` is not a ``str`` or ``windows`` is not a
        tuple (including bad window container, length or field types).
    :raises ValueError: ``level`` is negative, the window bounds are
        inverted, or the state's JSON, structure, values, window keys or
        canonical formatting are bad.
    """
    if not isinstance(state, str):
        raise TypeError("state must be a str")
    if not isinstance(windows, tuple):
        raise TypeError("windows must be a tuple")

    for window in windows:
        if not isinstance(window, tuple) or len(window) != 5:
            raise TypeError(
                "each window must be a 5-tuple "
                "(level, ix_min, iy_min, ix_max, iy_max)"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), window):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-bool int")

    for level, ix_min, iy_min, ix_max, iy_max in windows:
        if level < 0:
            raise ValueError("level must be non-negative")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("window bounds must satisfy ix_min <= ix_max "
                             "and iy_min <= iy_max")

    entries = _decode_reconciliation_state(state, windows)

    parts = ['{"windows":[']
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for index, (window, entry) in enumerate(zip(windows, entries)):
            if index:
                parts.append(",")
            level, ix_min, iy_min, ix_max, iy_max = window
            parts.append("[")
            parts.append(",".join((str(level), str(ix_min), str(iy_min),
                                   str(ix_max), str(iy_max))))
            parts.append(",")
            if entry is None:
                parts.append("null")
            else:
                emin, emax, qsum, asum, match_count = entry
                rmse = ((Decimal(qsum) / Decimal(2 * match_count)).sqrt()
                        / _MICRO)
                amean = Decimal(asum) / Decimal(match_count)
                parts.append("[")
                parts.append(",".join((_format_decimal6(emin),
                                       _format_decimal6(emax),
                                       _format_decimal6(rmse),
                                       _format_decimal6(amean),
                                       str(match_count))))
                parts.append("]")
            parts.append("]")
    parts.append("]}")
    return "".join(parts)


def _decode_gate_report(text: str) -> list:
    """Parse and validate a canonical finalized reconciliation report.

    Returns the report windows as a list of
    ``[level, ix_min, iy_min, ix_max, iy_max, summary]`` where ``summary`` is
    ``None`` or ``[emin, emax, rmse, amean, n]`` with the four metrics as
    Decimals and ``n`` an int.

    :raises ValueError: the JSON syntax or shape is bad, a value violates the
        report contract, or the text is not the canonical encoding.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("report is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"windows"}:
        raise ValueError("report top-level value must be an object with only "
                         "'windows'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("report 'windows' must be an array")

    windows = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 6:
            raise ValueError("each report window must be an array of six "
                             "values")
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), raw_window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"report {name} must be a non-bool integer")
        level, ix_min, iy_min, ix_max, iy_max = raw_window[:5]
        if level < 0:
            raise ValueError("report level must be non-negative")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("report window bounds must satisfy ix_min <= "
                             "ix_max and iy_min <= iy_max")

        raw_summary = raw_window[5]
        summary = None
        if raw_summary is not None:
            if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                raise ValueError("each report summary must be null or an "
                                 "array of five values")
            metrics_raw = list(raw_summary[:4])
            match_count = raw_summary[4]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics_raw):
                if isinstance(value, bool) or not isinstance(value,
                                                             (int, Decimal)):
                    raise ValueError(f"report {name} must be a number")
            metrics = [Decimal(value) for value in metrics_raw]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics):
                if not value.is_finite():
                    raise ValueError(f"report {name} must be finite")
            if isinstance(match_count, bool) or not isinstance(match_count,
                                                               int):
                raise ValueError("report n must be a non-bool integer")
            if match_count < 1:
                raise ValueError("report n must be a positive integer")
            if metrics[2] < 0 or metrics[3] < 0:
                raise ValueError("report rmse and amean must be "
                                 "non-negative")
            summary = [*metrics, match_count]

        windows.append([level, ix_min, iy_min, ix_max, iy_max, summary])

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    canonical = _format_finalized_report(windows)
    if canonical != text:
        raise ValueError("report is not the canonical reconciliation "
                         "finalization encoding")
    return windows


def _format_report_windows_array(windows: list) -> str:
    """Serialize report windows as a compact JSON array body."""
    parts = ["["]
    for index, window in enumerate(windows):
        if index:
            parts.append(",")
        level, ix_min, iy_min, ix_max, iy_max, summary = window
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if summary is None:
            parts.append("null")
        else:
            emin, emax, rmse, amean, match_count = summary
            parts.append("[")
            parts.append(",".join((_format_decimal6(emin),
                                   _format_decimal6(emax),
                                   _format_decimal6(rmse),
                                   _format_decimal6(amean),
                                   str(match_count))))
            parts.append("]")
        parts.append("]")
    parts.append("]")
    return "".join(parts)


def _format_finalized_report(windows: list) -> str:
    """Serialize report windows in the finalization (input) format."""
    return ('{"windows":' + _format_report_windows_array(windows) + "}")


def _format_gate_report(windows: list, passed_flags: list,
                        passed: bool) -> str:
    """Serialize gated report windows to the compact two-key JSON document."""
    parts = ['{"windows":[']
    for index, (window, window_passed) in enumerate(zip(windows,
                                                        passed_flags)):
        if index:
            parts.append(",")
        level, ix_min, iy_min, ix_max, iy_max, summary = window
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if summary is None:
            parts.append("null")
        else:
            emin, emax, rmse, amean, match_count = summary
            parts.append("[")
            parts.append(",".join((_format_decimal6(emin),
                                   _format_decimal6(emax),
                                   _format_decimal6(rmse),
                                   _format_decimal6(amean),
                                   str(match_count))))
            parts.append("]")
        parts.append(",true]" if window_passed else ",false]")
    parts.append('],"passed":')
    parts.append("true}" if passed else "false}")
    return "".join(parts)


def gate_report(report: str, limits: tuple) -> str:
    """Apply acceptance limits to a finalized reconciliation report.

    ``report`` must be canonical JSON in the
    :func:`finalize_reconciliation_state` format: the compact document whose
    sole top-level key is ``windows`` with one six-value array
    ``[level, ix_min, iy_min, ix_max, iy_max, R]`` per window. The five window
    fields are non-bool integers with ``level >= 0``, ``ix_min <= ix_max`` and
    ``iy_min <= iy_max``; ``R`` is either ``null`` or
    ``[emin, emax, rmse, amean, n]`` where the four metrics are finite
    numbers written with exactly six decimal places (negative zero written as
    ``0.000000``), ``rmse``/``amean`` are non-negative and ``n`` is a positive
    non-bool decimal integer; re-encoding the parsed report must reproduce it
    byte for byte.

    ``limits`` is a 4-tuple ``(max_abs, max_rmse, max_amean, min_n)``; the
    first three are finite non-bool non-negative ``int``/``float`` values and
    ``min_n`` is a positive non-bool integer.

    A window passes only when its ``R`` is not ``null`` and
    ``abs(emin) <= max_abs``, ``abs(emax) <= max_abs``,
    ``rmse <= max_rmse``, ``amean <= max_amean`` and ``n >= min_n``. Boundary
    comparisons are exact: the six-decimal report metrics and the limits are
    compared as ``Decimal(str(value))``.

    Returns a canonical compact JSON string with exactly the two keys
    ``windows`` and ``passed`` in that order: each window keeps its five
    integer fields and ``R`` in order, followed by the lowercase boolean
    ``true``/``false`` for that window, and the overall ``passed`` is true only
    when every window passes; an empty window set yields
    ``{"windows":[],"passed":true}``. Integers are decimal, the four metrics
    use exactly six decimal places, ``NaN``/``Infinity`` never appear and the
    output has no whitespace or trailing newline. The inputs are never
    modified and repeated calls return byte-identical strings.

    :raises TypeError: ``report`` is not a ``str``, ``limits`` is not a tuple,
        or ``limits`` has a bad length or field type.
    :raises ValueError: a limit value is out of range, or the report's JSON,
        structure, values or canonical formatting are bad.
    """
    if not isinstance(report, str):
        raise TypeError("report must be a str")
    if not isinstance(limits, tuple):
        raise TypeError("limits must be a tuple")
    if len(limits) != 4:
        raise TypeError(
            "limits must be a 4-tuple (max_abs, max_rmse, max_amean, min_n)"
        )

    for index, name in enumerate(("max_abs", "max_rmse", "max_amean")):
        value = limits[index]
        if isinstance(value, bool) or not isinstance(value, _NUMERIC_TYPES):
            raise TypeError(f"{name} must be a non-bool int or float")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    min_n = limits[3]
    if isinstance(min_n, bool) or not isinstance(min_n, int):
        raise TypeError("min_n must be a non-bool integer")
    if min_n < 1:
        raise ValueError("min_n must be a positive integer")

    windows = _decode_gate_report(report)
    max_abs = Decimal(str(limits[0]))
    max_rmse = Decimal(str(limits[1]))
    max_amean = Decimal(str(limits[2]))

    passed_flags = []
    overall_passed = True
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        for window in windows:
            summary = window[5]
            if summary is None:
                window_passed = False
            else:
                emin, emax, rmse, amean, match_count = summary
                window_passed = (
                    abs(emin) <= max_abs
                    and abs(emax) <= max_abs
                    and rmse <= max_rmse
                    and amean <= max_amean
                    and match_count >= min_n
                )
            passed_flags.append(window_passed)
            overall_passed = overall_passed and window_passed

    return _format_gate_report(windows, passed_flags, overall_passed)


def _decode_delivery_gate(text: str) -> tuple:
    """Parse and validate a canonical :func:`gate_report` document.

    Returns ``(windows, window_flags, passed)`` where ``windows`` is a list of
    ``[level, ix_min, iy_min, ix_max, iy_max, summary]`` (``summary`` is
    ``None`` or ``[emin, emax, rmse, amean, n]``), ``window_flags`` is a list
    of per-window booleans in window order and ``passed`` is the top-level
    flag.

    :raises ValueError: the JSON syntax or shape is bad, a value violates the
        gate report contract, the text is not the canonical gate encoding, or
        the top-level ``passed`` does not equal the logical AND of the window
        flags (an empty window set being ``true``).
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("gate is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"windows",
                                                           "passed"}:
        raise ValueError("gate top-level value must be an object with only "
                         "'windows' and 'passed'")
    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("gate 'windows' must be an array")
    passed = document["passed"]
    if not isinstance(passed, bool):
        raise ValueError("gate 'passed' must be a boolean")

    windows = []
    window_flags = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 7:
            raise ValueError("each gate window must be an array of seven "
                             "values")
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), raw_window[:5]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"gate {name} must be a non-bool integer")
        level, ix_min, iy_min, ix_max, iy_max = raw_window[:5]
        if level < 0:
            raise ValueError("gate level must be non-negative")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("gate window bounds must satisfy ix_min <= "
                             "ix_max and iy_min <= iy_max")

        raw_summary = raw_window[5]
        summary = None
        if raw_summary is not None:
            if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                raise ValueError("each gate summary must be null or an "
                                 "array of five values")
            metrics_raw = list(raw_summary[:4])
            match_count = raw_summary[4]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics_raw):
                if isinstance(value, bool) or not isinstance(value,
                                                             (int, Decimal)):
                    raise ValueError(f"gate {name} must be a number")
            metrics = [Decimal(value) for value in metrics_raw]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics):
                if not value.is_finite():
                    raise ValueError(f"gate {name} must be finite")
            if isinstance(match_count, bool) or not isinstance(match_count,
                                                               int):
                raise ValueError("gate n must be a non-bool integer")
            if match_count < 1:
                raise ValueError("gate n must be a positive integer")
            if metrics[2] < 0 or metrics[3] < 0:
                raise ValueError("gate rmse and amean must be "
                                 "non-negative")
            summary = [*metrics, match_count]

        window_passed = raw_window[6]
        if not isinstance(window_passed, bool):
            raise ValueError("each gate window flag must be a boolean")
        if summary is None and window_passed:
            raise ValueError("a gate window with a null summary cannot pass")

        windows.append([level, ix_min, iy_min, ix_max, iy_max, summary])
        window_flags.append(window_passed)

    expected_passed = all(window_flags)
    if passed != expected_passed:
        raise ValueError("gate top-level 'passed' must equal the logical "
                         "AND of the window flags")

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    canonical = _format_gate_report(windows, window_flags, passed)
    if canonical != text:
        raise ValueError("gate is not the canonical gate report encoding")
    return windows, window_flags, passed


_DELIVERY_IDENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _json_string(value: str) -> str:
    """Serialize a str as a compact JSON string literal."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_delivery_entries(normalized: list) -> str:
    """Serialize sorted delivery entries to the two-key manifest document."""
    parts = ['{"entries":[']
    for index, (batch, product, version, passed,
                failures) in enumerate(normalized):
        if index:
            parts.append(",")
        parts.append("[" + str(batch) + "," + _json_string(product) + ","
                     + _json_string(version) + ",")
        parts.append("true," if passed else "false,")
        parts.append(_format_report_windows_array(failures))
        parts.append("]")
    parts.append('],"releasable":')
    parts.append("true}" if all(entry[3] for entry in normalized) else "false}")
    return "".join(parts)


def build_delivery_manifest(items: tuple) -> str:
    """Build a delivery manifest from gated product reports.

    ``items`` is a tuple of 4-tuples ``(batch, product, version, gate)``.
    ``batch`` is a non-bool non-negative integer; ``product`` and ``version``
    are non-empty strings containing only ASCII letters, digits and the
    characters ``.``, ``_`` and ``-``; ``gate`` must byte-for-byte match the
    canonical output of :func:`gate_report`, whose top-level ``passed`` must
    equal the logical AND of its per-window flags (an empty window set being
    ``true``).

    Returns a canonical compact JSON document with exactly the two keys
    ``entries`` and ``releasable`` in that order. Entries are sorted
    lexicographically by ``(batch, product)`` and each entry is the array
    ``[batch, product, version, passed, failures]`` where ``passed`` is the
    gate's top-level flag and ``failures`` lists the first six failing windows
    (those whose per-window flag is ``false``) in gate window order, as their
    six-value ``[level, ix_min, iy_min, ix_max, iy_max, R]`` arrays (``R`` is
    ``null`` or ``[emin, emax, rmse, amean, n]``); ``failures`` is ``[]`` when
    no window fails. ``releasable`` is the logical AND of every entry's
    ``passed`` and is ``true`` for empty ``items``. Integers are decimal, the
    four metrics use exactly six decimal places (negative zero written as
    ``0.000000``), ``NaN``/``Infinity`` never appear and the output has no
    whitespace or trailing newline. The input is never modified and reordering
    ``items`` yields a byte-identical document.

    :raises TypeError: ``items`` is not a tuple, an item is not a tuple of
        four values, or a field has the wrong type.
    :raises ValueError: ``batch`` is negative, an identifier contains an
        illegal character, ``(batch, product)`` is duplicated, or a gate is
        not a canonical consistent gate report.
    """
    if not isinstance(items, tuple):
        raise TypeError("items must be a tuple")

    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 4:
            raise TypeError(
                "each item must be a 4-tuple (batch, product, version, gate)"
            )
        batch, product, version, gate = item
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise TypeError("batch must be a non-bool integer")
        if batch < 0:
            raise ValueError("batch must be non-negative")
        if not isinstance(product, str):
            raise TypeError("product must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(version, str):
            raise TypeError("version must be a str")
        if not version or not set(version) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "version must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(gate, str):
            raise TypeError("gate must be a str")

        key = (batch, product)
        if key in seen:
            raise ValueError(f"duplicate (batch, product): {key!r}")
        seen.add(key)

        windows, window_flags, passed = _decode_delivery_gate(gate)
        failures = []
        for window, window_passed in zip(windows, window_flags):
            if not window_passed:
                failures.append(window)
                if len(failures) == 6:
                    break
        normalized.append((batch, product, version, passed, failures))

    normalized.sort(key=lambda entry: (entry[0], entry[1]))

    return _format_delivery_entries(normalized)


def _decode_delivery_manifest(text: str) -> list:
    """Parse and validate a canonical :func:`build_delivery_manifest` string.

    Returns the entries as a list of
    ``[batch, product, version, passed, failures]`` where ``failures`` is a
    list of ``[level, ix_min, iy_min, ix_max, iy_max, R]`` (``R`` is ``None``
    or ``[emin, emax, rmse, amean, n]``), in manifest order.

    :raises ValueError: the JSON syntax or shape is bad, a value violates the
        manifest contract, the text is not the canonical manifest encoding, or
        the top-level ``releasable`` does not equal the logical AND of the
        entry flags (an empty entry set being ``true``).
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("manifest is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"entries",
                                                           "releasable"}:
        raise ValueError("manifest top-level value must be an object with "
                         "only 'entries' and 'releasable'")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("manifest 'entries' must be an array")
    releasable = document["releasable"]
    if not isinstance(releasable, bool):
        raise ValueError("manifest 'releasable' must be a boolean")

    entries = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, list) or len(raw_entry) != 5:
            raise ValueError("each manifest entry must be an array of five "
                             "values")
        batch, product, version, passed, raw_failures = raw_entry
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("manifest batch must be a non-bool integer")
        if batch < 0:
            raise ValueError("manifest batch must be non-negative")
        if not isinstance(product, str):
            raise ValueError("manifest product must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "manifest product must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(version, str):
            raise ValueError("manifest version must be a str")
        if not version or not set(version) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "manifest version must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(passed, bool):
            raise ValueError("manifest entry passed flag must be a boolean")
        if not isinstance(raw_failures, list) or len(raw_failures) > 6:
            raise ValueError("manifest failures must be an array of at most "
                             "six windows")

        failures = []
        for raw_window in raw_failures:
            if not isinstance(raw_window, list) or len(raw_window) != 6:
                raise ValueError("each manifest failure must be an array of "
                                 "six values")
            for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                    "iy_max"), raw_window[:5]):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"manifest failure {name} must be a "
                                     "non-bool integer")
            level, ix_min, iy_min, ix_max, iy_max = raw_window[:5]
            if level < 0:
                raise ValueError("manifest failure level must be "
                                 "non-negative")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("manifest failure bounds must satisfy "
                                 "ix_min <= ix_max and iy_min <= iy_max")

            raw_summary = raw_window[5]
            summary = None
            if raw_summary is not None:
                if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                    raise ValueError("each manifest failure R must be null or "
                                     "an array of five values")
                metrics_raw = list(raw_summary[:4])
                match_count = raw_summary[4]
                for name, value in zip(("emin", "emax", "rmse", "amean"),
                                       metrics_raw):
                    if isinstance(value, bool) or not isinstance(value,
                                                                 (int,
                                                                  Decimal)):
                        raise ValueError(f"manifest failure {name} must be a "
                                         "number")
                metrics = [Decimal(value) for value in metrics_raw]
                for name, value in zip(("emin", "emax", "rmse", "amean"),
                                       metrics):
                    if not value.is_finite():
                        raise ValueError(f"manifest failure {name} must be "
                                         "finite")
                if isinstance(match_count, bool) or not isinstance(
                        match_count, int):
                    raise ValueError("manifest failure n must be a non-bool "
                                     "integer")
                if match_count < 1:
                    raise ValueError("manifest failure n must be a positive "
                                     "integer")
                if metrics[2] < 0 or metrics[3] < 0:
                    raise ValueError("manifest failure rmse and amean must be "
                                     "non-negative")
                summary = [*metrics, match_count]

            failures.append([level, ix_min, iy_min, ix_max, iy_max, summary])

        if passed and failures:
            raise ValueError("a passing manifest entry must have no failures")
        if not passed and not failures:
            raise ValueError("a failing manifest entry must list a failure")

        entries.append([batch, product, version, passed, failures])

    # Entries must appear strictly ascending by (batch, product); a
    # byte-for-byte re-encode alone would tolerate reordering.
    keys = [(entry[0], entry[1]) for entry in entries]
    if any(keys[index] >= keys[index + 1]
           for index in range(len(keys) - 1)):
        raise ValueError("manifest entries must be sorted uniquely by "
                         "(batch, product)")

    # Byte-for-byte canonical equality also checks the top-level releasable
    # flag (the AND of the entry flags) and rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    if _format_delivery_entries(entries) != text:
        raise ValueError("manifest is not the canonical delivery manifest "
                         "encoding")
    return entries


def _format_delivery_changes(changes: list, releasable: bool) -> str:
    """Serialize sorted delivery changes to the two-key merge document."""
    parts = ['{"changes":[']
    for index, (batch, product, previous, version, passed,
                failures) in enumerate(changes):
        if index:
            parts.append(",")
        parts.append("[" + str(batch) + "," + _json_string(product) + ",")
        parts.append("null" if previous is None else _json_string(previous))
        parts.append("," + _json_string(version) + ",")
        parts.append("true," if passed else "false,")
        parts.append(_format_report_windows_array(failures))
        parts.append("]")
    parts.append('],"releasable":')
    parts.append("true}" if releasable else "false}")
    return "".join(parts)


def merge_delivery_manifests(manifests: tuple) -> str:
    """Merge canonical delivery manifests into a delivery-changes document.

    ``manifests`` is a tuple of strings, each byte-for-byte matching the
    canonical output of :func:`build_delivery_manifest`. The documents are
    decoded and their structure, sorting, numeric formatting and top-level
    ``releasable`` flags are re-encoded and checked byte for byte.

    Entries sharing a ``(batch, product)`` must be identical and are then
    deduplicated; conflicting entries raise :class:`ValueError`. Entries of
    the same product are ordered by ``batch`` and a product's ``version``
    must never repeat across its batches; each change's ``previous`` is the
    version of that product's preceding batch (``null`` for its first
    batch).

    Returns a canonical compact JSON document with exactly the two keys
    ``changes`` and ``releasable`` in that order. Changes are sorted
    lexicographically by ``(batch, product)`` and each change is the array
    ``[batch, product, previous, version, passed, failures]`` with
    ``failures`` preserved in their original window order as six-value
    ``[level, ix_min, iy_min, ix_max, iy_max, R]`` arrays (``R`` is ``null``
    or ``[emin, emax, rmse, amean, n]``). ``releasable`` is the logical AND
    of the ``passed`` flag of each product's final batch (its greatest
    batch) and is ``true`` for an empty set. Integers are decimal, the four
    metrics use exactly six decimal places (negative zero written as
    ``0.000000``), ``NaN``/``Infinity`` never appear and the output has no
    whitespace or trailing newline. The inputs are never modified: reordering
    ``manifests`` yields a byte-identical document.

    :raises TypeError: ``manifests`` is not a tuple or one of its members is
        not a ``str``.
    :raises ValueError: a manifest's JSON syntax, structure, fields, sort
        order, numeric formatting or ``releasable`` flag are bad, two
        ``(batch, product)`` entries conflict, or a product reuses a version
        across batches.
    """
    if not isinstance(manifests, tuple):
        raise TypeError("manifests must be a tuple")

    merged = {}
    for manifest in manifests:
        if not isinstance(manifest, str):
            raise TypeError("each manifest must be a str")
        for batch, product, version, passed, failures in (
                _decode_delivery_manifest(manifest)):
            key = (batch, product)
            value = (version, passed, failures)
            existing = merged.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"conflicting (batch, product) entries: "
                                 f"{key!r}")
            merged.setdefault(key, value)

    changes = [[batch, product, None, version, passed, failures]
               for (batch, product), (version, passed, failures)
               in merged.items()]
    changes.sort(key=lambda change: (change[0], change[1]))

    versions_by_product = {}
    previous_by_product = {}
    final_passed = {}
    for change in changes:
        batch, product, version, passed = (change[0], change[1], change[3],
                                           change[4])
        if version in versions_by_product.setdefault(product, set()):
            raise ValueError(f"version {version!r} is reused by product "
                             f"{product!r}")
        versions_by_product[product].add(version)
        change[2] = previous_by_product.get(product)
        previous_by_product[product] = version
        # Changes are (batch, product)-sorted, so the last write per product
        # is its greatest batch and hence its final passed flag.
        final_passed[product] = passed

    releasable = all(final_passed.values())
    return _format_delivery_changes(changes, releasable)


def _decode_delivery_changes(text: str) -> tuple:
    """Parse and validate a canonical :func:`merge_delivery_manifests` string.

    Returns ``(changes, releasable)`` where ``changes`` is a list of
    ``[batch, product, previous, version, passed, failures]`` (``previous`` is
    ``None`` or a version string and ``failures`` is a list of
    ``[level, ix_min, iy_min, ix_max, iy_max, R]`` with ``R`` ``None`` or
    ``[emin, emax, rmse, amean, n]``), in document order.

    Beyond JSON, shape and canonical-format checks, the structural merge
    contract is revalidated: unique ``(batch, product)`` keys sorted
    ascending; at most six failures per change with a passing change carrying
    none and a failing change carrying at least one; a change's ``previous``
    is the version of its product's preceding change in batch order (``null``
    for its first batch); a product's versions never repeat; and the
    top-level ``releasable`` is the logical AND of each product's final
    change flag (an empty change set being ``true``).

    :raises ValueError: the JSON syntax or shape is bad, a value violates the
        changes contract, the text is not the canonical changes encoding, or
        the top-level ``releasable`` flag is inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("changes are not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"changes",
                                                           "releasable"}:
        raise ValueError("changes top-level value must be an object with "
                         "only 'changes' and 'releasable'")
    raw_changes = document["changes"]
    if not isinstance(raw_changes, list):
        raise ValueError("'changes' must be an array")
    releasable = document["releasable"]
    if not isinstance(releasable, bool):
        raise ValueError("'releasable' must be a boolean")

    changes = []
    for raw_change in raw_changes:
        if not isinstance(raw_change, list) or len(raw_change) != 6:
            raise ValueError("each change must be an array of six values")
        batch, product, previous, version, passed, raw_failures = raw_change
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("change batch must be a non-bool integer")
        if batch < 0:
            raise ValueError("change batch must be non-negative")
        if not isinstance(product, str):
            raise ValueError("change product must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "change product must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if previous is not None and not isinstance(previous, str):
            raise ValueError("change previous must be null or a str")
        if previous is not None and (
                not previous or not set(previous) <= _DELIVERY_IDENT_CHARS):
            raise ValueError(
                "change previous must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(version, str):
            raise ValueError("change version must be a str")
        if not version or not set(version) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "change version must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(passed, bool):
            raise ValueError("change passed flag must be a boolean")
        if not isinstance(raw_failures, list) or len(raw_failures) > 6:
            raise ValueError("change failures must be an array of at most "
                             "six windows")

        failures = []
        for raw_window in raw_failures:
            if not isinstance(raw_window, list) or len(raw_window) != 6:
                raise ValueError("each change failure must be an array of "
                                 "six values")
            for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                    "iy_max"), raw_window[:5]):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"change failure {name} must be a "
                                     "non-bool integer")
            level, ix_min, iy_min, ix_max, iy_max = raw_window[:5]
            if level < 0:
                raise ValueError("change failure level must be non-negative")
            if ix_min > ix_max or iy_min > iy_max:
                raise ValueError("change failure bounds must satisfy "
                                 "ix_min <= ix_max and iy_min <= iy_max")

            raw_summary = raw_window[5]
            summary = None
            if raw_summary is not None:
                if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                    raise ValueError("each change failure R must be null or "
                                     "an array of five values")
                metrics_raw = list(raw_summary[:4])
                match_count = raw_summary[4]
                for name, value in zip(("emin", "emax", "rmse", "amean"),
                                       metrics_raw):
                    if isinstance(value, bool) or not isinstance(value,
                                                                 (int,
                                                                  Decimal)):
                        raise ValueError(f"change failure {name} must be a "
                                         "number")
                metrics = [Decimal(value) for value in metrics_raw]
                for name, value in zip(("emin", "emax", "rmse", "amean"),
                                       metrics):
                    if not value.is_finite():
                        raise ValueError(f"change failure {name} must be "
                                         "finite")
                if isinstance(match_count, bool) or not isinstance(
                        match_count, int):
                    raise ValueError("change failure n must be a non-bool "
                                     "integer")
                if match_count < 1:
                    raise ValueError("change failure n must be a positive "
                                     "integer")
                if metrics[2] < 0 or metrics[3] < 0:
                    raise ValueError("change failure rmse and amean must be "
                                     "non-negative")
                summary = [*metrics, match_count]

            failures.append([level, ix_min, iy_min, ix_max, iy_max, summary])

        if passed and failures:
            raise ValueError("a passing change must have no failures")
        if not passed and not failures:
            raise ValueError("a failing change must list a failure")

        changes.append([batch, product, previous, version, passed, failures])

    # Changes must appear strictly ascending by (batch, product); a
    # byte-for-byte re-encode alone would tolerate reordering.
    keys = [(change[0], change[1]) for change in changes]
    if any(keys[index] >= keys[index + 1]
           for index in range(len(keys) - 1)):
        raise ValueError("changes must be sorted uniquely by "
                         "(batch, product)")

    # Revalidate the per-product history: previous points at the preceding
    # batch's version and versions never repeat.
    previous_by_product = {}
    versions_by_product = {}
    final_passed = {}
    for change in changes:
        batch, product, previous, version, passed = change[:5]
        expected_previous = previous_by_product.get(product)
        if previous != expected_previous:
            raise ValueError("change previous must be the version of the "
                             "product's preceding batch")
        if version in versions_by_product.setdefault(product, set()):
            raise ValueError(f"version {version!r} is reused by product "
                             f"{product!r}")
        versions_by_product[product].add(version)
        previous_by_product[product] = version
        # Changes are (batch, product)-sorted, so the last write per product
        # is its greatest batch and hence its final passed flag.
        final_passed[product] = passed

    if releasable != all(final_passed.values()):
        raise ValueError("'releasable' must equal the logical AND of the "
                         "final change flag of each product")

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    if _format_delivery_changes(changes, releasable) != text:
        raise ValueError("changes are not the canonical delivery changes "
                         "encoding")
    return changes, releasable


def _format_failure_rows(failures: list) -> str:
    """Serialize expanded failure rows to the compact array body."""
    parts = []
    for failure_index, (batch, version, level, ix_min, iy_min, ix_max,
                        iy_max, summary) in enumerate(failures):
        if failure_index:
            parts.append(",")
        parts.append("[" + str(batch) + "," + _json_string(version) + ",")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",")
        if summary is None:
            parts.append("null")
        else:
            emin, emax, rmse, amean, match_count = summary
            parts.append("[")
            parts.append(",".join((_format_decimal6(emin),
                                   _format_decimal6(emax),
                                   _format_decimal6(rmse),
                                   _format_decimal6(amean),
                                   str(match_count))))
            parts.append("]")
        parts.append("]")
    return "".join(parts)


def _format_audit_products(products: list) -> str:
    """Serialize per-product audit rows to the compact products array body."""
    parts = ["["]
    for index, (product, current, passed, rollback, affected,
                failures) in enumerate(products):
        if index:
            parts.append(",")
        parts.append("[" + _json_string(product) + ","
                     + _json_string(current) + ",")
        parts.append("true," if passed else "false,")
        parts.append("null" if rollback is None else _json_string(rollback))
        parts.append(",[" + ",".join(str(batch) for batch in affected) + "],[")
        parts.append(_format_failure_rows(failures))
        parts.append("]]")
    parts.append("]")
    return "".join(parts)


def _format_audit_delivery(products: list, releasable: bool) -> str:
    """Serialize audited products to the two-key audit document."""
    return ('{"products":' + _format_audit_products(products)
            + ',"releasable":' + ("true}" if releasable else "false}"))


def audit_delivery_changes(changes: str) -> str:
    """Audit a delivery-changes document for per-product rollback state.

    ``changes`` must be a ``str`` byte-for-byte matching the canonical output
    of :func:`merge_delivery_manifests`: the compact document whose sole
    top-level keys are ``changes`` and ``releasable`` in that order, with
    changes ``[batch, product, previous, version, passed, failures]`` sorted
    uniquely by ``(batch, product)``. Each ``previous`` must be the version
    of the product's preceding batch (``null`` for its first batch), a
    product's versions must never repeat, a passing change must carry no
    failures and a failing change at least one (at most six), and
    ``releasable`` must equal the logical AND of each product's final change
    flag. Failure windows are six-value
    ``[level, ix_min, iy_min, ix_max, iy_max, R]`` arrays with ``R`` null or
    ``[emin, emax, rmse, amean, n]`` whose four metrics are finite numbers
    written with exactly six decimal places.

    Returns a canonical compact JSON document with exactly the two keys
    ``products`` and ``releasable`` in that order. Products are sorted by
    product name and each product row is
    ``[product, current, passed, rollback, affected, failures]``. ``current``
    and ``passed`` come from the product's last (greatest-batch) change. When
    that change passes, ``rollback`` is ``null`` and ``affected`` and
    ``failures`` are empty. Otherwise ``rollback`` is the version of the
    nearest earlier passing change of the product (``null`` when none
    exists), ``affected`` lists the failing batches strictly after that
    rollback change in ascending order (or every failing batch when the
    product never passed), and ``failures`` expands the failing windows of
    those changes, ordered by batch and by the windows' source order, as
    eight-value
    ``[batch, version, level, ix_min, iy_min, ix_max, iy_max, R]`` arrays
    with ``R`` preserved verbatim. The top-level ``releasable`` echoes the
    input document's flag. Integers are decimal, the four metrics use exactly
    six decimal places (negative zero written as ``0.000000``),
    ``NaN``/``Infinity`` never appear and the output has no whitespace or
    trailing newline.

    :raises TypeError: ``changes`` is not a ``str``.
    :raises ValueError: the document's JSON syntax, structure, values,
        history, sort order, numeric formatting or ``releasable`` flag are
        bad, or it is not the canonical changes encoding.
    """
    if not isinstance(changes, str):
        raise TypeError("changes must be a str")

    decoded, releasable = _decode_delivery_changes(changes)

    by_product = {}
    for change in decoded:
        batch, product, previous, version, passed, failures = change
        by_product.setdefault(product, []).append(
            (batch, version, passed, failures))

    products = []
    for product in sorted(by_product):
        history = by_product[product]
        # Changes are (batch, product)-sorted, so history is already in
        # ascending batch order for each product.
        current_batch, current, passed, current_failures = history[-1]

        if passed:
            products.append([product, current, True, None, [], []])
            continue

        rollback_index = None
        for index in range(len(history) - 2, -1, -1):
            if history[index][2]:
                rollback_index = index
                break
        rollback = None if rollback_index is None else history[
            rollback_index][1]

        affected = []
        failures_out = []
        start = 0 if rollback_index is None else rollback_index + 1
        for batch, version, change_passed, change_failures in history[start:]:
            if not change_passed:
                affected.append(batch)
                for level, ix_min, iy_min, ix_max, iy_max, summary in (
                        change_failures):
                    failures_out.append([batch, version, level, ix_min,
                                         iy_min, ix_max, iy_max, summary])

        products.append([product, current, False, rollback, affected,
                         failures_out])

    return _format_audit_delivery(products, releasable)


def _decode_failure_rows(raw_failures) -> list:
    """Validate and normalize product failure rows.

    Shared by the audit and plan decoders. Each row must be an eight-value
    ``[batch, version, level, ix_min, iy_min, ix_max, iy_max, R]`` array with
    ``R`` ``None`` or ``[emin, emax, rmse, amean, n]``; the four metrics are
    returned as finite Decimals (``rmse``/``amean`` non-negative) and ``n`` as
    a positive non-bool int.
    """
    if not isinstance(raw_failures, list):
        raise ValueError("product failures must be an array")

    failures = []
    for raw_failure in raw_failures:
        if not isinstance(raw_failure, list) or len(raw_failure) != 8:
            raise ValueError("each product failure must be an array of "
                             "eight values")
        batch, version = raw_failure[:2]
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("product failure batch must be a non-bool "
                             "integer")
        if batch < 0:
            raise ValueError("product failure batch must be "
                             "non-negative")
        if not isinstance(version, str):
            raise ValueError("product failure version must be a str")
        if not version or not set(version) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product failure version must be non-empty and contain "
                "only ASCII alphanumeric characters and ._-"
            )
        for name, value in zip(("level", "ix_min", "iy_min", "ix_max",
                                "iy_max"), raw_failure[2:7]):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"product failure {name} must be a "
                                 "non-bool integer")
        level, ix_min, iy_min, ix_max, iy_max = raw_failure[2:7]
        if level < 0:
            raise ValueError("product failure level must be "
                             "non-negative")
        if ix_min > ix_max or iy_min > iy_max:
            raise ValueError("product failure bounds must satisfy "
                             "ix_min <= ix_max and iy_min <= iy_max")

        raw_summary = raw_failure[7]
        summary = None
        if raw_summary is not None:
            if not isinstance(raw_summary, list) or len(raw_summary) != 5:
                raise ValueError("product failure R must be null or an "
                                 "array of five values")
            metrics_raw = list(raw_summary[:4])
            match_count = raw_summary[4]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics_raw):
                if isinstance(value, bool) or not isinstance(value,
                                                             (int,
                                                              Decimal)):
                    raise ValueError(f"product failure {name} must be a "
                                     "number")
            metrics = [Decimal(value) for value in metrics_raw]
            for name, value in zip(("emin", "emax", "rmse", "amean"),
                                   metrics):
                if not value.is_finite():
                    raise ValueError(f"product failure {name} must be "
                                     "finite")
            if isinstance(match_count, bool) or not isinstance(
                    match_count, int):
                raise ValueError("product failure n must be a non-bool "
                                 "integer")
            if match_count < 1:
                raise ValueError("product failure n must be a positive "
                                 "integer")
            if metrics[2] < 0 or metrics[3] < 0:
                raise ValueError("product failure rmse and amean must "
                                 "be non-negative")
            summary = [*metrics, match_count]

        failures.append([batch, version, level, ix_min, iy_min, ix_max,
                         iy_max, summary])
    return failures


def _decode_audit_delivery(text: str) -> tuple:
    """Parse and validate a canonical :func:`audit_delivery_changes` string.

    Returns ``(products, releasable)`` where ``products`` is a list of
    ``[product, current, passed, rollback, affected, failures]`` rows in
    document order (``rollback`` is ``None`` or a version string,
    ``affected`` a list of batch ints and ``failures`` a list of
    ``[batch, version, level, ix_min, iy_min, ix_max, iy_max, R]`` with ``R``
    ``None`` or ``[emin, emax, rmse, amean, n]``).

    Beyond JSON, shape and canonical-format checks, the audit contract is
    revalidated: product rows are sorted uniquely by product name; a passing
    row carries a ``null`` rollback and empty ``affected``/``failures``
    while a failing row lists at least one affected batch and one failure;
    ``affected`` is strictly ascending and matches exactly the batches of
    the row's failures, which are ordered by batch; and the top-level
    ``releasable`` is the logical AND of every row's ``passed`` flag (an
    empty product set being ``true``).

    :raises ValueError: the JSON syntax or shape is bad, a value violates
        the audit contract, the text is not the canonical audit encoding,
        or the top-level ``releasable`` flag is inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("audit is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"products",
                                                           "releasable"}:
        raise ValueError("audit top-level value must be an object with "
                         "only 'products' and 'releasable'")
    raw_products = document["products"]
    if not isinstance(raw_products, list):
        raise ValueError("'products' must be an array")
    releasable = document["releasable"]
    if not isinstance(releasable, bool):
        raise ValueError("'releasable' must be a boolean")

    products = []
    for raw_product in raw_products:
        if not isinstance(raw_product, list) or len(raw_product) != 6:
            raise ValueError("each product must be an array of six values")
        product, current, passed, rollback, raw_affected, raw_failures = (
            raw_product)
        if not isinstance(product, str):
            raise ValueError("product name must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product name must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(current, str):
            raise ValueError("product current must be a str")
        if not current or not set(current) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product current must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(passed, bool):
            raise ValueError("product passed flag must be a boolean")
        if rollback is not None and not isinstance(rollback, str):
            raise ValueError("product rollback must be null or a str")
        if rollback is not None and (
                not rollback or not set(rollback) <= _DELIVERY_IDENT_CHARS):
            raise ValueError(
                "product rollback must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(raw_affected, list):
            raise ValueError("product affected must be an array")
        affected = []
        for batch in raw_affected:
            if isinstance(batch, bool) or not isinstance(batch, int):
                raise ValueError("product affected batches must be non-bool "
                                 "integers")
            if batch < 0:
                raise ValueError("product affected batches must be "
                                 "non-negative")
            affected.append(batch)
        if any(affected[index] >= affected[index + 1]
               for index in range(len(affected) - 1)):
            raise ValueError("product affected batches must be strictly "
                             "ascending")
        failures = _decode_failure_rows(raw_failures)

        # Row field association: a passing row carries no rollback state,
        # a failing row carries at least one affected batch and one
        # failure, and the failures expand exactly the affected batches in
        # batch order.
        if passed:
            if rollback is not None or affected or failures:
                raise ValueError("a passing product must have a null "
                                 "rollback and no affected batches or "
                                 "failures")
        else:
            if not affected or not failures:
                raise ValueError("a failing product must list an affected "
                                 "batch and a failure")
            failure_batches = [failure[0] for failure in failures]
            if any(failure_batches[index] > failure_batches[index + 1]
                   for index in range(len(failure_batches) - 1)):
                raise ValueError("product failures must be ordered by "
                                 "batch")
            if sorted(set(failure_batches)) != affected:
                raise ValueError("product affected batches must match the "
                                 "batches of its failures")

        products.append([product, current, passed, rollback, affected,
                         failures])

    # Products must appear strictly ascending by name; a byte-for-byte
    # re-encode alone would tolerate reordering.
    names = [product[0] for product in products]
    if any(names[index] >= names[index + 1]
           for index in range(len(names) - 1)):
        raise ValueError("products must be sorted uniquely by product name")

    if releasable != all(product[2] for product in products):
        raise ValueError("'releasable' must equal the logical AND of the "
                         "passed flag of each product")

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting, negative zero as
    # anything but ``0.000000``, leading zeros, exponents and any other
    # non-canonical spelling.
    if _format_audit_delivery(products, releasable) != text:
        raise ValueError("audit is not the canonical audit delivery "
                         "encoding")
    return products, releasable


def _format_delivery_plan(operations: list, releasable: bool) -> str:
    """Serialize planned operations to the two-key plan document."""
    parts = ['{"operations":[']
    for index, (product, action, current, target, affected,
                failures) in enumerate(operations):
        if index:
            parts.append(",")
        parts.append("[" + _json_string(product) + ","
                     + _json_string(action) + ","
                     + _json_string(current) + ",")
        parts.append("null" if target is None else _json_string(target))
        parts.append(",[" + ",".join(str(batch) for batch in affected) + "],[")
        parts.append(_format_failure_rows(failures))
        parts.append("]]")
    parts.append('],"releasable":')
    parts.append("true}" if releasable else "false}")
    return "".join(parts)


def build_delivery_plan(audit: str) -> str:
    """Build a per-product delivery plan from a delivery audit document.

    ``audit`` must be a ``str`` byte-for-byte matching the canonical output
    of :func:`audit_delivery_changes`: the compact document whose sole
    top-level keys are ``products`` and ``releasable`` in that order, with
    product rows ``[product, current, passed, rollback, affected,
    failures]`` sorted uniquely by product name. A passing row must carry a
    ``null`` rollback and empty ``affected``/``failures``; a failing row
    must list at least one affected batch and one failure, with
    ``affected`` strictly ascending and matching exactly the batches of the
    row's failures, which are ordered by batch. Failure rows are
    eight-value ``[batch, version, level, ix_min, iy_min, ix_max, iy_max,
    R]`` arrays with ``R`` null or ``[emin, emax, rmse, amean, n]`` whose
    four metrics are finite numbers written with exactly six decimal
    places, and ``releasable`` must equal the logical AND of every row's
    ``passed`` flag.

    Returns a canonical compact JSON document with exactly the two keys
    ``operations`` and ``releasable`` in that order. Operations follow the
    product order of the audit and each operation is the array
    ``[product, action, current, target, affected, failures]``. A passing
    product yields action ``publish`` with ``target`` equal to ``current``
    and empty ``affected``/``failures``. A failing product with a non-null
    ``rollback`` yields action ``rollback`` with ``target`` equal to
    ``rollback``; a failing product with a ``null`` ``rollback`` yields
    action ``block`` with a ``null`` ``target``. Failing products keep
    their ``affected`` and ``failures`` verbatim. The top-level
    ``releasable`` echoes the input document's flag and an empty product
    set yields ``{"operations":[],"releasable":true}``. Integers are
    decimal, the four metrics use exactly six decimal places (negative zero
    written as ``0.000000``), ``NaN``/``Infinity`` never appear and the
    output has no whitespace or trailing newline. The input is never
    modified and repeated calls return a byte-identical document.

    :raises TypeError: ``audit`` is not a ``str``.
    :raises ValueError: the document's JSON syntax, structure, values,
        field associations, product order, numeric formatting or
        ``releasable`` flag are bad, or it is not the canonical audit
        encoding.
    """
    if not isinstance(audit, str):
        raise TypeError("audit must be a str")

    products, releasable = _decode_audit_delivery(audit)

    operations = []
    for product, current, passed, rollback, affected, failures in products:
        if passed:
            operations.append([product, "publish", current, current, [],
                               []])
        elif rollback is not None:
            operations.append([product, "rollback", current, rollback,
                               affected, failures])
        else:
            operations.append([product, "block", current, None, affected,
                               failures])
    return _format_delivery_plan(operations, releasable)

def _decode_delivery_plan(text: str) -> list:
    """Parse and validate a canonical :func:`build_delivery_plan` string.

    Returns a list of ``[product, action, current, target, affected,
    failures]`` operations in document order (``target`` is ``None`` or a
    version string, ``affected`` a list of batch ints and ``failures`` a
    list of normalized failure rows; see :func:`_decode_failure_rows`).

    Beyond JSON, shape and canonical-format checks, the plan contract is
    revalidated: each operation's action is ``publish``, ``rollback`` or
    ``block``; a ``publish`` operation targets ``current`` with empty
    ``affected``/``failures``; a ``rollback`` operation carries a non-null
    ``target`` and a ``block`` operation a null one, both with at least one
    affected batch and one failure; ``affected`` is strictly ascending and
    matches exactly the batches of the operation's failures, which are
    ordered by batch; and the top-level ``releasable`` flag is true exactly
    when every action is ``publish`` (an empty operation set being
    ``true``).

    :raises ValueError: the JSON syntax or shape is bad, a value violates
        the plan contract, the text is not the canonical plan encoding, or
        the top-level ``releasable`` flag is inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("plan is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"operations",
                                                           "releasable"}:
        raise ValueError("plan top-level value must be an object with only "
                         "'operations' and 'releasable'")
    raw_operations = document["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError("'operations' must be an array")
    releasable = document["releasable"]
    if not isinstance(releasable, bool):
        raise ValueError("'releasable' must be a boolean")

    operations = []
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, list) or len(raw_operation) != 6:
            raise ValueError("each operation must be an array of six "
                             "values")
        product, action, current, target, raw_affected, raw_failures = (
            raw_operation)
        if not isinstance(product, str):
            raise ValueError("product name must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product name must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if action not in ("publish", "rollback", "block"):
            raise ValueError("operation action must be publish, rollback "
                             "or block")
        if not isinstance(current, str):
            raise ValueError("operation current must be a str")
        if not current or not set(current) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "operation current must be non-empty and contain only "
                "ASCII alphanumeric characters and ._-"
            )
        if target is not None and not isinstance(target, str):
            raise ValueError("operation target must be null or a str")
        if target is not None and (
                not target or not set(target) <= _DELIVERY_IDENT_CHARS):
            raise ValueError(
                "operation target must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(raw_affected, list):
            raise ValueError("operation affected must be an array")
        affected = []
        for batch in raw_affected:
            if isinstance(batch, bool) or not isinstance(batch, int):
                raise ValueError("operation affected batches must be "
                                 "non-bool integers")
            if batch < 0:
                raise ValueError("operation affected batches must be "
                                 "non-negative")
            affected.append(batch)
        if any(affected[index] >= affected[index + 1]
               for index in range(len(affected) - 1)):
            raise ValueError("operation affected batches must be strictly "
                             "ascending")

        failures = _decode_failure_rows(raw_failures)

        # A publish carries no target delta or failure state; a rollback or
        # block lists at least one affected batch and one failure, and the
        # failures expand exactly the affected batches in batch order.
        if action == "publish":
            if target != current or affected or failures:
                raise ValueError("a publish operation must target current "
                                 "with no affected batches or failures")
        else:
            if action == "rollback" and target is None:
                raise ValueError("a rollback operation must carry a "
                                 "non-null target")
            if action == "block" and target is not None:
                raise ValueError("a block operation must carry a null "
                                 "target")
            if not affected or not failures:
                raise ValueError("a rollback or block operation must list "
                                 "an affected batch and a failure")
            failure_batches = [failure[0] for failure in failures]
            if any(failure_batches[index] > failure_batches[index + 1]
                   for index in range(len(failure_batches) - 1)):
                raise ValueError("operation failures must be ordered by "
                                 "batch")
            if sorted(set(failure_batches)) != affected:
                raise ValueError("operation affected batches must match "
                                 "the batches of its failures")

        operations.append([product, action, current, target, affected,
                           failures])

    # Operations inherit the audit's product order: strictly ascending by
    # name with no duplicates; a byte-for-byte re-encode alone would
    # tolerate reordering.
    names = [operation[0] for operation in operations]
    if any(names[index] >= names[index + 1]
           for index in range(len(names) - 1)):
        raise ValueError("operations must be sorted uniquely by product "
                         "name")

    if releasable != all(operation[1] == "publish"
                         for operation in operations):
        raise ValueError("'releasable' must be true exactly when every "
                         "operation action is publish")

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting and any other
    # non-canonical spelling.
    if _format_delivery_plan(operations, releasable) != text:
        raise ValueError("plan is not the canonical delivery plan "
                         "encoding")
    return operations


def _format_delivery_receipt(receipts: list, ready: bool) -> str:
    """Serialize per-product receipts to the two-key receipt document."""
    parts = ['{"receipts":[']
    for index, (product, action, current, target, affected, status, reason,
                failures) in enumerate(receipts):
        if index:
            parts.append(",")
        parts.append("[" + _json_string(product) + ","
                     + _json_string(action) + ","
                     + _json_string(current) + ",")
        parts.append("null" if target is None else _json_string(target))
        parts.append(",[" + ",".join(str(batch) for batch in affected) + "],")
        parts.append(_json_string(status) + "," + _json_string(reason) + ",[")
        parts.append(_format_failure_rows(failures))
        parts.append("]]")
    parts.append('],"ready":')
    parts.append("true}" if ready else "false}")
    return "".join(parts)


def build_delivery_receipt(plan: str, results: tuple) -> str:
    """Build a per-product delivery receipt from a plan and execution results.

    ``plan`` must be a ``str`` byte-for-byte matching the canonical output
    of :func:`build_delivery_plan`: the compact document whose sole
    top-level keys are ``operations`` and ``releasable`` in that order, with
    operations ``[product, action, current, target, affected, failures]``
    in product order. ``results`` must be a ``tuple`` whose items are
    strictly three-value tuples ``(product, status, reason)`` of ``str``.
    The result products must match the plan products one-to-one with no
    duplicates, but may appear in any order. A ``publish`` or ``rollback``
    operation accepts only ``succeeded`` or ``failed``; a ``block``
    operation accepts only ``blocked``. A ``succeeded`` result carries an
    empty ``reason``; ``failed`` and ``blocked`` results carry a non-empty
    one.

    Returns a canonical compact JSON document with exactly the two keys
    ``receipts`` and ``ready`` in that order. Receipts follow the plan
    order and each receipt is the array ``[product, action, current,
    target, affected, status, reason, failures]`` with every planned field
    preserved verbatim. ``ready`` is true only when every operation is a
    ``publish`` and every result is ``succeeded``; an empty plan yields
    ``{"receipts":[],"ready":true}``. Strings use ``ensure_ascii=False``,
    integers are decimal, the four failure metrics use exactly six decimal
    places (negative zero written as ``0.000000``), ``NaN``/``Infinity``
    never appear and the output has no whitespace or trailing newline. The
    inputs are never modified and reordering ``results`` yields a
    byte-identical document.

    :raises TypeError: ``plan`` is not a ``str``, ``results`` is not a
        ``tuple``, or a result item is not a three-value tuple of ``str``.
    :raises ValueError: the plan is not the canonical plan encoding, a
        result product is missing, unknown or duplicated, or a status or
        reason violates the per-action contract.
    """
    if not isinstance(plan, str):
        raise TypeError("plan must be a str")
    if not isinstance(results, tuple):
        raise TypeError("results must be a tuple")

    operations = _decode_delivery_plan(plan)

    normalized = []
    for item in results:
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("each result must be a three-value tuple "
                            "(product, status, reason)")
        product, status, reason = item
        if not all(isinstance(value, str)
                   for value in (product, status, reason)):
            raise TypeError("product, status and reason must each be a str")
        normalized.append((product, status, reason))

    by_product = {}
    for product, status, reason in normalized:
        if product in by_product:
            raise ValueError(f"duplicate result for product {product!r}")
        by_product[product] = (status, reason)

    receipts = []
    for product, action, current, target, affected, failures in operations:
        if product not in by_product:
            raise ValueError(f"missing result for product {product!r}")
        status, reason = by_product.pop(product)
        if action in ("publish", "rollback"):
            if status not in ("succeeded", "failed"):
                raise ValueError(
                    f"{action} result for product {product!r} must be "
                    "succeeded or failed"
                )
        elif status != "blocked":
            raise ValueError(
                f"block result for product {product!r} must be blocked"
            )
        if status == "succeeded":
            if reason:
                raise ValueError(
                    f"succeeded result for product {product!r} must carry "
                    "an empty reason"
                )
        elif not reason:
            raise ValueError(
                f"{status} result for product {product!r} must carry a "
                "non-empty reason"
            )
        receipts.append([product, action, current, target, affected, status,
                         reason, failures])

    if by_product:
        product = next(iter(by_product))
        raise ValueError(f"unknown result for product {product!r}")

    ready = all(receipt[1] == "publish" and receipt[5] == "succeeded"
                for receipt in receipts)
    return _format_delivery_receipt(receipts, ready)


def _decode_delivery_receipt(text: str) -> list:
    """Parse and validate a canonical :func:`build_delivery_receipt` string.

    Returns a list of ``[product, action, current, target, affected, status,
    reason, failures]`` receipts in document order (``target`` is ``None``
    or a version string, ``affected`` a list of batch ints and ``failures``
    a list of normalized failure rows; see :func:`_decode_failure_rows`).

    Beyond JSON, shape and canonical-format checks, the receipt contract is
    revalidated: each receipt's action is ``publish``, ``rollback`` or
    ``block`` with the same target/affected/failures associations as the
    plan contract; a ``publish`` or ``rollback`` receipt's status is
    ``succeeded`` or ``failed`` while a ``block`` receipt's status is
    ``blocked``; a ``succeeded`` receipt carries an empty ``reason`` and a
    ``failed`` or ``blocked`` receipt a non-empty one; receipts are sorted
    uniquely by product name; and the top-level ``ready`` flag is true
    exactly when every receipt is a ``publish`` whose status is
    ``succeeded`` (an empty receipt set being ``true``).

    :raises ValueError: the JSON syntax or shape is bad, a value violates
        the receipt contract, the text is not the canonical receipt
        encoding, or the top-level ``ready`` flag is inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("receipt is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"receipts",
                                                           "ready"}:
        raise ValueError("receipt top-level value must be an object with "
                         "only 'receipts' and 'ready'")
    raw_receipts = document["receipts"]
    if not isinstance(raw_receipts, list):
        raise ValueError("'receipts' must be an array")
    ready = document["ready"]
    if not isinstance(ready, bool):
        raise ValueError("'ready' must be a boolean")

    receipts = []
    for raw_receipt in raw_receipts:
        if not isinstance(raw_receipt, list) or len(raw_receipt) != 8:
            raise ValueError("each receipt must be an array of eight "
                             "values")
        (product, action, current, target, raw_affected, status, reason,
         raw_failures) = raw_receipt
        if not isinstance(product, str):
            raise ValueError("product name must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product name must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if action not in ("publish", "rollback", "block"):
            raise ValueError("receipt action must be publish, rollback "
                             "or block")
        if not isinstance(current, str):
            raise ValueError("receipt current must be a str")
        if not current or not set(current) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "receipt current must be non-empty and contain only "
                "ASCII alphanumeric characters and ._-"
            )
        if target is not None and not isinstance(target, str):
            raise ValueError("receipt target must be null or a str")
        if target is not None and (
                not target or not set(target) <= _DELIVERY_IDENT_CHARS):
            raise ValueError(
                "receipt target must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        if not isinstance(raw_affected, list):
            raise ValueError("receipt affected must be an array")
        affected = []
        for batch in raw_affected:
            if isinstance(batch, bool) or not isinstance(batch, int):
                raise ValueError("receipt affected batches must be "
                                 "non-bool integers")
            if batch < 0:
                raise ValueError("receipt affected batches must be "
                                 "non-negative")
            affected.append(batch)
        if any(affected[index] >= affected[index + 1]
               for index in range(len(affected) - 1)):
            raise ValueError("receipt affected batches must be strictly "
                             "ascending")
        if status not in ("succeeded", "failed", "blocked"):
            raise ValueError("receipt status must be succeeded, failed "
                             "or blocked")
        if not isinstance(reason, str):
            raise ValueError("receipt reason must be a str")

        failures = _decode_failure_rows(raw_failures)

        # Field associations mirror the plan contract: a publish carries
        # no target delta or failure state; a rollback or block lists at
        # least one affected batch and one failure, and the failures expand
        # exactly the affected batches in batch order.
        if action == "publish":
            if target != current or affected or failures:
                raise ValueError("a publish receipt must target current "
                                 "with no affected batches or failures")
        else:
            if action == "rollback" and target is None:
                raise ValueError("a rollback receipt must carry a "
                                 "non-null target")
            if action == "block" and target is not None:
                raise ValueError("a block receipt must carry a null "
                                 "target")
            if not affected or not failures:
                raise ValueError("a rollback or block receipt must list "
                                 "an affected batch and a failure")
            failure_batches = [failure[0] for failure in failures]
            if any(failure_batches[index] > failure_batches[index + 1]
                   for index in range(len(failure_batches) - 1)):
                raise ValueError("receipt failures must be ordered by "
                                 "batch")
            if sorted(set(failure_batches)) != affected:
                raise ValueError("receipt affected batches must match "
                                 "the batches of its failures")

        # A publish or rollback resolves as succeeded or failed; a block
        # stays blocked. Only a succeeded receipt carries no reason.
        if action in ("publish", "rollback"):
            if status not in ("succeeded", "failed"):
                raise ValueError("a publish or rollback receipt must be "
                                 "succeeded or failed")
        elif status != "blocked":
            raise ValueError("a block receipt must be blocked")
        if status == "succeeded":
            if reason:
                raise ValueError("a succeeded receipt must carry an "
                                 "empty reason")
        elif not reason:
            raise ValueError("a failed or blocked receipt must carry a "
                             "non-empty reason")

        receipts.append([product, action, current, target, affected,
                         status, reason, failures])

    # Receipts inherit the plan's product order: strictly ascending by
    # name with no duplicates; a byte-for-byte re-encode alone would
    # tolerate reordering.
    names = [receipt[0] for receipt in receipts]
    if any(names[index] >= names[index + 1]
           for index in range(len(names) - 1)):
        raise ValueError("receipts must be sorted uniquely by product "
                         "name")

    if ready != all(receipt[1] == "publish" and receipt[5] == "succeeded"
                    for receipt in receipts):
        raise ValueError("'ready' must be true exactly when every "
                         "receipt is a succeeded publish")

    # Byte-for-byte canonical equality rejects whitespace, reordered or
    # duplicate keys, non-six-decimal metric formatting and any other
    # non-canonical spelling.
    if _format_delivery_receipt(receipts, ready) != text:
        raise ValueError("receipt is not the canonical delivery receipt "
                         "encoding")
    return receipts


def _format_merged_receipts(products: list, ready: bool) -> str:
    """Serialize merged per-product receipt rows to the two-key document."""
    parts = ['{"products":[']
    for index, (product, action, current, target, affected, history,
                final, failures) in enumerate(products):
        if index:
            parts.append(",")
        parts.append("[" + _json_string(product) + ","
                     + _json_string(action) + ","
                     + _json_string(current) + ",")
        parts.append("null" if target is None else _json_string(target))
        parts.append(",[" + ",".join(str(batch) for batch in affected)
                     + "],[")
        parts.append(",".join("[" + _json_string(status) + ","
                              + _json_string(reason) + "]"
                              for status, reason in history))
        parts.append("]," + _json_string(final) + ",[")
        parts.append(_format_failure_rows(failures))
        parts.append("]]")
    parts.append('],"ready":')
    parts.append("true}" if ready else "false}")
    return "".join(parts)


def merge_delivery_receipts(receipts: tuple) -> str:
    """Merge canonical delivery receipts into a products-history document.

    ``receipts`` is a tuple of strings, each byte-for-byte matching the
    canonical output of :func:`build_delivery_receipt`: the compact
    document whose sole top-level keys are ``receipts`` and ``ready`` in
    that order, with receipts ``[product, action, current, target,
    affected, status, reason, failures]`` sorted uniquely by product name.

    Every receipt must cover the same products, and the receipts must
    agree on each product's ``action``, ``current``, ``target``,
    ``affected`` and ``failures``; only the per-receipt ``status`` and
    ``reason`` may differ.

    Returns a canonical compact JSON document with exactly the two keys
    ``products`` and ``ready`` in that order. Products are sorted
    lexicographically by name and each product is the array ``[product,
    action, current, target, affected, history, final, failures]`` where
    ``history`` lists every receipt's ``[status, reason]`` pair in input
    receipt order, ``final`` is the status of the last history entry and
    ``failures`` keeps the receipts' original row order. ``ready`` is true
    only when every product's action is ``publish`` and its ``final``
    status is ``succeeded``; an empty ``receipts`` or an empty product set
    yields ``{"products":[],"ready":true}``. Strings use
    ``ensure_ascii=False``, integers are decimal, the four failure metrics
    use exactly six decimal places (negative zero written as
    ``0.000000``), ``NaN``/``Infinity`` never appear and the output has no
    whitespace or trailing newline. The inputs are never modified and
    repeated calls return a byte-identical document.

    :raises TypeError: ``receipts`` is not a tuple or one of its members
        is not a ``str``.
    :raises ValueError: a receipt is not the canonical receipt encoding,
        the receipts do not share the same products, or the per-product
        ``action``, ``current``, ``target``, ``affected`` or ``failures``
        disagree.
    """
    if not isinstance(receipts, tuple):
        raise TypeError("receipts must be a tuple")

    decoded = []
    for receipt in receipts:
        if not isinstance(receipt, str):
            raise TypeError("each receipt must be a str")
        decoded.append(_decode_delivery_receipt(receipt))

    merged = {}
    names = None
    for rows in decoded:
        # Canonical receipts are sorted uniquely by product, so equal
        # product sets appear in the same order.
        if names is None:
            names = [row[0] for row in rows]
        elif [row[0] for row in rows] != names:
            raise ValueError("receipts must share the same products")
        for (product, action, current, target, affected, status, reason,
             failures) in rows:
            entry = merged.get(product)
            if entry is None:
                merged[product] = [product, action, current, target,
                                   affected, [[status, reason]], failures]
            else:
                if (entry[1], entry[2], entry[3], entry[4], entry[6]) != (
                        action, current, target, affected, failures):
                    raise ValueError(f"conflicting receipt fields for "
                                     f"product {product!r}")
                entry[5].append([status, reason])

    products = []
    for product in sorted(merged):
        _, action, current, target, affected, history, failures = (
            merged[product])
        products.append([product, action, current, target, affected,
                         history, history[-1][0], failures])

    ready = all(product[1] == "publish" and product[6] == "succeeded"
                for product in products)
    return _format_merged_receipts(products, ready)


def _decode_receipt_body(action, current, target, raw_affected,
                         raw_history, final, raw_failures) -> list:
    """Validate the seven shared receipt fields of merged/snapshot rows.

    Returns the normalized list
    ``[action, current, target, affected, history, final, failures]`` where
    ``affected`` is a list of batch ints, ``history`` a list of
    ``[status, reason]`` pairs and ``failures`` normalized failure rows (see
    :func:`_decode_failure_rows`).
    """
    if action not in ("publish", "rollback", "block"):
        raise ValueError("receipt action must be publish, rollback or block")
    if not isinstance(current, str):
        raise ValueError("receipt current must be a str")
    if not current or not set(current) <= _DELIVERY_IDENT_CHARS:
        raise ValueError(
            "receipt current must be non-empty and contain only ASCII "
            "alphanumeric characters and ._-"
        )
    if target is not None and not isinstance(target, str):
        raise ValueError("receipt target must be null or a str")
    if target is not None and (
            not target or not set(target) <= _DELIVERY_IDENT_CHARS):
        raise ValueError(
            "receipt target must be non-empty and contain only ASCII "
            "alphanumeric characters and ._-"
        )
    if not isinstance(raw_affected, list):
        raise ValueError("receipt affected must be an array")
    affected = []
    for batch in raw_affected:
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("receipt affected batches must be non-bool "
                             "integers")
        if batch < 0:
            raise ValueError("receipt affected batches must be non-negative")
        affected.append(batch)
    if any(affected[index] >= affected[index + 1]
           for index in range(len(affected) - 1)):
        raise ValueError("receipt affected batches must be strictly "
                         "ascending")

    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("receipt history must be a non-empty array")
    history = []
    for pair in raw_history:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each receipt history entry must be an array "
                             "of two values")
        status, reason = pair
        if action in ("publish", "rollback"):
            if status not in ("succeeded", "failed"):
                raise ValueError("a publish or rollback history status "
                                 "must be succeeded or failed")
        elif status != "blocked":
            raise ValueError("a block history status must be blocked")
        if not isinstance(reason, str):
            raise ValueError("receipt reason must be a str")
        if status == "succeeded":
            if reason:
                raise ValueError("a succeeded history entry must carry an "
                                 "empty reason")
        elif not reason:
            raise ValueError("a failed or blocked history entry must carry "
                             "a non-empty reason")
        history.append([status, reason])

    if not isinstance(final, str) or final != history[-1][0]:
        raise ValueError("receipt final must equal the last history status")

    failures = _decode_failure_rows(raw_failures)

    if action == "publish":
        if target != current or affected or failures:
            raise ValueError("a publish receipt must target current with "
                             "no affected batches or failures")
    else:
        if action == "rollback" and target is None:
            raise ValueError("a rollback receipt must carry a non-null "
                             "target")
        if action == "block" and target is not None:
            raise ValueError("a block receipt must carry a null target")
        if not affected or not failures:
            raise ValueError("a rollback or block receipt must list an "
                             "affected batch and a failure")
        failure_batches = [failure[0] for failure in failures]
        if any(failure_batches[index] > failure_batches[index + 1]
               for index in range(len(failure_batches) - 1)):
            raise ValueError("receipt failures must be ordered by batch")
        if sorted(set(failure_batches)) != affected:
            raise ValueError("receipt affected batches must match the "
                             "batches of its failures")

    return [action, current, target, affected, history, final, failures]


def _decode_merged_receipts(text: str) -> list:
    """Parse and validate a canonical :func:`merge_delivery_receipts` string.

    Returns a list of
    ``[product, action, current, target, affected, history, final,
    failures]`` product rows in document order.

    Beyond the JSON, shape and canonical-format checks performed by the
    merge output contract, every history entry's status/reason must obey
    the per-action rules of a canonical delivery receipt, ``final`` must
    equal the last history status, the action field associations must hold
    (a ``publish`` targets ``current`` with empty state while a
    ``rollback``/``block`` expands its affected batches through its
    failures), rows must be sorted uniquely by product name, and the
    top-level ``ready`` flag must be true exactly when every product is a
    ``publish`` whose final status is ``succeeded`` (an empty product set
    being ``true``).

    :raises ValueError: the JSON syntax or shape is bad, a value violates
        the merged-receipt contract, the text is not the canonical
        merged-receipt encoding, or the top-level ``ready`` flag is
        inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("receipts are not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"products",
                                                           "ready"}:
        raise ValueError("merged receipts top-level value must be an object "
                         "with only 'products' and 'ready'")
    raw_products = document["products"]
    if not isinstance(raw_products, list):
        raise ValueError("'products' must be an array")
    ready = document["ready"]
    if not isinstance(ready, bool):
        raise ValueError("'ready' must be a boolean")

    products = []
    for raw_product in raw_products:
        if not isinstance(raw_product, list) or len(raw_product) != 8:
            raise ValueError("each merged product must be an array of "
                             "eight values")
        (product, action, current, target, raw_affected, raw_history,
         final, raw_failures) = raw_product
        if not isinstance(product, str):
            raise ValueError("product name must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "product name must be non-empty and contain only ASCII "
                "alphanumeric characters and ._-"
            )
        body = _decode_receipt_body(action, current, target, raw_affected,
                                    raw_history, final, raw_failures)
        products.append([product] + body)

    names = [product[0] for product in products]
    if any(names[index] >= names[index + 1]
           for index in range(len(names) - 1)):
        raise ValueError("products must be sorted uniquely by product name")

    if ready != all(product[1] == "publish" and product[6] == "succeeded"
                    for product in products):
        raise ValueError("'ready' must be true exactly when every product "
                         "is a succeeded publish")

    if _format_merged_receipts(products, ready) != text:
        raise ValueError("receipts are not the canonical merged-receipts "
                         "encoding")
    return products


def _delivery_gaps(batches: list) -> list:
    """Compress batch numbers missing from ``[0, last]`` into closed runs."""
    gaps = []
    expected = 0
    for batch in batches:
        if batch > expected:
            gaps.append([expected, batch - 1])
        expected = batch + 1
    return gaps


def _format_snapshot_receipt(receipt: list) -> str:
    """Serialize the seven receipt fields of a snapshot product item."""
    action, current, target, affected, history, final, failures = receipt
    parts = ["[" + _json_string(action) + "," + _json_string(current) + ","]
    parts.append("null" if target is None else _json_string(target))
    parts.append(",[" + ",".join(str(batch) for batch in affected) + "],[")
    parts.append(",".join("[" + _json_string(status) + ","
                          + _json_string(reason) + "]"
                          for status, reason in history))
    parts.append("]," + _json_string(final) + ",[")
    parts.append(_format_failure_rows(failures))
    parts.append("]]")
    return "".join(parts)


def _format_delivery_snapshot(products: list, ready: bool) -> str:
    """Serialize snapshot product items to the two-key snapshot document."""
    parts = ['{"products":[']
    for index, (product, versions, gaps, receipt) in enumerate(products):
        if index:
            parts.append(",")
        parts.append("[" + _json_string(product) + ",[")
        for version_index, (batch, previous, version,
                            passed) in enumerate(versions):
            if version_index:
                parts.append(",")
            parts.append("[" + str(batch) + ",")
            parts.append("null" if previous is None
                         else _json_string(previous))
            parts.append("," + _json_string(version) + ",")
            parts.append("true" if passed else "false")
            parts.append("]")
        parts.append("],[")
        for gap_index, (first, last) in enumerate(gaps):
            if gap_index:
                parts.append(",")
            parts.append("[" + str(first) + "," + str(last) + "]")
        parts.append("],")
        parts.append("null" if receipt is None
                     else _format_snapshot_receipt(receipt))
        parts.append("]")
    parts.append('],"ready":')
    parts.append("true}" if ready else "false}")
    return "".join(parts)


def build_delivery_snapshot(changes: str, receipts: str) -> str:
    """Build a per-product delivery snapshot from changes and merged receipts.

    ``changes`` must be a ``str`` byte-for-byte matching the canonical output
    of :func:`merge_delivery_manifests` and ``receipts`` a ``str``
    byte-for-byte matching the canonical output of
    :func:`merge_delivery_receipts`.

    Returns a canonical compact JSON document with exactly the two keys
    ``products`` and ``ready`` in that order. Products are sorted
    lexicographically by name (every product occurring in ``changes``
    appears) and each product is the array
    ``[product, versions, gaps, receipt]``. ``versions`` lists the
    product's changes in ascending ``batch`` order as
    ``[batch, previous, version, passed]``; ``gaps`` compresses the batch
    numbers missing from the closed interval ``0``..greatest batch into
    maximal closed intervals ``[first, last]`` (``[]`` when none are
    missing); ``receipt`` is ``null`` when the merged receipts carry no
    row for the product and otherwise the merged-receipt product row with
    its ``product`` field removed, i.e.
    ``[action, current, target, affected, history, final, failures]``. A
    product is ready only when it has no gaps, has a receipt whose
    ``current`` equals the version of its final change, that change
    passes, the receipt action is ``publish`` and its ``final`` status is
    ``succeeded``; the top-level ``ready`` is the logical AND over every
    product and is ``true`` for an empty product set. Integers are
    decimal, the four failure metrics use exactly six decimal places
    (negative zero written as ``0.000000``), ``NaN``/``Infinity`` never
    appear and the output has no whitespace or trailing newline.

    :raises TypeError: ``changes`` or ``receipts`` is not a ``str``.
    :raises ValueError: either document is not its canonical encoding, a
        receipt names a product absent from ``changes``, or a receipt's
        ``current`` does not match the product's final change version.
    """
    if not isinstance(changes, str):
        raise TypeError("changes must be a str")
    if not isinstance(receipts, str):
        raise TypeError("receipts must be a str")

    decoded_changes, _changes_releasable = _decode_delivery_changes(changes)
    merged = _decode_merged_receipts(receipts)

    by_product = {}
    for batch, product, previous, version, passed, _failures in (
            decoded_changes):
        by_product.setdefault(product, []).append(
            [batch, previous, version, passed])

    receipt_by_product = {row[0]: row[1:] for row in merged}

    products = []
    flags = []
    for product in sorted(by_product):
        versions = by_product[product]
        receipt = receipt_by_product.pop(product, None)
        gaps = _delivery_gaps([version[0] for version in versions])

        final_version = versions[-1][2]
        final_passed = versions[-1][3]
        if receipt is not None and receipt[1] != final_version:
            raise ValueError(
                f"receipt current {receipt[1]!r} for product {product!r} "
                f"must equal its final change version {final_version!r}"
            )
        product_ready = (
            not gaps and receipt is not None and final_passed
            and receipt[0] == "publish" and receipt[5] == "succeeded"
        )
        flags.append(product_ready)
        products.append([product, versions, gaps, receipt])

    if receipt_by_product:
        product = next(iter(receipt_by_product))
        raise ValueError(f"unknown receipt product {product!r}")

    ready = all(flags)
    return _format_delivery_snapshot(products, ready)


def _decode_delivery_snapshot(text: str) -> tuple:
    """Parse and validate a canonical :func:`build_delivery_snapshot` string.

    Returns ``(products, ready)`` where ``products`` is a list of
    ``[product, versions, gaps, receipt]`` items (``versions`` a list of
    ``[batch, previous, version, passed]`` rows, ``gaps`` a list of
    ``[first, last]`` intervals and ``receipt`` ``None`` or the seven
    normalized receipt fields), in product order.

    :raises ValueError: the JSON syntax or shape is bad, a value violates
        the snapshot contract, the text is not the canonical snapshot
        encoding, or the top-level ``ready`` flag is inconsistent.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant,
                              parse_float=Decimal)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("snapshot is not valid JSON") from exc

    if not isinstance(document, dict) or set(document) != {"products",
                                                           "ready"}:
        raise ValueError("snapshot top-level value must be an object with "
                         "only 'products' and 'ready'")
    raw_products = document["products"]
    if not isinstance(raw_products, list):
        raise ValueError("snapshot 'products' must be an array")
    ready = document["ready"]
    if not isinstance(ready, bool):
        raise ValueError("snapshot 'ready' must be a boolean")

    products = []
    for raw_product in raw_products:
        if not isinstance(raw_product, list) or len(raw_product) != 4:
            raise ValueError("each snapshot product must be an array of "
                             "four values")
        product, raw_versions, raw_gaps, raw_receipt = raw_product
        if not isinstance(product, str):
            raise ValueError("snapshot product name must be a str")
        if not product or not set(product) <= _DELIVERY_IDENT_CHARS:
            raise ValueError(
                "snapshot product name must be non-empty and contain only "
                "ASCII alphanumeric characters and ._-"
            )

        if not isinstance(raw_versions, list) or not raw_versions:
            raise ValueError("snapshot versions must be a non-empty array")
        versions = []
        previous_by_product = None
        seen_versions = set()
        for raw_version in raw_versions:
            if not isinstance(raw_version, list) or len(raw_version) != 4:
                raise ValueError("each snapshot version must be an array "
                                 "of four values")
            batch, previous, version, passed = raw_version
            if isinstance(batch, bool) or not isinstance(batch, int):
                raise ValueError("snapshot batch must be a non-bool "
                                 "integer")
            if batch < 0:
                raise ValueError("snapshot batch must be non-negative")
            if versions and batch <= versions[-1][0]:
                raise ValueError("snapshot versions must be sorted "
                                 "uniquely by batch")
            if previous is not None and not isinstance(previous, str):
                raise ValueError("snapshot previous must be null or a str")
            if previous is not None and (
                    not previous or not set(previous)
                    <= _DELIVERY_IDENT_CHARS):
                raise ValueError(
                    "snapshot previous must be non-empty and contain only "
                    "ASCII alphanumeric characters and ._-"
                )
            if previous != previous_by_product:
                raise ValueError("snapshot previous must be the version "
                                 "of the product's preceding batch")
            if not isinstance(version, str):
                raise ValueError("snapshot version must be a str")
            if not version or not set(version) <= _DELIVERY_IDENT_CHARS:
                raise ValueError(
                    "snapshot version must be non-empty and contain only "
                    "ASCII alphanumeric characters and ._-"
                )
            if version in seen_versions:
                raise ValueError(f"version {version!r} is reused by "
                                 f"product {product!r}")
            if not isinstance(passed, bool):
                raise ValueError("snapshot passed flag must be a boolean")
            versions.append([batch, previous, version, passed])
            seen_versions.add(version)
            previous_by_product = version

        if not isinstance(raw_gaps, list):
            raise ValueError("snapshot gaps must be an array")
        gaps = []
        for raw_gap in raw_gaps:
            if not isinstance(raw_gap, list) or len(raw_gap) != 2:
                raise ValueError("each snapshot gap must be an array of "
                                 "two values")
            first, last = raw_gap
            if isinstance(first, bool) or not isinstance(first, int):
                raise ValueError("snapshot gap first must be a non-bool "
                                 "integer")
            if isinstance(last, bool) or not isinstance(last, int):
                raise ValueError("snapshot gap last must be a non-bool "
                                 "integer")
            if first < 0 or first > last:
                raise ValueError("snapshot gap bounds must be non-negative "
                                 "with first <= last")
            if gaps and first <= gaps[-1][1]:
                raise ValueError("snapshot gaps must be non-overlapping "
                                 "ascending closed intervals")
            gaps.append([first, last])

        if gaps != _delivery_gaps([version[0] for version in versions]):
            raise ValueError("snapshot gaps must be exactly the missing "
                             "batches between zero and the final batch")

        receipt = None
        if raw_receipt is not None:
            if not isinstance(raw_receipt, list) or len(raw_receipt) != 7:
                raise ValueError("snapshot receipt must be null or an "
                                 "array of seven values")
            action, current, target, raw_affected, raw_history, final, \
                raw_failures = raw_receipt
            receipt = _decode_receipt_body(
                action, current, target, raw_affected, raw_history, final,
                raw_failures)
            if receipt[1] != versions[-1][2]:
                raise ValueError("snapshot receipt current must equal the "
                                 "product's final change version")

        product_ready = bool(
            not gaps and receipt is not None and versions[-1][3]
            and receipt[0] == "publish" and receipt[5] == "succeeded"
        )
        products.append([product, versions, gaps, receipt, product_ready])

    names = [product[0] for product in products]
    if any(names[index] >= names[index + 1]
           for index in range(len(names) - 1)):
        raise ValueError("snapshot products must be sorted uniquely by "
                         "product name")

    if ready != all(product[4] for product in products):
        raise ValueError("snapshot 'ready' must be true exactly when every "
                         "product has no gaps, a succeeded publish receipt "
                         "and a passing final change")

    normalized = [[product[0], product[1], product[2], product[3]]
                  for product in products]
    if _format_delivery_snapshot(normalized, ready) != text:
        raise ValueError("snapshot is not the canonical delivery snapshot "
                         "encoding")
    return normalized, ready


def _snapshot_to_tuples(value):
    """Recursively convert snapshot lists to tuples and Decimals to floats."""
    if isinstance(value, list):
        return tuple(_snapshot_to_tuples(item) for item in value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def query_delivery_snapshot(snapshot: str, product: str) -> tuple | None:
    """Return one product item from a delivery snapshot as nested tuples.

    ``snapshot`` must be a ``str`` byte-for-byte matching the canonical
    output of :func:`build_delivery_snapshot` and ``product`` a ``str``.
    When the snapshot contains the product, returns its
    ``[product, versions, gaps, receipt]`` item with every array
    recursively converted to a tuple (JSON ``null`` becomes ``None`` and
    failure-summary metrics become ``float``); the ``versions`` tuples are
    ``(batch, previous, version, passed)`` rows, ``gaps`` tuples are
    ``(first, last)`` intervals and a present ``receipt`` is the
    seven-field tuple
    ``(action, current, target, affected, history, final, failures)``.
    Returns ``None`` when no product with that name exists (an empty
    snapshot never matches). The input is never modified.

    :raises TypeError: ``snapshot`` or ``product`` is not a ``str``.
    :raises ValueError: ``snapshot`` is not the canonical delivery
        snapshot encoding.
    """
    if not isinstance(snapshot, str):
        raise TypeError("snapshot must be a str")
    if not isinstance(product, str):
        raise TypeError("product must be a str")

    products, _ready = _decode_delivery_snapshot(snapshot)
    for item in products:
        if item[0] == product:
            return _snapshot_to_tuples(item)
    return None


def query_delivery_range(snapshot: str, product: str, batch_min: int,
                         batch_max: int) -> tuple | None:
    """Return one product's delivery state restricted to a batch range.

    ``snapshot`` must be a ``str`` byte-for-byte matching the canonical
    output of :func:`build_delivery_snapshot` (including its derived
    ``previous`` chains, ``gaps`` intervals and top-level ``ready`` flag,
    which are all revalidated), ``product`` a ``str`` and ``batch_min`` /
    ``batch_max`` non-bool ``int`` bounds with
    ``0 <= batch_min <= batch_max``.

    Returns ``None`` when the snapshot has no product with that name.
    Otherwise returns the four-item tuple ``(versions, gaps, receipt,
    ready)`` where:

    - ``versions`` is the product's ``(batch, previous, version, passed)``
      rows whose ``batch`` lies in the closed interval
      ``[batch_min, batch_max]``, kept in ascending ``batch`` order;
    - ``gaps`` compresses the batch numbers of the interval that have no
      version into maximal closed ``(first, last)`` interval tuples
      (``()`` when none are missing, and ``((batch_min, batch_max),)``
      when the interval contains no version at all);
    - ``receipt`` is ``None`` when the product carries no receipt and
      otherwise the five-field tuple ``(action, current, target, history,
      final)`` with ``history`` the receipt's ``(status, reason)`` pairs
      in their original order;
    - ``ready`` is recomputed from the product's complete chain — true
      exactly when the full chain has no gaps, its final version passes
      and it carries a receipt whose action is ``publish`` with ``final``
      status ``succeeded`` — and is unaffected by the range truncation.

    The input is never modified.

    :raises TypeError: ``snapshot`` or ``product`` is not a ``str``, or a
        batch bound is not a non-bool ``int``.
    :raises ValueError: ``snapshot`` is not the canonical delivery
        snapshot encoding, a batch bound is negative, or
        ``batch_min > batch_max``.
    """
    if not isinstance(snapshot, str):
        raise TypeError("snapshot must be a str")
    if not isinstance(product, str):
        raise TypeError("product must be a str")
    if isinstance(batch_min, bool) or not isinstance(batch_min, int):
        raise TypeError("batch_min must be a non-bool int")
    if isinstance(batch_max, bool) or not isinstance(batch_max, int):
        raise TypeError("batch_max must be a non-bool int")
    if batch_min < 0 or batch_max < 0:
        raise ValueError("batch bounds must be non-negative")
    if batch_min > batch_max:
        raise ValueError("batch_min must not exceed batch_max")

    products, _snapshot_ready = _decode_delivery_snapshot(snapshot)
    for name, versions, chain_gaps, receipt in products:
        if name != product:
            continue
        ranged = tuple(
            (batch, previous, version, passed)
            for batch, previous, version, passed in versions
            if batch_min <= batch <= batch_max)
        gaps = []
        expected = batch_min
        for row in ranged:
            if row[0] > expected:
                gaps.append((expected, row[0] - 1))
            expected = row[0] + 1
        if expected <= batch_max:
            gaps.append((expected, batch_max))
        if receipt is None:
            ranged_receipt = None
        else:
            action, current, target, _affected, history, final, \
                _failures = receipt
            ranged_receipt = (action, current, target,
                              tuple((status, reason)
                                    for status, reason in history),
                              final)
        ready = bool(
            not chain_gaps and receipt is not None and versions[-1][3]
            and receipt[0] == "publish" and receipt[5] == "succeeded")
        return (ranged, tuple(gaps), ranged_receipt, ready)
    return None


def update_delivery_snapshot(snapshot: str, changes: str,
                             receipts: str) -> str:
    """Apply new delivery changes and receipts to a delivery snapshot.

    ``snapshot`` must be a ``str`` byte-for-byte matching the canonical
    output of :func:`build_delivery_snapshot`, ``changes`` a ``str``
    byte-for-byte matching the canonical output of
    :func:`merge_delivery_manifests` and ``receipts`` a ``str``
    byte-for-byte matching the canonical output of
    :func:`merge_delivery_receipts`.

    The snapshot's existing versions and the new changes are unioned by
    ``(product, batch)``: an entry present in both must agree on
    ``version`` and ``passed`` (a conflict raises :class:`ValueError`), and
    a product's ``version`` must never repeat across its batches. Each
    product's ``previous`` fields are then recomputed in ascending
    ``batch`` order (``null`` for its first batch, the preceding batch's
    ``version`` afterwards). Every product named by ``receipts`` must exist
    in the merged set and its ``current`` must equal that product's final
    version (otherwise :class:`ValueError`); its seven-field receipt
    replaces the snapshot's receipt. A product without a new receipt keeps
    its old receipt only when that receipt's ``current`` still equals the
    merged final version, and otherwise carries ``null``.

    Returns the canonical compact JSON snapshot document exactly as
    specified for :func:`build_delivery_snapshot`: products sorted
    lexicographically by name, ``gaps`` recomputed as the maximal closed
    intervals of batch numbers missing from ``0``..greatest batch, and
    ``ready`` true exactly when every product has no gaps, a passing final
    change and a receipt whose action is ``publish`` with ``final`` status
    ``succeeded`` (an empty product set being ``true``). Integers are
    decimal, the four failure metrics use exactly six decimal places
    (negative zero written as ``0.000000``), ``NaN``/``Infinity`` never
    appear and the output has no whitespace or trailing newline. The
    inputs are never modified and repeated calls return a byte-identical
    document.

    :raises TypeError: ``snapshot``, ``changes`` or ``receipts`` is not a
        ``str``.
    :raises ValueError: an input is not its canonical encoding, a
        ``(product, batch)`` entry conflicts, a product reuses a version
        across batches, or a receipt names an unknown product or a
        ``current`` that is not the product's final version.
    """
    if not isinstance(snapshot, str):
        raise TypeError("snapshot must be a str")
    if not isinstance(changes, str):
        raise TypeError("changes must be a str")
    if not isinstance(receipts, str):
        raise TypeError("receipts must be a str")

    old_products, _old_ready = _decode_delivery_snapshot(snapshot)
    decoded_changes, _changes_releasable = _decode_delivery_changes(changes)
    merged_receipts = _decode_merged_receipts(receipts)

    rows_by_product = {}
    old_receipt_by_product = {}
    for product, versions, _gaps, receipt in old_products:
        rows = rows_by_product.setdefault(product, {})
        for batch, _previous, version, passed in versions:
            rows[batch] = [version, passed]
        old_receipt_by_product[product] = receipt

    for batch, product, _previous, version, passed, _failures in (
            decoded_changes):
        rows = rows_by_product.setdefault(product, {})
        existing = rows.get(batch)
        if existing is not None and existing != [version, passed]:
            raise ValueError(f"conflicting (product, batch) entries: "
                             f"{(product, batch)!r}")
        rows[batch] = [version, passed]

    receipt_by_product = {row[0]: row[1:] for row in merged_receipts}

    products = []
    flags = []
    for product in sorted(rows_by_product):
        rows = rows_by_product[product]
        versions = []
        seen_versions = set()
        previous = None
        for batch in sorted(rows):
            version, passed = rows[batch]
            if version in seen_versions:
                raise ValueError(f"version {version!r} is reused by "
                                 f"product {product!r}")
            seen_versions.add(version)
            versions.append([batch, previous, version, passed])
            previous = version
        gaps = _delivery_gaps([version[0] for version in versions])

        final_version = versions[-1][2]
        final_passed = versions[-1][3]
        receipt = receipt_by_product.pop(product, None)
        if receipt is None:
            old_receipt = old_receipt_by_product.get(product)
            if old_receipt is not None and old_receipt[1] == final_version:
                receipt = old_receipt
        elif receipt[1] != final_version:
            raise ValueError(
                f"receipt current {receipt[1]!r} for product {product!r} "
                f"must equal its final change version {final_version!r}"
            )
        product_ready = (
            not gaps and receipt is not None and final_passed
            and receipt[0] == "publish" and receipt[5] == "succeeded"
        )
        flags.append(product_ready)
        products.append([product, versions, gaps, receipt])

    if receipt_by_product:
        product = next(iter(receipt_by_product))
        raise ValueError(f"unknown receipt product {product!r}")

    ready = all(flags)
    return _format_delivery_snapshot(products, ready)
