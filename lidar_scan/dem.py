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


def _as_index(value, name: str) -> int:
    """Validate a grid index or count field as a non-bool int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool int")
    return value


def merge_dems(dems: tuple) -> tuple:
    """Merge DEM tuples produced by :func:`build_dem` over the union of cells.

    ``dems`` must be a tuple (possibly empty) of DEM tuples, each as returned
    by :func:`build_dem` (also possibly empty). Every cell must be a 5-tuple
    ``(ix, iy, z, sigma, count)`` sorted strictly increasingly by ``(ix, iy)``
    with no duplicates; ``ix``, ``iy`` and ``count`` must be non-bool ints and
    ``z`` and ``sigma`` must be finite floats with ``sigma > 0``. A non-tuple
    ``dems`` raises ``TypeError``; malformed members or cells, non-increasing
    indices or invalid fields raise ``ValueError``.

    Cells are combined over the union of ``(ix, iy)`` coordinates, so a
    coordinate missing from some members is still emitted. For coincident
    coordinates the weight is ``w = 1 / sigma**2``, the merged z is
    ``sum(w * z) / sum(w)``, the merged sigma is ``sqrt(1 / sum(w))`` and the
    merged count is the exact integer sum of the member counts. Decimal
    arithmetic (precision 50, ROUND_HALF_EVEN) accumulates weights and
    weighted terms in ascending value order, so results are independent of
    the order of ``dems`` and their members.

    Returns a tuple of ``(ix, iy, z, sigma, count)`` tuples sorted
    lexicographically by ``(ix, iy)``, with ``z`` and ``sigma`` quantized to
    six decimal places; an empty union returns ``()``. Inputs are not
    modified.
    """
    if not isinstance(dems, tuple):
        raise TypeError("dems must be a tuple of DEM tuples")

    groups: dict[tuple[int, int], list] = {}

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        for member in dems:
            if not isinstance(member, tuple):
                raise ValueError("each DEM must be a tuple of "
                                 "(ix, iy, z, sigma, count) cells")

            previous_key = None
            for cell in member:
                if not isinstance(cell, tuple) or len(cell) != 5:
                    raise ValueError("each cell must be a 5-tuple "
                                     "(ix, iy, z, sigma, count)")
                ix, iy, z, sigma, count = cell
                if isinstance(ix, bool) or not isinstance(ix, int):
                    raise ValueError("ix must be a non-bool int")
                if isinstance(iy, bool) or not isinstance(iy, int):
                    raise ValueError("iy must be a non-bool int")
                if not isinstance(z, float) or not math.isfinite(z):
                    raise ValueError("z must be a finite float")
                if not isinstance(sigma, float) or not math.isfinite(sigma):
                    raise ValueError("sigma must be a finite float")
                if sigma <= 0.0:
                    raise ValueError("sigma must be positive")
                if isinstance(count, bool) or not isinstance(count, int):
                    raise ValueError("count must be a non-bool int")

                key = (ix, iy)
                if previous_key is not None and key <= previous_key:
                    raise ValueError("cells must be sorted strictly increasingly "
                                     "by (ix, iy) with no duplicates")
                previous_key = key

                dz = Decimal(str(z))
                dsigma = Decimal(str(sigma))
                weight = Decimal(1) / (dsigma * dsigma)

                entry = groups.get(key)
                if entry is None:
                    entry = [0, [], []]
                    groups[key] = entry
                entry[0] += count
                entry[1].append(weight)
                entry[2].append(weight * dz)

        result = []
        for key, (total_count, weights, weighted_zs) in groups.items():
            sum_w = _sorted_sum(weights)
            merged_z = _sorted_sum(weighted_zs) / sum_w
            merged_sigma = (Decimal(1) / sum_w).sqrt()
            result.append((
                key[0], key[1],
                _quantize(merged_z), _quantize(merged_sigma),
                total_count,
            ))

    result.sort(key=lambda item: (item[0], item[1]))
    return tuple(result)


def _read_cells(cells: Iterable[tuple | list], name: str) -> dict:
    """Consume a single-pass iterable of ``(ix, iy, z, sigma, count)`` cells."""
    try:
        cell_iter = iter(cells)
    except TypeError:
        raise TypeError(f"{name} must be an iterable of cells") from None

    result: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    for cell in cell_iter:
        if not isinstance(cell, (tuple, list)) or len(cell) != 5:
            raise TypeError("each cell must be a tuple or list of 5 items "
                            "(ix, iy, z, sigma, count)")
        ix = _as_index(cell[0], "ix")
        iy = _as_index(cell[1], "iy")
        dz = _as_decimal(cell[2])
        dsigma = _as_decimal(cell[3])
        _as_index(cell[4], "count")
        if dsigma <= 0:
            raise ValueError("sigma must be positive")
        key = (ix, iy)
        if key in result:
            raise ValueError(f"duplicate cell index ({ix}, {iy})")
        result[key] = (dz, dsigma)
    return result


def assess_dem(estimate: Iterable[tuple | list],
               reference: Iterable[tuple | list]) -> tuple:
    """Assess an estimated DEM against a reference DEM cell by cell.

    Both inputs are single-pass iterables of 5-item ``(ix, iy, z, sigma,
    count)`` tuples/lists, where ``ix``, ``iy`` and ``count`` are non-bool
    ints and ``z`` and ``sigma`` are finite non-bool int/float with
    ``sigma > 0``. Duplicate ``(ix, iy)`` indices within one input, or
    differing index sets between the two inputs, raise ``ValueError``.

    Returns a tuple of ``(ix, iy, bias, abs_error, combined_sigma, z_score)``
    tuples sorted lexicographically by ``(ix, iy)``, where
    ``bias = estimate.z - reference.z``, ``abs_error = abs(bias)``,
    ``combined_sigma = sqrt(estimate.sigma**2 + reference.sigma**2)`` and
    ``z_score = bias / combined_sigma``. Results are computed with Decimal
    arithmetic, quantized to six decimal places and are independent of input
    order. Two empty inputs return ``()``.
    """
    est_cells = _read_cells(estimate, "estimate")
    ref_cells = _read_cells(reference, "reference")

    if est_cells.keys() != ref_cells.keys():
        raise ValueError("estimate and reference must contain the same "
                         "set of (ix, iy) cell indices")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        result = []
        for key in sorted(est_cells):
            est_z, est_sigma = est_cells[key]
            ref_z, ref_sigma = ref_cells[key]
            bias = est_z - ref_z
            combined = (est_sigma * est_sigma + ref_sigma * ref_sigma).sqrt()
            result.append((
                key[0], key[1],
                _quantize(bias), _quantize(abs(bias)),
                _quantize(combined), _quantize(bias / combined),
            ))

    return tuple(result)


def build_dsm(points: Iterable[tuple | list], cell_size: int | float = 1.0) -> tuple:
    """Build a 2D DSM keeping the highest-z point of each grid cell.

    Each point is a 5-item ``(x, y, z, intensity, sigma)`` tuple/list. Coordinates
    divided by ``cell_size`` and floored toward negative infinity give the cell
    index ``(ix, iy)``. Within a cell the point with the highest z is kept;
    exact z ties are resolved by ascending ``(sigma, intensity, x, y)``.

    Returns a tuple of ``(ix, iy, z, sigma, count)`` tuples, where ``count`` is
    the number of points in the cell, sorted lexicographically by ``(ix, iy)``;
    an empty input returns ``()``.
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
