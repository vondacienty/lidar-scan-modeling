"""Streaming iteration over LAS/LAZ point files."""

from __future__ import annotations

import math
from collections.abc import Iterator
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import laspy
import numpy as np

try:
    import lazrs
except ImportError:  # pragma: no cover - lazrs is a declared dependency
    lazrs = None

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_CHUNK_SIZE = 65536
_SUFFIXES = (".las", ".laz")
_REQUIRED_DIMENSIONS = frozenset({"X", "Y", "Z", "intensity"})
_NUMERIC_DIMENSION_KINDS = frozenset({
    laspy.DimensionKind.SignedInteger,
    laspy.DimensionKind.UnsignedInteger,
    laspy.DimensionKind.FloatingPoint,
})

_DECODE_ERRORS = (laspy.errors.LaspyException,) + (
    (lazrs.LazrsError,) if lazrs is not None else ()
)


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    result = float(value.quantize(_QUANTUM))
    return 0.0 if result == 0.0 else result


def iter_las(path: str) -> Iterator[tuple]:
    """Return a one-shot iterator over the points of a LAS/LAZ file.

    ``path`` must be a ``str`` ending in ``.las`` or ``.laz`` (case-insensitive);
    anything else raises ``TypeError``/``ValueError`` at call time. The returned
    iterator yields tuples of up to 65536 points in file record order — every
    chunk except the last holds exactly 65536 points, and an empty file yields
    no chunks at all. Points are streamed from disk; the file is never loaded
    whole, and its handle is closed on exhaustion or error. The iterator is
    single-use: once exhausted it never reopens the file.

    Each point is a ``(x, y, z, intensity, sigma)`` tuple. Coordinates are the
    raw integer X/Y/Z records scaled and offset per the file header, computed
    with ``Decimal`` arithmetic (precision 50, ROUND_HALF_EVEN); ``intensity``
    is the standard field as ``int``; ``sigma`` is the decoded value of a
    numeric extra dimension named ``sigma`` when present, else ``1.0``.
    ``x``/``y``/``z``/``sigma`` are quantized to six decimal places.

    Raises ``OSError`` if the file cannot be opened, ``RuntimeError`` if no
    LAZ decompression backend is available, and ``ValueError`` if the header
    or records cannot be parsed, required standard fields are missing, the
    scale/offset values are not finite, or a ``sigma`` value is not finite
    and positive.
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if not path.lower().endswith(_SUFFIXES):
        raise ValueError("path must end with .las or .laz (case-insensitive)")
    return _iter_las(path)


def _iter_las(path: str) -> Iterator[tuple]:
    reader = None
    try:
        try:
            reader = laspy.open(path)
        except OSError:
            raise
        except laspy.errors.LaspyException as exc:
            raise ValueError(f"cannot parse LAS/LAZ header: {exc}") from exc

        header = reader.header
        scales = [float(v) for v in header.scales]
        offsets = [float(v) for v in header.offsets]
        if not all(math.isfinite(v) for v in scales + offsets):
            raise ValueError("LAS scale/offset values must be finite")

        point_format = header.point_format
        missing = _REQUIRED_DIMENSIONS - set(point_format.standard_dimension_names)
        if missing:
            raise ValueError(f"missing required standard fields: {sorted(missing)}")

        has_sigma = False
        if "sigma" in set(point_format.extra_dimension_names):
            kind = point_format.dimension_by_name("sigma").kind
            has_sigma = kind in _NUMERIC_DIMENSION_KINDS

        if header.are_points_compressed and not laspy.LazBackend.detect_available():
            raise RuntimeError("no LAZ decompression backend is available")

        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            dscales = [Decimal(str(v)) for v in scales]
            doffsets = [Decimal(str(v)) for v in offsets]

            chunk: list[tuple] = []
            try:
                for points in reader.chunk_iterator(_CHUNK_SIZE):
                    raw_x = points["X"]
                    raw_y = points["Y"]
                    raw_z = points["Z"]
                    intensities = points["intensity"]
                    sigmas = (
                        [float(v) for v in np.asarray(points["sigma"], dtype=np.float64)]
                        if has_sigma else None
                    )

                    for i in range(len(points)):
                        x = Decimal(str(int(raw_x[i]))) * dscales[0] + doffsets[0]
                        y = Decimal(str(int(raw_y[i]))) * dscales[1] + doffsets[1]
                        z = Decimal(str(int(raw_z[i]))) * dscales[2] + doffsets[2]

                        if sigmas is None:
                            sigma = Decimal(1)
                        else:
                            value = sigmas[i]
                            if not math.isfinite(value) or value <= 0:
                                raise ValueError("sigma values must be finite and positive")
                            sigma = Decimal(str(value))

                        chunk.append((
                            _quantize(x), _quantize(y), _quantize(z),
                            int(intensities[i]), _quantize(sigma),
                        ))
                        if len(chunk) == _CHUNK_SIZE:
                            yield tuple(chunk)
                            chunk = []
            except _DECODE_ERRORS as exc:
                raise ValueError(f"cannot parse LAS/LAZ records: {exc}") from exc

            if chunk:
                yield tuple(chunk)
    finally:
        if reader is not None:
            reader.close()
