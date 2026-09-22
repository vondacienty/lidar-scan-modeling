"""Streaming reader for E57 LiDAR point cloud files.

Only :func:`iter_e57` is public; the rest is an implementation detail of the
iterator it returns.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

# pye57 wraps the libE57Format C++ library; it is a declared runtime dependency
# but its absence must be reported as RuntimeError rather than breaking the
# package import (mirroring the optional lazrs decoder in .las).
try:
    import numpy as np
    from pye57 import libe57
except ModuleNotFoundError:  # pragma: no cover - declared via pyproject dependencies
    np = None
    libe57 = None

# Bound as a module attribute so the OS-level open probe can be observed in tests.
_open = open

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_DEFAULT_CHUNK_SIZE = 65536
_SUPPORTED_EXTENSION = ".e57"
_REQUIRED_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")
_INTENSITY_FIELD = "intensity"


def _quantize_coordinate(value: float) -> float:
    """Decimal(str(value)) quantized to six decimals (negative zero normalized)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("E57 coordinates must be finite")
    try:
        decimal_value = Decimal(str(value))
        result = float(decimal_value.quantize(_QUANTUM))
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise ValueError(f"E57 coordinate could not be represented: {value!r}") from exc
    result = 0.0 if result == 0.0 else result
    if not math.isfinite(result):
        raise ValueError("E57 coordinates must be finite")
    return result


def _as_intensity(value: float) -> int:
    """Convert a decoded intensity value to a plain finite integer."""
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("E57 intensity must be a finite integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("E57 intensity must be a finite integer") from exc


class _E57BlockIterator:
    """One-shot iterator over fixed-size point tuples read from an E57 file.

    Points are concatenated across scans in scan index order and, within each
    scan, in compressed-vector record order. Only fixed-capacity chunks of at
    most ``chunk_size`` records are decoded at a time; the whole file is never
    loaded.
    """

    def __init__(self, path: str, chunk_size: int) -> None:
        self._path = path
        self._chunk_size = chunk_size
        self._image_file = None
        self._data3d = None
        self._closed = False
        self._scan_count = 0
        self._scan_index = 0
        self._reader = None
        self._point_count = 0
        self._read_count = 0
        self._fields: tuple[str, ...] = ()
        self._buffers: dict = {}
        self._pending: list[tuple] = []

    def __iter__(self) -> "_E57BlockIterator":
        return self

    def __next__(self) -> tuple:
        if self._closed:
            raise StopIteration
        if self._image_file is None:
            try:
                self._open()
            except BaseException:
                # A failed open must also be terminal: never retry the file.
                self.close()
                raise

        try:
            block = self._next_block()
        except StopIteration:
            self.close()
            raise
        except (ValueError, OSError):
            self.close()
            raise
        except BaseException:
            # The RuntimeError meaning "pye57 not installed" originates in
            # _open; every other library failure is data-related and has
            # already been normalized to ValueError by the helpers above.
            self.close()
            raise
        return block

    def _open(self) -> None:
        if libe57 is None:
            raise RuntimeError("pye57 is required to read E57 files")

        # Probe OS-level openability first so missing files, permissions and
        # non-regular paths surface as OSError instead of a library error.
        probe = _open(self._path, "rb")
        probe.close()

        try:
            image_file = libe57.ImageFile(self._path, "r")
        except libe57.E57Exception as exc:
            raise ValueError(f"E57 container could not be opened: {exc}") from exc
        except RuntimeError as exc:
            raise ValueError(f"E57 container could not be decoded: {exc}") from exc

        try:
            root = image_file.root()
            data3d = root["data3D"]
            scan_count = len(data3d)
        except (libe57.E57Exception, TypeError, AttributeError, RuntimeError) as exc:
            image_file.close()
            raise ValueError(f"E57 container is invalid: {exc}") from exc

        self._image_file = image_file
        self._data3d = data3d
        self._scan_count = scan_count

    def _start_scan(self, index: int) -> None:
        """Validate scan *index* and open a streaming reader for its records."""
        try:
            scan = self._data3d[index]
            points = scan["points"]
            prototype = libe57.StructureNode(points.prototype())
            field_names = tuple(
                prototype.get(field_index).elementName()
                for field_index in range(prototype.childCount()))
            point_count = points.childCount()
        except (libe57.E57Exception, TypeError, IndexError,
                AttributeError, RuntimeError) as exc:
            raise ValueError(f"E57 scan {index} could not be read: {exc}") from exc

        if not all(name in field_names for name in _REQUIRED_FIELDS):
            raise ValueError(
                f"E57 scan {index} must provide cartesianX, cartesianY and "
                f"cartesianZ point fields (got: {', '.join(field_names)})")

        # Zero-record scans carry no data blocks and are simply skipped.
        if point_count == 0:
            return

        fields = list(_REQUIRED_FIELDS)
        if _INTENSITY_FIELD in field_names:
            fields.append(_INTENSITY_FIELD)

        buffers = libe57.VectorSourceDestBuffer()
        arrays: dict = {}
        for name in fields:
            # Conversion/scaling in the library turns integer and scaled
            # integer nodes into plain float64 values during decoding.
            array = np.empty(self._chunk_size, dtype="float64")
            arrays[name] = array
            buffers.append(libe57.SourceDestBuffer(
                self._image_file, name, array, self._chunk_size, True, True))

        try:
            reader = points.reader(buffers)
        except (libe57.E57Exception, RuntimeError) as exc:
            raise ValueError(f"E57 scan {index} records could not be decoded: "
                             f"{exc}") from exc

        self._reader = reader
        self._point_count = point_count
        self._read_count = 0
        self._fields = tuple(fields)
        self._buffers = arrays

    def _append_records(self, count: int) -> None:
        xs = self._buffers["cartesianX"]
        ys = self._buffers["cartesianY"]
        zs = self._buffers["cartesianZ"]
        intensities = self._buffers.get(_INTENSITY_FIELD)

        rows: list[tuple] = []
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            for index in range(count):
                x = _quantize_coordinate(xs[index].item())
                y = _quantize_coordinate(ys[index].item())
                z = _quantize_coordinate(zs[index].item())
                intensity = (_as_intensity(intensities[index].item())
                             if intensities is not None else 0)
                rows.append((x, y, z, intensity, 1.0))
        self._pending.extend(rows)

    def _flush(self) -> tuple:
        block = tuple(self._pending[:self._chunk_size])
        del self._pending[:self._chunk_size]
        return block

    def _next_block(self) -> tuple:
        while len(self._pending) < self._chunk_size:
            if self._reader is None:
                if self._scan_index >= self._scan_count:
                    if self._pending:
                        return self._flush()
                    break
                self._start_scan(self._scan_index)
                self._scan_index += 1
                continue

            try:
                count = self._reader.read()
            except (libe57.E57Exception, RuntimeError) as exc:
                raise ValueError(
                    f"E57 scan {self._scan_index - 1} records could not be "
                    f"decoded: {exc}") from exc

            if count == 0:
                if self._read_count != self._point_count:
                    raise ValueError(
                        f"E57 scan {self._scan_index - 1} point records are "
                        "truncated")
                self._close_reader()
                continue

            self._read_count += count
            self._append_records(count)

        if self._pending:
            return self._flush()
        self.close()
        raise StopIteration

    def _close_reader(self) -> None:
        reader = self._reader
        self._reader = None
        self._buffers = {}
        if reader is not None:
            reader.close()

    def close(self) -> None:
        self._closed = True
        reader = self._reader
        image_file = self._image_file
        self._reader = None
        self._image_file = None
        if reader is not None:
            try:
                reader.close()
            except Exception:  # pragma: no cover - never mask close failures
                pass
        if image_file is not None:
            image_file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise out of __del__
            pass


def iter_e57(path, chunk_size=65536):
    """Stream points from an E57 file as fixed-size blocks of tuples.

    Returns a one-shot iterator that yields blocks concatenated across scans in
    E57 scan index order and, within each scan, in point record order. Every
    block except the last contains exactly ``chunk_size`` points; the last block
    holds the remaining points and is omitted entirely when it would be empty, so
    a file with no scans or only zero-point scans yields no blocks. The file is
    read incrementally (never loaded as a whole) and closed when the iterator is
    exhausted, fails or is collected by garbage collection; iterating it again
    does not reopen the file.

    Each scan must provide ``cartesianX``, ``cartesianY`` and ``cartesianZ``
    point fields; scans that only carry spherical coordinates are rejected. An
    optional ``intensity`` field defaults to ``0`` when absent. Every point is a
    ``(x, y, z, intensity, sigma)`` tuple where ``sigma`` is fixed at ``1.0``.

    Coordinates are converted via ``Decimal(str(value))`` (precision 50,
    ROUND_HALF_EVEN), quantized to six decimal places before conversion to
    float; negative zero is normalized to positive zero. Intensity values must be
    finite integers and coordinates must be finite.

    :param path: filesystem path to a ``.e57`` file (str only); the extension
        check is case-insensitive
    :param chunk_size: points per block, a positive non-bool int
    :raises TypeError: ``path`` is not a str, or ``chunk_size`` is not a
        non-bool int
    :raises ValueError: unsupported extension; the E57 container, a scan or a
        point field cannot be read; cartesian coordinates are missing; records
        are truncated or cannot be decoded; coordinates are non-finite; or
        intensity is not a finite integer
    :raises OSError: the file cannot be opened
    :raises RuntimeError: the pye57 dependency is not installed
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if os.path.splitext(path)[1].lower() != _SUPPORTED_EXTENSION:
        raise ValueError("path must have a .e57 extension")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a non-bool int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return _E57BlockIterator(path, chunk_size)
