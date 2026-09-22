"""Streaming reader for E57 LiDAR point cloud files.

Only :func:`iter_e57` is public; the rest is an implementation detail of the
iterator it returns.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

# Bound as a module attribute so the open call can be observed in tests.
_open = open

try:
    import pye57
    from pye57.libe57 import E57Exception
except ModuleNotFoundError:  # pragma: no cover - exercised via monkeypatching
    pye57 = None
    E57Exception = None

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_DEFAULT_CHUNK_SIZE = 65536
_SUPPORTED_EXTENSION = ".e57"
_REQUIRED_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")

# Errors raised by libe57 while parsing the container or decoding records.
_DECODE_ERRORS: tuple[type[BaseException], ...] = (ArithmeticError,)
if E57Exception is not None:
    _DECODE_ERRORS = (E57Exception, ArithmeticError)


def _quantize_coordinate(value) -> float:
    """Decimal(str(value)) quantized to six decimals (negative zero normalized)."""
    if not math.isfinite(value):
        raise ValueError("coordinates must be finite")
    try:
        result = float(Decimal(str(value)).quantize(_QUANTUM))
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError(f"coordinate could not be represented: {value!r}") from exc
    result = 0.0 if result == 0.0 else result
    if not math.isfinite(result):
        raise ValueError("coordinates must be finite")
    return result


def _as_intensity(value) -> int:
    """Convert a decoded intensity field to a plain integer."""
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("intensity must be a finite integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intensity must be a finite integer") from exc


class _E57BlockIterator:
    """One-shot iterator over fixed-size point tuples read from an E57 file."""

    def __init__(self, path: str, chunk_size: int) -> None:
        self._path = path
        self._chunk_size = chunk_size
        self._e57 = None
        self._closed = False
        self._scan_count = 0
        self._scan_index = 0
        self._reader = None
        self._arrays = None
        self._carry: list = []

    def __iter__(self) -> "_E57BlockIterator":
        return self

    def __next__(self) -> tuple:
        if self._closed:
            raise StopIteration
        if self._e57 is None:
            try:
                self._open()
            except BaseException:
                # A failed open must also be terminal: never retry the file.
                self.close()
                raise

        try:
            return self._next_block()
        except StopIteration:
            self.close()
            raise
        except ValueError:
            self.close()
            raise
        except _DECODE_ERRORS as exc:
            self.close()
            raise ValueError(f"E57 data could not be decoded: {exc}") from exc
        except OSError:
            self.close()
            raise
        except BaseException:
            # The handle must be released on any other failure too.
            self.close()
            raise

    def _open(self) -> None:
        if pye57 is None:
            raise RuntimeError("pye57 is not available")
        # Probe the file so plain OS failures (missing file, permission
        # denied, ...) surface as OSError subclasses unchanged; libe57
        # reports them as a generic E57Exception instead.
        with _open(self._path, "rb"):
            pass
        try:
            e57 = pye57.E57(self._path)
        except E57Exception as exc:
            raise ValueError(f"E57 container could not be parsed: {exc}") from exc

        try:
            self._scan_count = e57.scan_count
        except _DECODE_ERRORS as exc:
            e57.close()
            raise ValueError(f"E57 container could not be parsed: {exc}") from exc
        except BaseException:
            e57.close()
            raise

        self._e57 = e57

    def _next_block(self) -> tuple:
        while len(self._carry) < self._chunk_size:
            if self._reader is None and not self._advance_scan():
                break  # every scan has been consumed
            if not self._fill_from_current_scan():
                self._reader = None  # scan exhausted; advance on the next lap
        if not self._carry:
            self.close()
            raise StopIteration
        if len(self._carry) < self._chunk_size:
            # The final block holds the remaining points; the file is fully
            # consumed, so release the handle right away.
            block = tuple(self._carry)
            self._carry.clear()
            self.close()
            return block
        block = tuple(self._carry[:self._chunk_size])
        del self._carry[:self._chunk_size]
        return block

    def _advance_scan(self) -> bool:
        """Open the next scan for reading; False when no scan remains."""
        if self._scan_index >= self._scan_count:
            return False
        header = self._e57.get_header(self._scan_index)
        self._scan_index += 1
        fields = header.point_fields
        if not all(name in fields for name in _REQUIRED_FIELDS):
            raise ValueError("E57 scan is missing required cartesianX/"
                             "cartesianY/cartesianZ fields")
        wanted = list(_REQUIRED_FIELDS)
        if "intensity" in fields:
            wanted.append("intensity")
        self._arrays, buffers = self._e57.make_buffers(wanted, self._chunk_size)
        self._reader = header.points.reader(buffers)
        return True

    def _fill_from_current_scan(self) -> bool:
        """Decode one chunk of records into the carry buffer; False at EOF."""
        got = self._reader.read()
        if got == 0:
            self._reader.close()
            return False
        xs = self._arrays["cartesianX"]
        ys = self._arrays["cartesianY"]
        zs = self._arrays["cartesianZ"]
        intensities = self._arrays.get("intensity")

        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            for index in range(got):
                x = _quantize_coordinate(xs[index].item())
                y = _quantize_coordinate(ys[index].item())
                z = _quantize_coordinate(zs[index].item())
                intensity = (_as_intensity(intensities[index].item())
                             if intensities is not None else 0)
                self._carry.append((x, y, z, intensity, 1.0))
        return True

    def close(self) -> None:
        self._closed = True
        e57 = self._e57
        self._e57 = None
        self._reader = None
        self._arrays = None
        if e57 is not None:
            e57.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise out of __del__
            pass


def iter_e57(path, chunk_size=_DEFAULT_CHUNK_SIZE):
    """Stream points from an E57 file as fixed-size blocks of tuples.

    Returns a one-shot iterator that yields blocks in scan index and record
    order, concatenated across scans. Every block except the last contains
    exactly ``chunk_size`` points; the last block holds the remaining points
    and is omitted entirely when it would be empty, so a file with no scans
    (or only empty scans) yields no blocks. The file is read incrementally
    (never loaded as a whole) and closed when the iterator is exhausted,
    fails or is collected by garbage collection; iterating it again does not
    reopen the file.

    Every scan must provide ``cartesianX``, ``cartesianY`` and
    ``cartesianZ`` fields; scans with only spherical coordinates are
    rejected. ``intensity`` is optional and defaults to ``0``.

    Each point is a ``(x, y, z, intensity, sigma)`` tuple:

    * ``x``/``y``/``z`` are the finite cartesian coordinates;
    * ``intensity`` is a finite integer converted to int, or ``0``;
    * ``sigma`` is always ``1.0``.

    Coordinates are converted via ``Decimal(str(value))`` (precision 50,
    ROUND_HALF_EVEN), quantized to six decimal places before conversion to
    float; negative zero is normalized to positive zero.

    :param path: filesystem path to a ``.e57`` file (str only)
    :param chunk_size: points per block, a positive non-bool int
    :raises TypeError: ``path`` is not a str, or ``chunk_size`` is not a
        non-bool int
    :raises ValueError: unsupported extension; ``chunk_size <= 0``; the
        container, a scan or a record cannot be parsed; required cartesian
        fields are missing; coordinates are non-finite; or intensity is not
        a finite integer
    :raises OSError: the file cannot be opened
    :raises RuntimeError: pye57 is not available
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
