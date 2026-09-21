"""Streaming reader for LAS/LAZ LiDAR point cloud files.

Only :func:`iter_las` is public; the rest is an implementation detail of the
iterator it returns.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import laspy
import numpy as np
from laspy import LazBackend
from laspy.errors import LaspyException
from laspy.point.dims import DimensionKind

# laspy surfaces lazrs' own errors directly instead of wrapping them.
try:
    import lazrs
except ModuleNotFoundError:  # pragma: no cover - declared via laspy[lazrs]
    lazrs = None

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_POINTS_PER_CHUNK = 65536
_SUPPORTED_EXTENSIONS = (".las", ".laz")
_STANDARD_FIELDS = ("X", "Y", "Z", "intensity")
_NUMERIC_KINDS = (
    DimensionKind.SignedInteger,
    DimensionKind.UnsignedInteger,
    DimensionKind.FloatingPoint,
)

# Errors raised while parsing headers or decoding point records.
_PARSE_ERRORS: tuple[type[BaseException], ...] = (LaspyException,)
if lazrs is not None:
    _PARSE_ERRORS = (LaspyException, lazrs.LazrsError)

# LazrsError is itself a RuntimeError, so a plain RuntimeError from the decode
# path is also treated as unparseable data rather than a missing decoder.
# ArithmeticError covers Decimal failures (overflow, quantization beyond the
# 50-digit precision) caused by implausible but technically finite values.
_DECODE_ERRORS = _PARSE_ERRORS + (RuntimeError, ArithmeticError)


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    try:
        result = float(value.quantize(_QUANTUM))
    except OverflowError:
        return float("inf")
    return 0.0 if result == 0.0 else result


def _has_numeric_sigma(reader: laspy.LasReader) -> bool:
    """Whether a scalar numeric extra dimension named ``sigma`` exists."""
    point_format = reader.header.point_format
    if "sigma" not in point_format.dimension_names:
        return False
    info = point_format["sigma"]
    if info.is_standard or info.num_elements != 1:
        return False
    return info.kind in _NUMERIC_KINDS


def _is_missing_decoder(exc: BaseException) -> bool:
    message = str(exc)
    return "not available" in message or "No LazBackend selected" in message


class _LasBlockIterator:
    """One-shot iterator over fixed-size point tuples read from a LAS/LAZ file."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._reader: laspy.LasReader | None = None
        self._chunk_iterator = None
        self._closed = False
        self._scales: tuple[Decimal, Decimal, Decimal] = ()
        self._offsets: tuple[Decimal, Decimal, Decimal] = ()
        self._sigma_present = False

    def __iter__(self) -> "_LasBlockIterator":
        return self

    def __next__(self) -> tuple:
        if self._closed:
            raise StopIteration
        if self._reader is None:
            try:
                self._open()
            except BaseException:
                # A failed open must also be terminal: never retry the file.
                self.close()
                raise

        try:
            chunk = next(self._chunk_iterator)
            block = self._build_block(chunk)
        except StopIteration:
            self.close()
            raise
        except _DECODE_ERRORS as exc:
            self.close()
            raise ValueError(f"LAS/LAZ record could not be decoded: {exc}") from exc
        except OSError:
            self.close()
            raise
        except BaseException:
            # The handle must be released on any other failure too.
            self.close()
            raise
        return block

    def _open(self) -> None:
        try:
            reader = laspy.open(
                self._path,
                laz_backend=LazBackend.Lazrs,
                read_evlrs=False,
            )
        except LaspyException as exc:
            raise ValueError(f"LAS/LAZ header could not be parsed: {exc}") from exc
        # Plain OS failures (missing file, permission denied, ...) surface as
        # OSError subclasses and must propagate unchanged.

        try:
            # Build the point source eagerly so LAZ decoder problems are
            # reported from here rather than mid-iteration: this parses the
            # LasZip VLR/chunk table and constructs the lazrs decompressor.
            _ = reader.point_source
            self._configure(reader)
        except _DECODE_ERRORS as exc:
            reader.close()
            if _is_missing_decoder(exc):
                raise RuntimeError("LAZ decoder (lazrs) is not available") from exc
            raise ValueError(f"LAS/LAZ data could not be parsed: {exc}") from exc
        except BaseException:
            reader.close()
            raise

        self._reader = reader

    def _configure(self, reader: laspy.LasReader) -> None:
        header = reader.header
        point_format = header.point_format
        if not all(name in point_format.dimension_names for name in _STANDARD_FIELDS):
            raise ValueError("LAS point format is missing required standard fields "
                             "(X, Y, Z, intensity)")

        scales = tuple(float(value) for value in header.scales)
        offsets = tuple(float(value) for value in header.offsets)
        if (len(scales) != 3 or len(offsets) != 3
                or not all(map(math.isfinite, scales + offsets))):
            raise ValueError("LAS scale/offset values must be finite")

        self._sigma_present = _has_numeric_sigma(reader)
        self._scales = tuple(Decimal(str(value)) for value in scales)
        self._offsets = tuple(Decimal(str(value)) for value in offsets)
        self._chunk_iterator = reader.chunk_iterator(_POINTS_PER_CHUNK)

    def _build_block(self, chunk) -> tuple:
        raw = chunk.array
        xs = raw["X"]
        ys = raw["Y"]
        zs = raw["Z"]
        intensities = raw["intensity"]
        # Record-level access applies extra-bytes scale/offset decoding;
        # np.asarray is a zero-copy view for plain integer/float fields.
        sigmas = np.asarray(chunk["sigma"]) if self._sigma_present else None

        count = len(chunk)
        block: list = [None] * count

        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            sx, sy, sz = self._scales
            ox, oy, oz = self._offsets

            for index in range(count):
                x = _quantize(Decimal(str(xs[index].item())) * sx + ox)
                y = _quantize(Decimal(str(ys[index].item())) * sy + oy)
                z = _quantize(Decimal(str(zs[index].item())) * sz + oz)
                intensity = int(intensities[index])
                if not all(map(math.isfinite, (x, y, z))):
                    raise ValueError("decoded coordinates must be finite")

                if self._sigma_present:
                    sigma_value = sigmas[index].item()
                    if isinstance(sigma_value, float) and not math.isfinite(sigma_value):
                        raise ValueError("sigma must be finite and positive")
                    sigma_decimal = Decimal(str(sigma_value))
                    if sigma_decimal <= 0:
                        raise ValueError("sigma must be finite and positive")
                    sigma = _quantize(sigma_decimal)
                    if not math.isfinite(sigma):
                        raise ValueError("sigma must be finite and positive")
                else:
                    sigma = 1.0

                block[index] = (x, y, z, intensity, sigma)

        return tuple(block)

    def close(self) -> None:
        self._closed = True
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise out of __del__
            pass


def iter_las(path):
    """Stream points from a LAS/LAZ file as fixed-size blocks of tuples.

    Returns a one-shot iterator that yields blocks in file record order. Every
    block except the last contains exactly 65536 points; the last block holds
    the remaining points and is omitted entirely when it would be empty, so an
    empty file yields no blocks. The file is read incrementally (never loaded
    as a whole) and closed when the iterator is exhausted, fails or is
    collected by garbage collection; iterating it again does not reopen the
    file.

    Each point is a ``(x, y, z, intensity, sigma)`` tuple:

    * ``x``/``y``/``z`` are ``raw_integer * scale + offset`` computed with
      :class:`~decimal.Decimal` (precision 50, ROUND_HALF_EVEN);
    * ``intensity`` is the standard LAS intensity field converted to int;
    * ``sigma`` is the decoded value of a scalar numeric extra dimension named
      ``sigma``, or ``1.0`` when no such dimension exists.

    Coordinates and sigma are quantized to six decimal places before
    conversion to float; negative zero is normalized to positive zero.

    :param path: filesystem path to a ``.las`` or ``.laz`` file (str only)
    :raises TypeError: ``path`` is not a str
    :raises ValueError: unsupported extension; header or records cannot be
        parsed; required standard fields are missing; scale/offset or sigma
        values are non-finite; or sigma is non-positive
    :raises OSError: the file cannot be opened
    :raises RuntimeError: the LAZ decoder (lazrs) is not available
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if os.path.splitext(path)[1].lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError("path must have a .las or .laz extension")
    return _LasBlockIterator(path)
