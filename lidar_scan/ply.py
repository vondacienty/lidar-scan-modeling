"""Streaming reader for PLY LiDAR point cloud files.

Only :func:`iter_ply` is public; the rest is an implementation detail of the
iterator it returns.
"""

from __future__ import annotations

import math
import os
import struct
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

# Bound as a module attribute so the open call can be observed in tests.
_open = open

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_DEFAULT_CHUNK_SIZE = 65536
_SUPPORTED_EXTENSION = ".ply"

# struct module codes use standard (non-native) sizes once prefixed with "<".
_SCALAR_TYPES: dict[str, str] = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
}
_INTEGER_TYPES = frozenset(("char", "uchar", "short", "ushort", "int", "uint"))
_ALLOWED_PROPERTIES = frozenset(("x", "y", "z", "intensity", "sigma"))
_SUPPORTED_FORMATS = frozenset(("ascii", "binary_little_endian"))


def _quantize_value(value, *, positive: bool = False) -> float:
    """Decimal(str(value)) quantized to six decimals (negative zero normalized)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("coordinates and sigma must be finite")
    try:
        decimal_value = Decimal(str(value))
    except ArithmeticError as exc:
        raise ValueError(f"value could not be decoded: {value!r}") from exc
    if positive and decimal_value <= 0:
        raise ValueError("sigma must be finite and positive")
    try:
        result = float(decimal_value.quantize(_QUANTUM))
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError(f"value could not be represented: {value!r}") from exc
    result = 0.0 if result == 0.0 else result
    if not math.isfinite(result):
        raise ValueError("coordinates and sigma must be finite")
    return result


def _as_intensity(value) -> int:
    """Convert a parsed intensity field to a plain integer."""
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("intensity must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intensity must be an integer") from exc


def _parse_header(stream):
    """Parse a PLY header from ``stream`` (positioned at the first byte).

    Returns ``(format_name, vertex_count, properties)`` where *properties* is
    a list of ``(name, type_name)`` pairs in declaration order. The stream is
    left positioned at the first byte of the data section.
    """
    magic = stream.readline()
    if magic.strip() != b"ply":
        raise ValueError("PLY file must start with a 'ply' magic line")

    format_name: str | None = None
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    seen: set[str] = set()

    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY header is missing an 'end_header' line")
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header must be ASCII encoded") from exc
        if not text:
            raise ValueError("empty line in PLY header")

        tokens = text.split()
        keyword = tokens[0]

        if keyword == "end_header":
            break
        if keyword in ("comment", "obj_info"):
            continue
        if keyword == "format":
            if len(tokens) != 3 or format_name is not None:
                raise ValueError("exactly one 'format ascii 1.0' or "
                                 "'format binary_little_endian 1.0' line is required")
            kind, version = tokens[1], tokens[2]
            if kind not in _SUPPORTED_FORMATS or version != "1.0":
                raise ValueError("only 'format ascii 1.0' and "
                                 "'format binary_little_endian 1.0' are supported")
            format_name = kind
        elif keyword == "element":
            if len(tokens) != 3:
                raise ValueError("malformed 'element' header line")
            name, count_text = tokens[1], tokens[2]
            if name != "vertex" or vertex_count is not None:
                raise ValueError("the header must contain exactly one element, "
                                 "'element vertex N'")
            try:
                count = int(count_text, 10)
            except ValueError as exc:
                raise ValueError("element vertex count must be a non-negative integer") from exc
            if count < 0:
                raise ValueError("element vertex count must be a non-negative integer")
            vertex_count = count
        elif keyword == "property":
            if vertex_count is None or len(tokens) < 3:
                raise ValueError("malformed 'property' header line")
            if tokens[1] == "list":
                raise ValueError("list properties are not supported")
            if len(tokens) != 3:
                raise ValueError("malformed scalar 'property' header line")
            type_name, prop_name = tokens[1], tokens[2]
            if type_name not in _SCALAR_TYPES:
                raise ValueError(f"unsupported property type: {type_name!r}")
            if prop_name not in _ALLOWED_PROPERTIES:
                raise ValueError(f"unsupported property: {prop_name!r}")
            if prop_name in seen:
                raise ValueError(f"duplicate property: {prop_name!r}")
            seen.add(prop_name)
            properties.append((prop_name, type_name))
        else:
            raise ValueError(f"unrecognized PLY header line: {keyword!r}")

    if format_name is None:
        raise ValueError("PLY header is missing a 'format' line")
    if vertex_count is None:
        raise ValueError("PLY header is missing an 'element vertex N' line")
    if not {"x", "y", "z"} <= seen:
        raise ValueError("element vertex must define x, y and z properties")

    return format_name, vertex_count, properties


class _PlyBlockIterator:
    """One-shot iterator over fixed-size point tuples read from a PLY file."""

    def __init__(self, path: str, chunk_size: int) -> None:
        self._path = path
        self._chunk_size = chunk_size
        self._file = None
        self._closed = False
        self._format: str | None = None
        self._vertex_count = 0
        self._remaining = 0
        self._properties: list[tuple[str, str]] = []
        self._struct: struct.Struct | None = None
        self._record_size = 0

    def __iter__(self) -> "_PlyBlockIterator":
        return self

    def __next__(self) -> tuple:
        if self._closed:
            raise StopIteration
        if self._file is None:
            try:
                self._open()
            except BaseException:
                # A failed open must also be terminal: never retry the file.
                self.close()
                raise

        try:
            if self._remaining == 0:
                self._check_no_more_data()
                self.close()
                raise StopIteration
            if self._format == "ascii":
                block = self._next_ascii_block()
            else:
                block = self._next_binary_block()
        except StopIteration:
            self.close()
            raise
        except (ValueError, OSError):
            self.close()
            raise
        except struct.error as exc:
            self.close()
            raise ValueError(f"PLY record could not be decoded: {exc}") from exc
        except BaseException:
            # The handle must be released on any other failure too.
            self.close()
            raise
        return block

    def _open(self) -> None:
        file = _open(self._path, "rb")
        try:
            format_name, vertex_count, properties = _parse_header(file)
        except BaseException:
            file.close()
            raise

        self._file = file
        self._format = format_name
        self._vertex_count = vertex_count
        self._remaining = vertex_count
        self._properties = properties
        if format_name == "binary_little_endian":
            layout = "<" + "".join(_SCALAR_TYPES[type_name]
                                   for _, type_name in properties)
            self._struct = struct.Struct(layout)
            self._record_size = self._struct.size

    def _check_no_more_data(self) -> None:
        """The declared vertex count must match the data section exactly."""
        if self._format == "ascii":
            # Tolerate trailing blank lines, but nothing that looks like data.
            while True:
                extra = self._file.readline()
                if not extra:
                    return
                if extra.strip():
                    break
        else:
            extra = self._file.read(1)
        if extra:
            raise ValueError("declared element vertex count does not match "
                             "the number of point records")

    def _build_block(self, rows) -> tuple:
        block: list = []
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            for values in rows:
                x = y = z = None
                intensity = 0
                sigma = 1.0
                for (name, _type_name), value in zip(self._properties, values):
                    if name == "x":
                        x = _quantize_value(value)
                    elif name == "y":
                        y = _quantize_value(value)
                    elif name == "z":
                        z = _quantize_value(value)
                    elif name == "intensity":
                        intensity = _as_intensity(value)
                    else:  # "sigma"
                        sigma = _quantize_value(value, positive=True)
                block.append((x, y, z, intensity, sigma))
        return tuple(block)

    def _parse_ascii_record(self, tokens) -> list:
        if len(tokens) != len(self._properties):
            raise ValueError("ASCII point record does not match the declared "
                             "number of properties")
        values = []
        for token, (_name, type_name) in zip(tokens, self._properties):
            try:
                if type_name in _INTEGER_TYPES:
                    value = int(token, 10)
                else:
                    value = float(token)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"could not parse {type_name} value "
                                 f"{token!r}") from exc
            values.append(value)
        return values

    def _next_ascii_block(self) -> tuple:
        want = min(self._chunk_size, self._remaining)
        rows: list = []
        while len(rows) < want:
            line = self._file.readline()
            if not line:
                raise ValueError("PLY data section is truncated")
            tokens = line.split()
            if not tokens:
                raise ValueError("unexpected blank line in PLY data section")
            rows.append(self._parse_ascii_record(tokens))

        block = self._build_block(rows)
        self._remaining -= want
        if self._remaining == 0:
            self._check_no_more_data()
            self.close()
        return block

    def _next_binary_block(self) -> tuple:
        want = min(self._chunk_size, self._remaining)
        raw = self._file.read(want * self._record_size)
        if len(raw) != want * self._record_size:
            raise ValueError("PLY data section is truncated")
        block = self._build_block(self._struct.iter_unpack(raw))
        self._remaining -= want
        if self._remaining == 0:
            self._check_no_more_data()
            self.close()
        return block

    def close(self) -> None:
        self._closed = True
        file = self._file
        self._file = None
        if file is not None:
            file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise out of __del__
            pass


def iter_ply(path, chunk_size=65536):
    """Stream points from a PLY file as fixed-size blocks of tuples.

    Returns a one-shot iterator that yields blocks in file record order. Every
    block except the last contains exactly ``chunk_size`` points; the last
    block holds the remaining points and is omitted entirely when it would be
    empty, so an empty file yields no blocks. The file is read incrementally
    (never loaded as a whole) and closed when the iterator is exhausted, fails
    or is collected by garbage collection; iterating it again does not reopen
    the file.

    The header must contain exactly one non-negative ``element vertex N``
    declaration (no other elements), ``format ascii 1.0`` or
    ``format binary_little_endian 1.0``, and scalar ``x``/``y``/``z``
    properties plus optional scalar ``intensity``/``sigma`` properties of
    types char, uchar, short, ushort, int, uint, float or double. List
    properties, unknown or duplicate properties and missing x/y/z are
    rejected.

    Each point is a ``(x, y, z, intensity, sigma)`` tuple in that fixed order
    regardless of header property order:

    * ``x``/``y``/``z`` are the coordinate fields (numeric);
    * ``intensity`` is converted to int (it must be integral when stored in a
      floating-point property) and defaults to ``0`` when absent;
    * ``sigma`` must be a finite positive number and defaults to ``1.0``.

    Coordinates and sigma are converted via ``Decimal(str(value))``
    (precision 50, ROUND_HALF_EVEN), quantized to six decimal places before
    conversion to float; negative zero is normalized to positive zero.

    :param path: filesystem path to a ``.ply`` file (str only)
    :param chunk_size: points per block, a positive non-bool int
    :raises TypeError: ``path`` is not a str, or ``chunk_size`` is not a
        non-bool int
    :raises ValueError: unsupported extension; ``chunk_size <= 0``; the header
        or data cannot be parsed; the declared point count does not match the
        records; coordinates are non-finite; intensity is non-integral; or
        sigma is non-finite or non-positive
    :raises OSError: the file cannot be opened
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if os.path.splitext(path)[1].lower() != _SUPPORTED_EXTENSION:
        raise ValueError("path must have a .ply extension")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a non-bool int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return _PlyBlockIterator(path, chunk_size)
