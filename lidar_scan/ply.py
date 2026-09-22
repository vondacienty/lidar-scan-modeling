"""Streaming reader for PLY LiDAR point cloud files.

Only :func:`iter_ply` is public; the rest is an implementation detail of the
iterator it returns.
"""

from __future__ import annotations

import math
import os
import re
import struct
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

_PRECISION = 50
_QUANTUM = Decimal("0.000001")
_POINTS_PER_CHUNK = 65536
_SUPPORTED_EXTENSIONS = (".ply",)

_FORMAT_ASCII = "ascii"
_FORMAT_BINARY = "binary_little_endian"
_FORMAT_VERSION = "1.0"

# PLY scalar property types: name -> (struct code, is_integer).
_PROPERTY_TYPES = {
    "char": ("b", True),
    "uchar": ("B", True),
    "short": ("h", True),
    "ushort": ("H", True),
    "int": ("i", True),
    "uint": ("I", True),
    "float": ("f", False),
    "double": ("d", False),
}
_INTEGER_TYPES = frozenset(
    name for name, (_, is_integer) in _PROPERTY_TYPES.items() if is_integer
)
_KNOWN_PROPERTIES = ("x", "y", "z", "intensity", "sigma")
_REQUIRED_PROPERTIES = ("x", "y", "z")

_INT_TOKEN = re.compile(r"[+-]?\d+")


def _quantize(value: Decimal) -> float:
    """Quantize to six decimal places and return a float (negative zero normalized)."""
    try:
        result = float(value.quantize(_QUANTUM))
    except OverflowError:
        return float("inf")
    return 0.0 if result == 0.0 else result


def _finite_decimal(value) -> Decimal:
    """Convert a parsed scalar to Decimal, rejecting non-finite floats."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("point values must be finite")
    return Decimal(str(value))


class _PlyBlockIterator:
    """One-shot iterator over fixed-size point tuples read from a PLY file."""

    def __init__(self, path: str, chunk_size: int) -> None:
        self._path = path
        self._chunk_size = chunk_size
        self._file = None
        self._closed = False
        self._format = None
        self._properties: list[tuple[str, str]] = []
        self._index: dict[str, int] = {}
        self._record: struct.Struct | None = None
        self._remaining = 0

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
                # The declared points are all read; trailing data means the
                # file does not match its own header.
                self._check_trailing()
                raise StopIteration
            block = self._read_block()
        except StopIteration:
            self.close()
            raise
        except OSError:
            self.close()
            raise
        except BaseException:
            # The handle must be released on any other failure too.
            self.close()
            raise
        return block

    def _open(self) -> None:
        # Plain OS failures (missing file, permission denied, ...) surface as
        # OSError subclasses and must propagate unchanged.
        handle = open(self._path, "rb")
        try:
            fmt, properties, count = _parse_header(handle)
        except BaseException:
            handle.close()
            raise

        self._file = handle
        self._format = fmt
        self._properties = properties
        self._index = {name: i for i, (name, _) in enumerate(properties)}
        self._remaining = count
        if fmt == _FORMAT_BINARY:
            fmt_codes = "".join(_PROPERTY_TYPES[t][0] for _, t in properties)
            self._record = struct.Struct("<" + fmt_codes)

    def _read_block(self) -> tuple:
        count = min(self._chunk_size, self._remaining)
        if self._format == _FORMAT_ASCII:
            rows = self._read_ascii_rows(count)
        else:
            rows = self._read_binary_rows(count)
        self._remaining -= count

        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            try:
                return tuple(self._make_point(row) for row in rows)
            except ArithmeticError as exc:
                raise ValueError(f"PLY point value could not be converted: {exc}") from exc

    def _read_ascii_rows(self, count: int) -> list:
        rows = []
        while len(rows) < count:
            line = self._file.readline()
            if line == b"":
                raise ValueError("PLY file is truncated: fewer points than declared")
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != len(self._properties):
                raise ValueError("PLY point does not match the declared properties")
            rows.append([
                self._parse_ascii_token(token, ptype)
                for token, (_, ptype) in zip(tokens, self._properties)
            ])
        return rows

    @staticmethod
    def _parse_ascii_token(token: bytes, ptype: str):
        text = token.decode("ascii")
        if ptype in _INTEGER_TYPES:
            if _INT_TOKEN.fullmatch(text) is None:
                raise ValueError(f"invalid integer value {text!r} in PLY data")
            return int(text)
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"invalid float value {text!r} in PLY data") from None

    def _read_binary_rows(self, count: int) -> list:
        record = self._record
        data = self._file.read(count * record.size)
        if len(data) != count * record.size:
            raise ValueError("PLY file is truncated: fewer points than declared")
        return [record.unpack_from(data, i * record.size) for i in range(count)]

    def _make_point(self, row) -> tuple:
        index = self._index
        x = _quantize(_finite_decimal(row[index["x"]]))
        y = _quantize(_finite_decimal(row[index["y"]]))
        z = _quantize(_finite_decimal(row[index["z"]]))
        if not all(map(math.isfinite, (x, y, z))):
            raise ValueError("decoded coordinates must be finite")

        intensity_index = index.get("intensity")
        intensity = int(row[intensity_index]) if intensity_index is not None else 0

        sigma_index = index.get("sigma")
        if sigma_index is not None:
            sigma_decimal = _finite_decimal(row[sigma_index])
            if sigma_decimal <= 0:
                raise ValueError("sigma must be finite and positive")
            sigma = _quantize(sigma_decimal)
            if not math.isfinite(sigma):
                raise ValueError("sigma must be finite and positive")
        else:
            sigma = 1.0

        return (x, y, z, intensity, sigma)

    def _check_trailing(self) -> None:
        if self._format == _FORMAT_ASCII:
            for line in self._file:
                if line.split():
                    raise ValueError("PLY file holds more points than declared")
        elif self._file.read(1) != b"":
            raise ValueError("PLY file holds more points than declared")

    def close(self) -> None:
        self._closed = True
        handle = self._file
        self._file = None
        if handle is not None:
            handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise out of __del__
            pass


def _parse_header(handle) -> tuple[str, list, int]:
    """Parse a PLY header, leaving ``handle`` positioned at the point data."""
    magic = handle.readline()
    if magic.rstrip(b"\r\n") != b"ply":
        raise ValueError("not a PLY file: missing 'ply' magic line")

    fmt = None
    count = None
    properties: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    while True:
        line = handle.readline()
        if line == b"":
            raise ValueError("PLY header is truncated: missing 'end_header'")
        tokens = line.split()
        if not tokens:
            continue
        keyword = tokens[0].decode("ascii")

        if keyword == "format":
            if fmt is not None or len(tokens) != 3:
                raise ValueError("malformed PLY format line")
            name = tokens[1].decode("ascii")
            version = tokens[2].decode("ascii")
            if name not in (_FORMAT_ASCII, _FORMAT_BINARY) or version != _FORMAT_VERSION:
                raise ValueError(f"unsupported PLY format: {name} {version}")
            fmt = name
        elif keyword == "comment" or keyword == "obj_info":
            continue
        elif keyword == "element":
            if count is not None or properties or len(tokens) != 3:
                raise ValueError("malformed PLY element line")
            name = tokens[1].decode("ascii")
            if name != "vertex":
                raise ValueError("only a single 'element vertex' is supported")
            try:
                text = tokens[2].decode("ascii")
            except UnicodeDecodeError:
                raise ValueError("malformed PLY element count") from None
            if _INT_TOKEN.fullmatch(text) is None:
                raise ValueError("malformed PLY element count")
            count = int(text)
            if count < 0:
                raise ValueError("PLY element count must be non-negative")
        elif keyword == "property":
            if count is None:
                raise ValueError("PLY property declared before any element")
            if len(tokens) != 3:
                # This also rejects vector ('property list ...') declarations.
                raise ValueError("only scalar PLY properties are supported")
            ptype = tokens[1].decode("ascii")
            name = tokens[2].decode("ascii")
            if ptype not in _PROPERTY_TYPES:
                raise ValueError(f"unsupported PLY property type: {ptype}")
            if name not in _KNOWN_PROPERTIES:
                raise ValueError(f"unsupported PLY property: {name}")
            if name in seen_names:
                raise ValueError(f"duplicate PLY property: {name}")
            if name == "intensity" and ptype not in _INTEGER_TYPES:
                raise ValueError("PLY intensity property must have an integer type")
            seen_names.add(name)
            properties.append((name, ptype))
        elif keyword == "end_header":
            if len(tokens) != 1:
                raise ValueError("malformed PLY end_header line")
            break
        else:
            raise ValueError(f"unsupported PLY header line: {keyword}")

    if fmt is None:
        raise ValueError("PLY header is missing the format line")
    if count is None:
        raise ValueError("PLY header is missing 'element vertex'")
    missing = [name for name in _REQUIRED_PROPERTIES if name not in seen_names]
    if missing:
        raise ValueError("PLY vertex properties are missing: " + ", ".join(missing))

    return fmt, properties, count


def iter_ply(path, chunk_size=_POINTS_PER_CHUNK):
    """Stream points from a PLY file as fixed-size blocks of tuples.

    Returns a one-shot iterator that yields blocks in file record order. Every
    block except the last contains exactly ``chunk_size`` points; the last
    block holds the remaining points and is omitted entirely when it would be
    empty, so a file declaring zero vertices yields no blocks. The file is read
    incrementally (never loaded as a whole) and closed when the iterator is
    exhausted, fails or is collected by garbage collection; iterating it again
    does not reopen the file.

    Only ``format ascii 1.0`` and ``format binary_little_endian 1.0`` files
    with a single ``element vertex`` are supported. Vertex properties must be
    scalar ``x``/``y``/``z`` plus optional ``intensity`` (integer type) and
    ``sigma``; allowed types are char, uchar, short, ushort, int, uint, float
    and double.

    Each point is a ``(x, y, z, intensity, sigma)`` tuple, assembled in that
    order regardless of the property order in the file:

    * ``x``/``y``/``z`` are the parsed property values converted with
      :class:`~decimal.Decimal` (precision 50, ROUND_HALF_EVEN);
    * ``intensity`` is the property value converted to int, or ``0`` when the
      property is absent;
    * ``sigma`` is the parsed property value, or ``1.0`` when absent.

    Coordinates and sigma are quantized to six decimal places before
    conversion to float; negative zero is normalized to positive zero.

    :param path: filesystem path to a ``.ply`` file (str only)
    :param chunk_size: points per block (non-bool int, must be positive)
    :raises TypeError: ``path`` is not a str, or ``chunk_size`` is not a
        non-bool int
    :raises ValueError: unsupported extension; non-positive ``chunk_size``;
        unsupported or malformed header; list, unknown, duplicate or missing
        required properties; point count mismatch; truncated, non-finite or
        unparseable data; or non-positive sigma
    :raises OSError: the file cannot be opened
    """
    if not isinstance(path, str):
        raise TypeError("path must be a str")
    if os.path.splitext(path)[1].lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError("path must have a .ply extension")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a non-bool int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _PlyBlockIterator(path, chunk_size)
