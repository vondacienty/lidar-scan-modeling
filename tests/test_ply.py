"""Tests for :func:`lidar_scan.iter_ply`."""

from __future__ import annotations

import gc
import math
import os
import struct

import pytest

from lidar_scan import iter_ply
import lidar_scan.ply as ply_module

CHUNK = 65536


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _ascii_header(properties, n, *, fmt="ascii"):
    lines = ["ply", f"format {fmt} 1.0", "comment generated for tests",
             f"element vertex {n}"]
    lines.extend(f"property {type_name} {name}" for name, type_name in properties)
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _write_ascii_ply(path, points, properties, *, n=None, fmt="ascii"):
    if n is None:
        n = len(points)
    content = _ascii_header(properties, n, fmt=fmt)
    for point in points:
        content += " ".join(str(value) for value in point) + "\n"
    path.write_text(content)
    return path


def _write_binary_ply(path, records, properties, *, n=None, trailing=b""):
    if n is None:
        n = len(records)
    header = _ascii_header(properties, n, fmt="binary_little_endian")
    layout = "<" + "".join(ply_module._SCALAR_TYPES[type_name]
                           for _name, type_name in properties)
    body = b"".join(struct.pack(layout, *record) for record in records)
    path.write_bytes(header.encode("ascii") + body + trailing)
    return path


_XYZ = [("x", "float"), ("y", "float"), ("z", "float")]
_XYZS = [("x", "double"), ("y", "double"), ("z", "double"),
         ("intensity", "uint"), ("sigma", "double")]


# ---------------------------------------------------------------------------
# Parameter validation (errors raised at call time, before any file access)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [123, 1.5, None, b"/tmp/x.ply", ["/tmp/x.ply"],
                                 ("/tmp/x.ply",), object()])
def test_path_must_be_str(bad):
    with pytest.raises(TypeError):
        iter_ply(bad)


@pytest.mark.parametrize("name", ["x.txt", "x.plyx", "x.p", "xply", "x",
                                  "x.ply.txt", ".plyless"])
def test_bad_extension(name, tmp_path):
    with pytest.raises(ValueError):
        iter_ply(str(tmp_path / name))


@pytest.mark.parametrize("ext", [".ply", ".PLY", ".Ply", ".pLy"])
def test_extension_case_insensitive(tmp_path, ext):
    target = tmp_path / f"scan{ext}"
    _write_ascii_ply(target, [(1.0, 2.0, 3.0)], _XYZ)
    blocks = list(iter_ply(str(target)))
    assert len(blocks) == 1
    assert blocks[0][0] == (1.0, 2.0, 3.0, 0, 1.0)


@pytest.mark.parametrize("bad", [True, False, 1.0, 0.0, "4096", None, 1 + 0j,
                                 2**64 * 1.0])
def test_chunk_size_must_be_non_bool_int(tmp_path, bad):
    path = _write_ascii_ply(tmp_path / "pts.ply", [(1.0, 2.0, 3.0)], _XYZ)
    with pytest.raises(TypeError):
        iter_ply(str(path), chunk_size=bad)


@pytest.mark.parametrize("bad", [0, -1, -65536])
def test_chunk_size_must_be_positive(tmp_path, bad):
    path = _write_ascii_ply(tmp_path / "pts.ply", [(1.0, 2.0, 3.0)], _XYZ)
    with pytest.raises(ValueError):
        iter_ply(str(path), chunk_size=bad)


# ---------------------------------------------------------------------------
# Blocking / streaming shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["ascii", "binary_little_endian"])
def test_empty_file_yields_nothing(tmp_path, fmt):
    if fmt == "ascii":
        path = _write_ascii_ply(tmp_path / "empty.ply", [], _XYZ)
    else:
        path = _write_binary_ply(tmp_path / "empty.ply", [], _XYZ)
    assert list(iter_ply(str(path))) == []


@pytest.mark.parametrize("n", [1, 2, CHUNK - 1, CHUNK, CHUNK + 1,
                               CHUNK * 2 + 3])
def test_default_block_sizes_ascii(tmp_path, n):
    points = [(float(i), float(i + 1), float(i - 1)) for i in range(n)]
    path = _write_ascii_ply(tmp_path / "pts.ply", points, _XYZ)
    sizes = [len(block) for block in iter_ply(str(path))]
    full, remainder = divmod(n, CHUNK)
    assert sizes == [CHUNK] * full + ([remainder] if remainder else [])
    assert sum(sizes) == n


@pytest.mark.parametrize("chunk_size", [1, 3, 1024, 65536])
def test_custom_chunk_size_binary(tmp_path, chunk_size):
    records = [(float(i), 0.0, -1.0) for i in range(chunk_size * 2 + 1)]
    path = _write_binary_ply(tmp_path / "pts.ply", records, _XYZ)
    blocks = list(iter_ply(str(path), chunk_size=chunk_size))
    assert [len(b) for b in blocks] == [chunk_size, chunk_size, 1]


def test_iterator_is_one_shot_and_self_iterating(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)], _XYZ)
    result = iter_ply(str(path), chunk_size=2)
    assert iter(result) is result
    first = list(result)
    assert sum(len(b) for b in first) == 3
    # exhausted iterator must not reopen the file
    assert list(result) == []
    with pytest.raises(StopIteration):
        next(result)


def test_lazy_opening(tmp_path):
    path = _write_ascii_ply(tmp_path / "pts.ply", [(1.0, 2.0, 3.0)], _XYZ)
    result = iter_ply(str(path))
    assert result._file is None  # file not opened until first next()
    next(result)
    assert result._closed is True


def test_points_in_file_record_order(tmp_path):
    records = [(float(i), float(i), 0.0, i * 2, 1.0)
               for i in range(10)]
    path = _write_binary_ply(tmp_path / "pts.ply", records, _XYZS)
    points = list(iter_ply(str(path)))[0]
    assert [p[3] for p in points] == [i * 2 for i in range(10)]
    assert points[7] == (7.0, 7.0, 0.0, 14, 1.0)


def test_does_not_read_whole_file(tmp_path, monkeypatch):
    records = [(float(i), 0.0, 0.0) for i in range(10)]
    path = _write_binary_ply(tmp_path / "pts.ply", records, _XYZ)
    record_size = sum(
        struct.calcsize("<" + ply_module._SCALAR_TYPES[t]) for _n, t in _XYZ)
    max_read = 4 * record_size
    real_open = ply_module._open

    class TrackingFile:
        def __init__(self, inner):
            self._inner = inner

        def read(self, size=-1):
            assert isinstance(size, int) and 0 < size <= max_read
            return self._inner.read(size)

        def readline(self):
            return self._inner.readline()

        def close(self):
            return self._inner.close()

    def tracking_open(name, mode="r", *args, **kwargs):
        return TrackingFile(real_open(name, mode, *args, **kwargs))

    monkeypatch.setattr(ply_module, "_open", tracking_open)
    blocks = list(iter_ply(str(path), chunk_size=4))
    assert sum(len(b) for b in blocks) == 10


# ---------------------------------------------------------------------------
# Point decoding
# ---------------------------------------------------------------------------

def test_default_intensity_and_sigma_ascii(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [(1.5, -2.25, 3.0), (0.0, 0.0, 0.0)], _XYZ)
    points = list(iter_ply(str(path)))[0]
    assert points == ((1.5, -2.25, 3.0, 0, 1.0), (0.0, 0.0, 0.0, 0, 1.0))
    point = points[0]
    assert isinstance(point, tuple) and len(point) == 5
    assert all(isinstance(v, float) for v in (point[0], point[1], point[2],
                                              point[4]))
    assert isinstance(point[3], int) and not isinstance(point[3], bool)


def test_properties_parsed_in_declaration_order_binary(tmp_path):
    # intensity first, coordinates interleaved, sigma in the middle.
    properties = [("intensity", "ushort"), ("y", "double"), ("x", "float"),
                  ("sigma", "float"), ("z", "int")]
    layout = "<" + "".join(ply_module._SCALAR_TYPES[t]
                           for _n, t in properties)
    header = _ascii_header(properties, 2, fmt="binary_little_endian")
    records = [
        struct.pack(layout, 42, 2.5, 1.25, 0.5, 7),
        struct.pack(layout, 7, -3.0, 0.0, 2.0, -9),
    ]
    path = tmp_path / "ordered.ply"
    path.write_bytes(header.encode("ascii") + b"".join(records))
    points = list(iter_ply(str(path)))[0]
    assert points == ((1.25, 2.5, 7.0, 42, 0.5),
                      (0.0, -3.0, -9.0, 7, 2.0))


@pytest.mark.parametrize("type_name,code,value", [
    ("char", "b", -12),
    ("uchar", "B", 200),
    ("short", "h", -300),
    ("ushort", "H", 60000),
    ("int", "i", -123456),
    ("uint", "I", 4000000000),
    ("float", "f", 1.5),
    ("double", "d", 1.5),
])
def test_all_scalar_types_supported(tmp_path, type_name, code, value):
    properties = [("x", type_name), ("y", type_name), ("z", type_name)]
    header = _ascii_header(properties, 1, fmt="binary_little_endian")
    body = struct.pack(f"<{code * 3}", value, value, value)
    path = tmp_path / "typed.ply"
    path.write_bytes(header.encode("ascii") + body)
    point = list(iter_ply(str(path)))[0][0]
    assert point[:3] == (float(value), float(value), float(value))


def test_integer_sigma_becomes_float(tmp_path):
    properties = [("x", "int"), ("y", "int"), ("z", "int"),
                  ("sigma", "ushort")]
    path = _write_binary_ply(tmp_path / "pts.ply",
                             [(1, 2, 3, 7), (0, 0, 0, 1)], properties)
    points = list(iter_ply(str(path)))[0]
    assert [p[4] for p in points] == [7.0, 1.0]
    assert all(isinstance(p[4], float) for p in points)


def test_intensity_float_property_must_be_integral(tmp_path):
    properties = [("x", "float"), ("y", "float"), ("z", "float"),
                  ("intensity", "float")]
    path = _write_binary_ply(tmp_path / "pts.ply",
                             [(0.0, 0.0, 0.0, 2.5)], properties)
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_intensity_float_property_integral_value_ok(tmp_path):
    properties = [("x", "float"), ("y", "float"), ("z", "float"),
                  ("intensity", "double")]
    path = _write_binary_ply(tmp_path / "pts.ply",
                             [(0.0, 0.0, 0.0, 12345.0)], properties)
    point = list(iter_ply(str(path)))[0][0]
    assert point[3] == 12345
    assert isinstance(point[3], int)


def test_ascii_integer_and_float_tokens(tmp_path):
    properties = [("x", "int"), ("y", "double"), ("z", "float"),
                  ("intensity", "uint")]
    path = _write_ascii_ply(tmp_path / "pts.ply",
                            [("-7", "2.5e1", "1.25", "99")], properties)
    point = list(iter_ply(str(path)))[0][0]
    assert point == (-7.0, 25.0, 1.25, 99, 1.0)


def test_quantized_to_six_decimals_and_negative_zero(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [("0.333333333333", "0.1234567", "-0.0000004")], _XYZ)
    point = list(iter_ply(str(path)))[0][0]
    assert point[0] == round(1 / 3, 6)
    assert point[1] == 0.123457
    assert point[2] == 0.0
    for index in (0, 1, 2):
        assert math.copysign(1.0, point[index]) == 1.0


def test_round_half_even_tie_breaking(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [("0.1234565", "0.1234575", "0.1234555")], _XYZ)
    x, y, z, _i, _s = list(iter_ply(str(path)))[0][0]
    from decimal import Decimal, ROUND_HALF_EVEN
    def expected(token):
        return float(Decimal(token).quantize(Decimal("0.000001"),
                                            rounding=ROUND_HALF_EVEN))
    assert x == expected("0.1234565")
    assert y == expected("0.1234575")
    assert z == expected("0.1234555")


def test_crlf_header_and_data(tmp_path):
    content = ("ply\r\nformat ascii 1.0\r\nelement vertex 1\r\n"
               "property float x\r\nproperty float y\r\nproperty float z\r\n"
               "end_header\r\n1 2 3\r\n")
    path = tmp_path / "crlf.ply"
    path.write_bytes(content.encode("ascii"))
    assert list(iter_ply(str(path)))[0][0] == (1.0, 2.0, 3.0, 0, 1.0)


def test_trailing_blank_lines_ascii_ignored(tmp_path):
    path = _write_ascii_ply(tmp_path / "pts.ply",
                            [(1.0, 2.0, 3.0)], _XYZ)
    with open(path, "a") as handle:
        handle.write("\n  \n")
    points = list(iter_ply(str(path)))[0]
    assert points == ((1.0, 2.0, 3.0, 0, 1.0),)


def test_extra_byte_binary_is_count_mismatch(tmp_path):
    path = _write_binary_ply(tmp_path / "pts.ply",
                             [(1.0, 2.0, 3.0)], _XYZ, trailing=b"\x00")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_extra_record_ascii_is_count_mismatch(tmp_path):
    path = _write_ascii_ply(tmp_path / "pts.ply",
                            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], _XYZ, n=1)
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def test_bad_magic(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("not ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("format_line", [
    "format ascii 1.2",
    "format binary_big_endian 1.0",
    "format binary_little_endian 1.2",
    "format ascii 2.0",
])
def test_unsupported_format(tmp_path, format_line):
    path = tmp_path / "bad.ply"
    path.write_text(f"ply\n{format_line}\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_missing_format_line(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_two_format_lines(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nformat ascii 1.0\n"
                    "element vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_second_element_rejected_even_when_empty(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\n"
                    "element face 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_two_vertex_elements_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\n"
                    "element vertex 2\nproperty float x\nproperty float y\n"
                    "property float z\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_missing_vertex_element(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("count_line", ["element vertex -1",
                                        "element vertex 1.5",
                                        "element vertex abc"])
def test_bad_vertex_count(tmp_path, count_line):
    path = tmp_path / "bad.ply"
    path.write_text(f"ply\nformat ascii 1.0\n{count_line}\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_zero_vertex_count_without_xyz_still_requires_xyz(tmp_path):
    # Missing x/y/z is a header error even when the vertex count is zero.
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_list_property_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property list uchar int indices\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_unknown_property_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_duplicate_property_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property float x\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("missing", ["x", "y", "z"])
def test_missing_coordinate_rejected(tmp_path, missing):
    names = [n for n in ("x", "y", "z") if n != missing]
    props = "".join(f"property float {n}\n" for n in names)
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    + props + "end_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_unknown_property_type_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property long x\nproperty float y\nproperty float z\n"
                    "end_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_missing_end_header(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_unknown_header_keyword_rejected(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\n"
                    "mystery line\nend_header\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

def test_truncated_ascii(tmp_path):
    path = _write_ascii_ply(tmp_path / "pts.ply",
                            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], _XYZ, n=3)
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_truncated_binary(tmp_path):
    header = _ascii_header(_XYZ, 2, fmt="binary_little_endian")
    one_record = struct.pack("<fff", 1.0, 2.0, 3.0)
    path = tmp_path / "pts.ply"
    path.write_bytes(header.encode("ascii") + one_record)
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_ascii_wrong_field_count(tmp_path):
    path = tmp_path / "pts.ply"
    path.write_text(_ascii_header(_XYZ, 1) + "1 2\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_ascii_unparseable_value(tmp_path):
    path = tmp_path / "pts.ply"
    path.write_text(_ascii_header(_XYZ, 1) + "1 oops 3\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_ascii_blank_data_line(tmp_path):
    path = tmp_path / "pts.ply"
    path.write_text(_ascii_header(_XYZ, 1) + "\n1 2 3\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "infinity"])
def test_nonfinite_coordinates_ascii(tmp_path, token):
    path = tmp_path / "pts.ply"
    path.write_text(_ascii_header(_XYZ, 1) + f"1 2 {token}\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_nonfinite_coordinates_binary(tmp_path):
    properties = [("x", "float"), ("y", "float"), ("z", "float")]
    path = _write_binary_ply(tmp_path / "pts.ply",
                             [(float("nan"), 0.0, 0.0)], properties)
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("bad_sigma", [0.0, -0.5, -100.0])
def test_non_positive_sigma(tmp_path, bad_sigma):
    path = _write_ascii_ply(tmp_path / "pts.ply",
                            [(1.0, 2.0, 3.0, bad_sigma)],
                            _XYZ + [("sigma", "double")])
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


@pytest.mark.parametrize("token", ["nan", "inf"])
def test_nonfinite_sigma(tmp_path, token):
    path = tmp_path / "pts.ply"
    header = _ascii_header(_XYZ + [("sigma", "double")], 1)
    path.write_text(header + f"1 2 3 {token}\n")
    with pytest.raises(ValueError):
        list(iter_ply(str(path)))


def test_error_is_terminal_per_block(tmp_path):
    # Two points in a chunk of two; the second is invalid so the whole block
    # fails and the iterator must not be restartable.
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [(1.0, 2.0, 3.0), (4.0, 5.0, float("nan"))], _XYZ)
    result = iter_ply(str(path))
    with pytest.raises(ValueError):
        next(result)
    assert list(result) == []
    with pytest.raises(StopIteration):
        next(result)


# ---------------------------------------------------------------------------
# OS-level failures
# ---------------------------------------------------------------------------

def test_missing_file_raises_oserror(tmp_path):
    result = iter_ply(str(tmp_path / "gone.ply"))
    with pytest.raises(OSError):
        next(result)


def test_opening_directory_raises_oserror(tmp_path):
    # A directory with a .ply name cannot be opened as a regular file.
    target = tmp_path / "dir.ply"
    target.mkdir()
    result = iter_ply(str(target))
    with pytest.raises(OSError):
        next(result)


# ---------------------------------------------------------------------------
# Handle lifecycle
# ---------------------------------------------------------------------------

def test_handle_closed_on_exhaustion(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], _XYZ)
    result = iter_ply(str(path), chunk_size=1)
    blocks = list(result)
    assert sum(len(b) for b in blocks) == 2
    assert result._closed is True
    assert result._file is None
    with pytest.raises(StopIteration):
        next(result)


def test_handle_closed_after_error(tmp_path):
    path = _write_ascii_ply(
        tmp_path / "pts.ply",
        [(1.0, 2.0, 3.0), (4.0, 5.0, -1.0)],
        _XYZ + [("sigma", "double")])
    result = iter_ply(str(path))
    with pytest.raises(ValueError):
        list(result)
    assert result._closed is True
    assert result._file is None
    assert list(result) == []


def test_handle_closed_after_open_failure(tmp_path):
    result = iter_ply(str(tmp_path / "gone.ply"))
    with pytest.raises(OSError):
        next(result)
    assert result._closed is True
    with pytest.raises(StopIteration):
        next(result)


def test_handle_closed_after_bad_header(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_bytes(b"garbage" * 10)
    result = iter_ply(str(path))
    with pytest.raises(ValueError):
        next(result)
    assert result._closed is True
    assert result._file is None


def test_handle_released_on_garbage_collection(tmp_path):
    records = [(float(i), 0.0, 0.0) for i in range(CHUNK + 1)]
    path = _write_binary_ply(tmp_path / "pts.ply", records, _XYZ)
    before = set(os.listdir("/proc/self/fd"))
    result = iter_ply(str(path))
    next(result)  # full block consumed; one point and the handle remain
    assert len(set(os.listdir("/proc/self/fd")) - before) >= 1
    del result
    gc.collect()
    assert set(os.listdir("/proc/self/fd")) == before
