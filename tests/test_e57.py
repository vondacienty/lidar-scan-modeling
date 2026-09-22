"""Tests for :func:`lidar_scan.iter_e57`."""

from __future__ import annotations

import gc
import os

import numpy as np
import pytest
from pye57 import E57, libe57

from lidar_scan import iter_e57
import lidar_scan.e57 as e57_module

CHUNK = 65536


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _new_image_file(path):
    image_file = libe57.ImageFile(str(path), "w")
    image_file.extensionsAdd("", libe57.E57_V1_0_URI)
    root = image_file.root()
    root.set("formatName", libe57.StringNode(image_file,
                                             "ASTM E57 3D Imaging Data File"))
    root.set("guid", libe57.StringNode(image_file, "{test}"))
    root.set("versionMajor", libe57.IntegerNode(image_file,
                                               libe57.E57_FORMAT_MAJOR))
    root.set("versionMinor", libe57.IntegerNode(image_file,
                                               libe57.E57_FORMAT_MINOR))
    root.set("data3D", libe57.VectorNode(image_file, True))
    root.set("images2D", libe57.VectorNode(image_file, True))
    return image_file


def _write_raw_scan(image_file, fields, n, values=None, *, integer_fields=(),
                   precision=None):
    """Append one scan to an open image file using raw libe57 nodes.

    Integer-valued fields are written as exact double float nodes (real E57
    files commonly encode intensity that way); they decode to integral floats.
    """
    scan_node = libe57.StructureNode(image_file)
    scan_node.set("guid", libe57.StringNode(image_file, "{scan}"))
    prototype = libe57.StructureNode(image_file)
    for name in fields:
        if name in integer_fields:
            prototype.set(name, libe57.FloatNode(
                image_file, 1.0, libe57.E57_DOUBLE, -1e300, 1e300))
        else:
            prototype.set(name, libe57.FloatNode(
                image_file, 1.0,
                libe57.E57_SINGLE if precision == "single" else libe57.E57_DOUBLE,
                -1e300, 1e300))
    codecs = libe57.VectorNode(image_file, True)
    points = libe57.CompressedVectorNode(image_file, prototype, codecs)
    scan_node.set("points", points)
    image_file.root()["data3D"].append(scan_node)
    if n == 0:
        return
    buffers = libe57.VectorSourceDestBuffer()
    arrays = {}
    for name in fields:
        dtype = "float64" if name in integer_fields else "float64"
        array = np.empty(n, dtype=dtype)
        arrays[name] = array
        buffers.append(libe57.SourceDestBuffer(image_file, name, array, n,
                                               True, True))
    writer = points.writer(buffers)
    for name, values_for_field in values:
        arrays[name][:] = values_for_field
    writer.write(n)
    writer.close()


def _write_cartesian_file(path, scans, *, with_intensity=None):
    """Write a file from cartesian scan dicts via pye57; scans may be empty."""
    if os.path.exists(path):
        os.remove(path)
    image_file = _new_image_file(path)
    for index, data in enumerate(scans):
        n = len(data["cartesianX"])
        if n == 0:
            _write_raw_scan(
                image_file,
                ["cartesianX", "cartesianY", "cartesianZ"], 0)
            continue
        fields = ["cartesianX", "cartesianY", "cartesianZ"]
        if n and with_intensity is not None and with_intensity[index]:
            fields.append("intensity")
        elif n and with_intensity is None and "intensity" in data:
            fields.append("intensity")
        values = tuple((name, data[name]) for name in fields)
        _write_raw_scan(image_file, fields, n, values)
    image_file.close()


def _scan_data(n, *, start=0, with_intensity=True):
    data = {
        "cartesianX": np.arange(start, start + n, dtype="float64") + 0.5,
        "cartesianY": np.arange(start, start + n, dtype="float64") - 1.0,
        "cartesianZ": np.arange(start, start + n, dtype="float64") * 0.25,
    }
    if with_intensity:
        data["intensity"] = np.arange(start * 2, start * 2 + n,
                                     dtype="float64")
    return data


# ---------------------------------------------------------------------------
# Parameter validation (errors raised at call time, before any file access)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [123, 1.5, None, b"/tmp/x.e57", ["/tmp/x.e57"],
                                 ("/tmp/x.e57",), object()])
def test_path_must_be_str(bad):
    with pytest.raises(TypeError):
        iter_e57(bad)


@pytest.mark.parametrize("name", ["x.txt", "x.e57x", "x.e5", "xe57", "x",
                                  "x.e57.txt", ".e57less"])
def test_bad_extension(name, tmp_path):
    with pytest.raises(ValueError):
        iter_e57(str(tmp_path / name))


@pytest.mark.parametrize("ext", [".e57", ".E57", ".E57"])
def test_extension_case_insensitive(tmp_path, ext):
    path = tmp_path / f"scan{ext}"
    _write_cartesian_file(path, [_scan_data(1)])
    blocks = list(iter_e57(str(path)))
    assert len(blocks) == 1
    assert blocks[0][0] == (0.5, -1.0, 0.0, 0, 1.0)


@pytest.mark.parametrize("bad", [True, False, 1.0, 0.0, "4096", None, 1 + 0j,
                                 2 ** 64 * 1.0])
def test_chunk_size_must_be_non_bool_int(tmp_path, bad):
    path = tmp_path / "pts.e57"
    _write_cartesian_file(path, [_scan_data(1)])
    with pytest.raises(TypeError):
        iter_e57(str(path), chunk_size=bad)


@pytest.mark.parametrize("bad", [0, -1, -65536])
def test_chunk_size_must_be_positive(tmp_path, bad):
    path = tmp_path / "pts.e57"
    _write_cartesian_file(path, [_scan_data(1)])
    with pytest.raises(ValueError):
        iter_e57(str(path), chunk_size=bad)


# ---------------------------------------------------------------------------
# Blocking / streaming shape
# ---------------------------------------------------------------------------

def test_empty_container_yields_nothing(tmp_path):
    image_file = _new_image_file(tmp_path / "empty.e57")
    image_file.close()
    assert list(iter_e57(str(tmp_path / "empty.e57"))) == []


def test_zero_point_scan_yields_nothing(tmp_path):
    path = tmp_path / "zero.e57"
    _write_cartesian_file(path, [_scan_data(0)])
    assert list(iter_e57(str(path))) == []


@pytest.mark.parametrize("layout", [
    [CHUNK],
    [CHUNK + 1],
    [CHUNK * 2 + 3],
    [1, 1, 1],
    [2, 0, 3, 0, 1],
])
def test_block_sizes_across_scans(tmp_path, layout):
    scans = [_scan_data(n, start=sum(layout[:i]) * 7)
             for i, n in enumerate(layout)]
    path = tmp_path / "pts.e57"
    _write_cartesian_file(path, scans)
    sizes = [len(block) for block in iter_e57(str(path))]
    total = sum(layout)
    full, remainder = divmod(total, CHUNK)
    assert sizes == [CHUNK] * full + ([remainder] if remainder else [])
    assert sum(sizes) == total


@pytest.mark.parametrize("chunk_size", [1, 3, 1024, 65536])
def test_custom_chunk_size(tmp_path, chunk_size):
    n = chunk_size * 2 + 1
    _write_cartesian_file(tmp_path / "pts.e57", [_scan_data(n)])
    blocks = list(iter_e57(str(tmp_path / "pts.e57"), chunk_size=chunk_size))
    assert [len(b) for b in blocks] == [chunk_size, chunk_size, 1]


def test_scans_concatenated_in_index_and_record_order(tmp_path):
    # scan0: 3 pts (intensity), empty scan, scan2: 2 pts (no intensity)
    scans = [_scan_data(3, start=10),
             _scan_data(0),
             _scan_data(2, start=100, with_intensity=False)]
    path = tmp_path / "ordered.e57"
    _write_cartesian_file(path, scans, with_intensity=[True, True, False])
    blocks = list(iter_e57(str(path), chunk_size=2))
    assert [len(b) for b in blocks] == [2, 2, 1]
    points = tuple(point for block in blocks for point in block)
    assert points == (
        (10.5, 9.0, 2.5, 20, 1.0),
        (11.5, 10.0, 2.75, 21, 1.0),
        (12.5, 11.0, 3.0, 22, 1.0),
        (100.5, 99.0, 25.0, 0, 1.0),
        (101.5, 100.0, 25.25, 0, 1.0),
    )


def test_iterator_is_one_shot_and_self_iterating(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(3), _scan_data(2, start=9)])
    result = iter_e57(str(tmp_path / "pts.e57"), chunk_size=2)
    assert iter(result) is result
    first = list(result)
    assert sum(len(b) for b in first) == 5
    # exhausted iterator must not reopen the file
    assert list(result) == []
    with pytest.raises(StopIteration):
        next(result)


def test_lazy_opening(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57", [_scan_data(1)])
    result = iter_e57(str(tmp_path / "pts.e57"))
    assert result._image_file is None  # file not opened until first next()
    blocks = list(result)
    assert len(blocks) == 1
    assert result._closed is True
    assert result._image_file is None


def test_does_not_read_whole_file(tmp_path, monkeypatch):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(4), _scan_data(3, start=10)])
    chunk_size = 4
    real_buffer = libe57.SourceDestBuffer
    capacities = []

    def tracking_buffer(image_file, name, array, capacity, do_conversion,
                        do_scaling):
        capacities.append(capacity)
        return real_buffer(image_file, name, array, capacity,
                          do_conversion, do_scaling)

    monkeypatch.setattr(e57_module.libe57, "SourceDestBuffer", tracking_buffer)
    blocks = list(iter_e57(str(tmp_path / "pts.e57"), chunk_size=chunk_size))
    assert sum(len(b) for b in blocks) == 7
    assert capacities and max(capacities) == chunk_size


# ---------------------------------------------------------------------------
# Point decoding
# ---------------------------------------------------------------------------

def test_default_intensity_and_sigma(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(2, with_intensity=False)])
    points = list(iter_e57(str(tmp_path / "pts.e57")))[0]
    assert points == ((0.5, -1.0, 0.0, 0, 1.0), (1.5, 0.0, 0.25, 0, 1.0))
    point = points[0]
    assert isinstance(point, tuple) and len(point) == 5
    assert all(isinstance(v, float) for v in
               (point[0], point[1], point[2], point[4]))
    assert isinstance(point[3], int) and not isinstance(point[3], bool)


def test_intensity_values_are_integers(tmp_path):
    image_file = _new_image_file(tmp_path / "ints.e57")
    _write_raw_scan(image_file,
                   ["cartesianX", "cartesianY", "cartesianZ", "intensity"], 3,
                   (("cartesianX", [1.0, 2.0, 3.0]),
                    ("cartesianY", [0.0, 0.0, 0.0]),
                    ("cartesianZ", [0.0, 0.0, 0.0]),
                    ("intensity", [7, 0, -42])),
                   integer_fields=("intensity",))
    image_file.close()
    points = list(iter_e57(str(tmp_path / "ints.e57")))[0]
    assert [p[3] for p in points] == [7, 0, -42]
    assert all(isinstance(p[3], int) for p in points)


def test_scaled_integer_intensity_decoded(tmp_path):
    image_file = _new_image_file(tmp_path / "scaled.e57")
    scan_node = libe57.StructureNode(image_file)
    scan_node.set("guid", libe57.StringNode(image_file, "{scan}"))
    prototype = libe57.StructureNode(image_file)
    prototype.set("cartesianX", libe57.FloatNode(
        image_file, 1.0, libe57.E57_DOUBLE, -1e6, 1e6))
    prototype.set("cartesianY", libe57.FloatNode(
        image_file, 1.0, libe57.E57_DOUBLE, -1e6, 1e6))
    prototype.set("cartesianZ", libe57.FloatNode(
        image_file, 1.0, libe57.E57_DOUBLE, -1e6, 1e6))
    # raw value 5 scaled (2.0) with offset 1.0 -> 11 (must decode exactly)
    prototype.set("intensity", libe57.ScaledIntegerNode(
        image_file, 5, -2000, 2000, 2.0, 1.0))
    codecs = libe57.VectorNode(image_file, True)
    points_node = libe57.CompressedVectorNode(image_file, prototype, codecs)
    scan_node.set("points", points_node)
    image_file.root()["data3D"].append(scan_node)
    n = 1
    buffers = libe57.VectorSourceDestBuffer()
    arrays = {}
    for name, dtype in (("cartesianX", "float64"), ("cartesianY", "float64"),
                        ("cartesianZ", "float64"), ("intensity", "float64")):
        array = np.empty(n, dtype=dtype)
        arrays[name] = array
        buffers.append(libe57.SourceDestBuffer(image_file, name, array, n,
                                               True, True))
    writer = points_node.writer(buffers)
    arrays["cartesianX"][:] = [1.0]
    arrays["cartesianY"][:] = [2.0]
    arrays["cartesianZ"][:] = [3.0]
    arrays["intensity"][:] = [11.0]
    writer.write(n)
    writer.close()
    image_file.close()
    point = list(iter_e57(str(tmp_path / "scaled.e57")))[0][0]
    assert point[3] == 11


def test_quantized_to_six_decimals_and_negative_zero(tmp_path):
    image_file = _new_image_file(tmp_path / "q.e57")
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ"], 1,
                    (("cartesianX", [0.333333333333]),
                     ("cartesianY", [0.1234567]),
                     ("cartesianZ", [-0.0000004])))
    image_file.close()
    point = list(iter_e57(str(tmp_path / "q.e57")))[0][0]
    assert point[0] == round(1 / 3, 6)
    assert point[1] == 0.123457
    assert point[2] == 0.0
    for index in (0, 1, 2):
        assert np.copysign(1.0, point[index]) == 1.0


def test_round_half_even_tie_breaking(tmp_path):
    image_file = _new_image_file(tmp_path / "tie.e57")
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ"], 1,
                    (("cartesianX", [0.1234565]),
                     ("cartesianY", [0.1234575]),
                     ("cartesianZ", [0.1234555])))
    image_file.close()
    x, y, z, _i, _s = list(iter_e57(str(tmp_path / "tie.e57")))[0][0]
    from decimal import Decimal, ROUND_HALF_EVEN

    def expected(token):
        return float(Decimal(token).quantize(Decimal("0.000001"),
                                            rounding=ROUND_HALF_EVEN))

    assert x == expected("0.1234565")
    assert y == expected("0.1234575")
    assert z == expected("0.1234555")


# ---------------------------------------------------------------------------
# Scan / container validation
# ---------------------------------------------------------------------------

def _write_spherical_only(path, n=3):
    image_file = _new_image_file(path)
    _write_raw_scan(
        image_file,
        ["sphericalRange", "sphericalAzimuth", "sphericalElevation"], n,
        (("sphericalRange", [1.0] * n),
         ("sphericalAzimuth", [0.1] * n),
         ("sphericalElevation", [0.0] * n)))
    image_file.close()


def test_spherical_only_scan_rejected(tmp_path):
    _write_spherical_only(tmp_path / "sph.e57")
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "sph.e57")))


@pytest.mark.parametrize("missing", ["cartesianX", "cartesianY", "cartesianZ"])
def test_missing_coordinate_rejected(tmp_path, missing):
    fields = [n for n in
              ("cartesianX", "cartesianY", "cartesianZ") if n != missing]
    image_file = _new_image_file(tmp_path / "miss.e57")
    _write_raw_scan(image_file, fields, 1,
                    tuple((name, [1.0]) for name in fields))
    image_file.close()
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "miss.e57")))


def test_missing_field_in_later_scan_is_terminal(tmp_path):
    image_file = _new_image_file(tmp_path / "mixed.e57")
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ"], 2,
                    (("cartesianX", [1.0, 2.0]),
                     ("cartesianY", [0.0, 0.0]),
                     ("cartesianZ", [0.0, 0.0])))
    _write_raw_scan(image_file,
                    ["cartesianY", "cartesianZ"], 1,
                    (("cartesianY", [1.0]), ("cartesianZ", [2.0])))
    image_file.close()
    result = iter_e57(str(tmp_path / "mixed.e57"), chunk_size=2)
    assert len(next(result)) == 2
    with pytest.raises(ValueError):
        next(result)
    assert list(result) == []


def test_bad_container_rejected(tmp_path):
    path = tmp_path / "bad.e57"
    path.write_bytes(b"garbage" * 20)
    with pytest.raises(ValueError):
        list(iter_e57(str(path)))


def test_truncated_container_rejected(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57", [_scan_data(4)])
    data = (tmp_path / "pts.e57").read_bytes()
    (tmp_path / "trunc.e57").write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "trunc.e57")))


def test_corrupted_records_rejected(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(CHUNK + 10)])
    data = bytearray((tmp_path / "pts.e57").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (tmp_path / "corrupt.e57").write_bytes(bytes(data))
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "corrupt.e57")))


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

def test_non_finite_coordinates_rejected(tmp_path):
    image_file = _new_image_file(tmp_path / "nan.e57")
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ"], 3,
                    (("cartesianX", [1.0, float("nan"), float("inf")]),
                     ("cartesianY", [2.0, 3.0, 4.0]),
                     ("cartesianZ", [5.0, 6.0, 7.0])))
    image_file.close()
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "nan.e57")))


def test_fractional_intensity_rejected(tmp_path):
    _write_cartesian_file(tmp_path / "frac.e57",
                         [{"cartesianX": np.array([1.0]),
                           "cartesianY": np.array([2.0]),
                           "cartesianZ": np.array([3.0]),
                           "intensity": np.array([2.5])}])
    with pytest.raises(ValueError):
        list(iter_e57(str(tmp_path / "frac.e57")))


def test_error_is_terminal_per_block(tmp_path):
    image_file = _new_image_file(tmp_path / "term.e57")
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ"], 2,
                    (("cartesianX", [1.0, 2.0]),
                     ("cartesianY", [0.0, 0.0]),
                     ("cartesianZ", [0.0, 0.0])))
    _write_raw_scan(image_file,
                    ["cartesianX", "cartesianY", "cartesianZ", "intensity"], 1,
                    (("cartesianX", [7.0]),
                     ("cartesianY", [8.0]),
                     ("cartesianZ", [9.0]),
                     ("intensity", [1.5])))
    image_file.close()
    result = iter_e57(str(tmp_path / "term.e57"), chunk_size=2)
    assert [p[0] for p in next(result)] == [1.0, 2.0]
    with pytest.raises(ValueError):
        next(result)
    assert list(result) == []
    with pytest.raises(StopIteration):
        next(result)


# ---------------------------------------------------------------------------
# OS-level failures
# ---------------------------------------------------------------------------

def test_missing_file_raises_oserror(tmp_path):
    result = iter_e57(str(tmp_path / "gone.e57"))
    with pytest.raises(OSError):
        next(result)


def test_opening_directory_raises_oserror(tmp_path):
    target = tmp_path / "dir.e57"
    target.mkdir()
    result = iter_e57(str(target))
    with pytest.raises(OSError):
        next(result)


# ---------------------------------------------------------------------------
# Handle lifecycle
# ---------------------------------------------------------------------------

def test_handle_closed_on_exhaustion(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(2), _scan_data(1, start=9)])
    result = iter_e57(str(tmp_path / "pts.e57"), chunk_size=1)
    blocks = list(result)
    assert sum(len(b) for b in blocks) == 3
    assert result._closed is True
    assert result._image_file is None
    with pytest.raises(StopIteration):
        next(result)


def test_handle_closed_after_error(tmp_path):
    path = tmp_path / "bad.e57"
    path.write_bytes(b"garbage" * 10)
    result = iter_e57(str(path))
    with pytest.raises(ValueError):
        list(result)
    assert result._closed is True
    assert result._image_file is None
    assert list(result) == []


def test_handle_closed_after_open_failure(tmp_path):
    result = iter_e57(str(tmp_path / "gone.e57"))
    with pytest.raises(OSError):
        next(result)
    assert result._closed is True
    with pytest.raises(StopIteration):
        next(result)


def test_handle_released_on_garbage_collection(tmp_path):
    _write_cartesian_file(tmp_path / "pts.e57",
                         [_scan_data(CHUNK + 1)])
    before = set(os.listdir("/proc/self/fd"))
    result = iter_e57(str(tmp_path / "pts.e57"))
    next(result)  # full block consumed; one point and the handle remain
    assert len(set(os.listdir("/proc/self/fd")) - before) >= 1
    del result
    gc.collect()
    assert set(os.listdir("/proc/self/fd")) == before


def test_missing_pye57_raises_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(e57_module, "libe57", None)
    _write_cartesian_file(tmp_path / "pts.e57", [_scan_data(1)])
    result = iter_e57(str(tmp_path / "pts.e57"))
    with pytest.raises(RuntimeError):
        next(result)
    assert result._closed is True
