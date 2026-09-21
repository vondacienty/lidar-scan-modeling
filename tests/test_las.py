"""Tests for :func:`lidar_scan.iter_las`."""

from __future__ import annotations

import gc
import math
import os
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import laspy
import numpy as np
import pytest

from lidar_scan import iter_las
from laspy import LazBackend

CHUNK = 65536


def _make_file(
    path,
    *,
    n,
    scales=(0.01, 0.02, 0.03),
    offsets=(1.5, 2.5, 3.5),
    compress=False,
    sigma=None,
    sigma_type="f8",
    sigma_scales=None,
    sigma_offsets=None,
    point_format=0,
):
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.offsets = list(offsets)
    header.scales = list(scales)
    if sigma is not None:
        kwargs = {"name": "sigma", "type": np.dtype(sigma_type)}
        if sigma_scales is not None:
            kwargs["scales"] = np.asarray(sigma_scales, dtype=np.float64)
            kwargs["offsets"] = np.asarray(sigma_offsets, dtype=np.float64)
        header.add_extra_dim(laspy.ExtraBytesParams(**kwargs))
    data = laspy.LasData(header)
    if n:
        data.X = np.arange(n, dtype=np.int32)
        data.Y = (np.arange(n, dtype=np.int32) * 2)
        data.Z = (np.arange(n, dtype=np.int32) - 1)
        data.intensity = np.arange(n, dtype=np.uint16)
    if sigma is not None:
        data["sigma"] = np.asarray(sigma)
    data.write(str(path), do_compress=compress)
    return path


def _expected_point(i, scales=(0.01, 0.02, 0.03), offsets=(1.5, 2.5, 3.5),
                    sigma=1.0):
    def q(raw, scale, offset):
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.rounding = ROUND_HALF_EVEN
            value = Decimal(str(raw)) * Decimal(str(scale)) + Decimal(str(offset))
            result = float(value.quantize(Decimal("0.000001")))
            return 0.0 if result == 0.0 else result
    return (q(i, scales[0], offsets[0]),
            q(i * 2, scales[1], offsets[1]),
            q(i - 1, scales[2], offsets[2]),
            i % 65536,
            sigma)


# ---------------------------------------------------------------------------
# Parameter validation (errors raised at call time, before any file access)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [123, 1.5, None, b"/tmp/x.las", ["/tmp/x.las"],
                                 ("/tmp/x.las",), object()])
def test_path_must_be_str(bad):
    with pytest.raises(TypeError):
        iter_las(bad)


@pytest.mark.parametrize("name", ["x.txt", "x.lass", "x.lasx", "x.la", "x",
                                  "x.las.txt", ".lasless"])
def test_bad_extension(name, tmp_path):
    with pytest.raises(ValueError):
        iter_las(str(tmp_path / name))


@pytest.mark.parametrize("ext", [".las", ".LAS", ".Las", ".laz", ".LAZ", ".Laz"])
def test_extension_case_insensitive(tmp_path, ext):
    path = _make_file(tmp_path / "data.tmp", n=1)
    target = tmp_path / f"scan{ext}"
    os.replace(path, target)
    blocks = list(iter_las(str(target)))
    assert len(blocks) == 1
    assert len(blocks[0]) == 1


# ---------------------------------------------------------------------------
# Blocking / streaming shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compress", [False, True])
def test_empty_file_yields_nothing(tmp_path, compress):
    path = _make_file(tmp_path / "empty.las", n=0, compress=compress)
    assert list(iter_las(str(path))) == []


@pytest.mark.parametrize("n", [1, CHUNK - 1, CHUNK, CHUNK + 1,
                               CHUNK * 2 + 3])
def test_block_sizes(tmp_path, n):
    path = _make_file(tmp_path / "pts.las", n=n)
    sizes = [len(block) for block in iter_las(str(path))]
    expected_full, remainder = divmod(n, CHUNK)
    assert sizes == [CHUNK] * expected_full + ([remainder] if remainder else [])
    assert sum(sizes) == n


@pytest.mark.parametrize("compress", [False, True])
def test_full_blocks_exactly_65536(tmp_path, compress):
    path = _make_file(tmp_path / ("pts.laz" if compress else "pts.las"),
                      n=CHUNK * 2, compress=compress)
    blocks = list(iter_las(str(path)))
    assert [len(b) for b in blocks] == [CHUNK, CHUNK]


def test_iterator_is_one_shot_and_self_iterating(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=3)
    result = iter_las(str(path))
    assert iter(result) is result
    first = list(result)
    assert sum(len(b) for b in first) == 3
    # exhausted iterator must not reopen the file
    assert list(result) == []
    with pytest.raises(StopIteration):
        next(result)


def test_lazy_opening(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=2)
    result = iter_las(str(path))
    assert result._reader is None  # file not opened until first next()
    next(result)
    assert result._reader is not None


def test_points_in_file_record_order(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=10)
    points = list(iter_las(str(path)))[0]
    assert [p[3] for p in points] == list(range(10))
    assert points[7] == _expected_point(7)


# ---------------------------------------------------------------------------
# Point decoding
# ---------------------------------------------------------------------------

def test_default_sigma_and_coordinate_math(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=4)
    points = list(iter_las(str(path)))[0]
    for i, point in enumerate(points):
        assert point == _expected_point(i)
        assert point[4] == 1.0


def test_point_is_five_tuple_of_expected_types(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=1)
    point = list(iter_las(str(path)))[0][0]
    assert isinstance(point, tuple) and len(point) == 5
    x, y, z, intensity, sigma = point
    assert all(isinstance(v, float) for v in (x, y, z, sigma))
    assert isinstance(intensity, int) and not isinstance(intensity, bool)


def test_float_sigma_extra_dimension(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=3,
                      sigma=[0.25, 1.5, 10.0], sigma_type="f8")
    points = list(iter_las(str(path)))[0]
    assert [p[4] for p in points] == [0.25, 1.5, 10.0]


def test_integer_sigma_extra_dimension(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=2, sigma=[2, 7],
                      sigma_type="i4")
    points = list(iter_las(str(path)))[0]
    assert [p[4] for p in points] == [2.0, 7.0]


def test_scaled_integer_sigma_extra_dimension(tmp_path):
    # Raw int32 values 500/1250 with scale 0.01 + offset 10 decode to
    # 15.0 / 22.5; the reader must use decoded values, not raw integers.
    path = _make_file(tmp_path / "pts.las", n=2,
                      sigma=[15.0, 22.5], sigma_type="i4",
                      sigma_scales=[0.01], sigma_offsets=[10.0])
    points = list(iter_las(str(path)))[0]
    assert [p[4] for p in points] == [pytest.approx(15.0),
                                      pytest.approx(22.5)]


def test_intensity_uses_standard_field(tmp_path):
    header = laspy.LasHeader(point_format=1, version="1.2")
    data = laspy.LasData(header)
    data.X = np.array([0, 0, 0], dtype=np.int32)
    data.Y = np.array([0, 0, 0], dtype=np.int32)
    data.Z = np.array([0, 0, 0], dtype=np.int32)
    data.intensity = np.array([0, 42, 65535], dtype=np.uint16)
    path = tmp_path / "pf1.las"
    data.write(str(path))
    points = list(iter_las(str(path)))[0]
    assert [p[3] for p in points] == [0, 42, 65535]
    assert all(isinstance(p[3], int) for p in points)


def test_quantized_to_six_decimals_and_negative_zero(tmp_path):
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = [-0.0, -0.0, -0.0]
    header.scales = [1 / 3, 1 / 7, 1.0]
    data = laspy.LasData(header)
    data.X = np.array([1], dtype=np.int32)
    data.Y = np.array([2], dtype=np.int32)
    data.Z = np.array([0], dtype=np.int32)
    data.intensity = np.array([7], dtype=np.uint16)
    path = tmp_path / "pts.las"
    data.write(str(path))
    point = list(iter_las(str(path)))[0][0]
    assert point[0] == round(1 / 3, 6)
    assert point[1] == round(2 / 7, 6)
    assert point[2] == 0.0
    for index in (0, 1, 2):
        assert math.copysign(1.0, point[index]) == 1.0


@pytest.mark.parametrize("compress", [False, True])
def test_las_and_laz_agree(tmp_path, compress):
    path = _make_file(tmp_path / ("pts.laz" if compress else "pts.las"),
                      n=CHUNK + 2, compress=compress,
                      sigma=np.linspace(0.1, 9.0, CHUNK + 2))
    blocks = list(iter_las(str(path)))
    assert len(blocks) == 2
    assert blocks[1][-1][3] == (CHUNK + 1) % 65536


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        list(iter_las(str(tmp_path / "gone.las")))


def test_unparseable_header_raises_value_error(tmp_path):
    path = tmp_path / "bad.las"
    path.write_bytes(b"not a las file" * 20)
    with pytest.raises(ValueError):
        list(iter_las(str(path)))


def test_corrupt_laz_raises_value_error(tmp_path):
    good = _make_file(tmp_path / "good.laz", n=CHUNK * 2 + 3, compress=True)
    raw = good.read_bytes()
    corrupt = tmp_path / "corrupt.laz"
    # Header parses, point stream does not.
    corrupt.write_bytes(raw[:300])
    with pytest.raises(ValueError):
        list(iter_las(str(corrupt)))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_nonfinite_scale_or_offset(tmp_path, bad_value):
    path = _make_file(tmp_path / "pts.las", n=1,
                      scales=(bad_value, 1.0, 1.0),
                      offsets=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        list(iter_las(str(path)))


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"),
                                       float("inf"), float("-inf")])
def test_bad_sigma_value(tmp_path, bad_value):
    path = _make_file(tmp_path / "pts.las", n=2,
                      sigma=[1.0, bad_value], sigma_type="f8")
    with pytest.raises(ValueError):
        list(iter_las(str(path)))


def test_decoder_unavailable_laz_raises_runtime_error(tmp_path, monkeypatch):
    path = _make_file(tmp_path / "pts.laz", n=4, compress=True)
    monkeypatch.setattr(LazBackend.Lazrs, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        list(iter_las(str(path)))


def test_las_still_reads_without_lazrs(tmp_path, monkeypatch):
    path = _make_file(tmp_path / "pts.las", n=3)
    monkeypatch.setattr(LazBackend.Lazrs, "is_available", lambda: False)
    assert sum(len(b) for b in list(iter_las(str(path)))) == 3


# ---------------------------------------------------------------------------
# Handle lifecycle
# ---------------------------------------------------------------------------

def test_handle_closed_on_exhaustion(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=2)
    result = iter_las(str(path))
    list(result)
    assert result._closed is True
    assert result._reader is None
    with pytest.raises(StopIteration):
        next(result)


def test_handle_closed_after_error(tmp_path):
    path = _make_file(tmp_path / "pts.las", n=2,
                      sigma=[1.0, -1.0], sigma_type="f8")
    result = iter_las(str(path))
    with pytest.raises(ValueError):
        list(result)
    assert result._closed is True
    assert result._reader is None
    # no reopening
    assert list(result) == []


def test_handle_closed_after_open_failure(tmp_path):
    result = iter_las(str(tmp_path / "gone.las"))
    with pytest.raises(OSError):
        next(result)
    assert result._closed is True
    with pytest.raises(StopIteration):
        next(result)


def test_handle_released_on_garbage_collection(tmp_path):
    path = _make_file(tmp_path / "pts.laz", n=CHUNK, compress=True)
    before = set(os.listdir("/proc/self/fd"))
    result = iter_las(str(path))
    next(result)
    assert len(set(os.listdir("/proc/self/fd")) - before) >= 1
    del result
    gc.collect()
    assert set(os.listdir("/proc/self/fd")) == before


def test_does_not_read_whole_file(tmp_path, monkeypatch):
    # The whole-file read() API must never be used; only chunk iteration.
    path = _make_file(tmp_path / "pts.las", n=3)
    called = {"read": False}

    def fail_read(*args, **kwargs):  # pragma: no cover - must not run
        called["read"] = True
        raise AssertionError("read() loads the whole file")

    monkeypatch.setattr(laspy.LasReader, "read", fail_read)
    blocks = list(iter_las(str(path)))
    assert sum(len(b) for b in blocks) == 3
    assert called["read"] is False
