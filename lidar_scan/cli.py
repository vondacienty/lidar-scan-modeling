"""Command line entry point for lidar-scan-modeling."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from . import tiles as _tiles

_TOP_LEVEL_KEYS = ("points", "windows", "cell_size", "tile_cells", "levels")


def _reject_constant(value: str) -> None:
    """Reject ``NaN``/``Infinity``/``-Infinity`` in JSON input."""
    raise ValueError(f"invalid JSON constant {value!r}")


def _read_stdin_text() -> str:
    """Read all of stdin and decode it as UTF-8.

    :raises ValueError: the bytes are not valid UTF-8.
    """
    stream = sys.stdin
    buffer = getattr(stream, "buffer", None)
    data = (buffer if buffer is not None else stream).read()
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("stdin is not valid UTF-8") from exc


def _parse_window_tiles_document(text: str):
    """Parse and structurally validate the ``window-tiles`` JSON document.

    Returns ``(points, windows, cell_size, tile_cells, levels)`` where
    ``windows`` is a tuple of 5-tuples ready for
    :func:`lidar_scan.tiles.window_tiles`.

    :raises ValueError: the JSON syntax, the top-level object or the array
        structures are bad.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("stdin is not valid JSON") from exc

    if not isinstance(document, dict):
        raise ValueError("top-level value must be a JSON object")
    unknown = set(document) - set(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            "top-level keys must be a subset of "
            "'points', 'windows', 'cell_size', 'tile_cells', 'levels'")
    for name in ("points", "windows"):
        if name not in document:
            raise ValueError(f"top-level key '{name}' is required")

    points = document["points"]
    if not isinstance(points, list):
        raise ValueError("'points' must be an array")
    for point in points:
        if not isinstance(point, list) or len(point) != 5:
            raise ValueError(
                "each point must be an array of 5 items "
                "[x, y, z, intensity, sigma]")

    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")
    for window in raw_windows:
        if not isinstance(window, list) or len(window) != 5:
            raise ValueError(
                "each window must be an array of 5 items "
                "[level, ix_min, iy_min, ix_max, iy_max]")
    windows = tuple(tuple(window) for window in raw_windows)

    return (points, windows,
            document.get("cell_size", 1.0),
            document.get("tile_cells", 256),
            document.get("levels", 3))


def _format_window_tiles_text(results) -> str:
    """Build the canonical compact JSON text of a window-tiles result."""
    parts = ['{"windows":[']
    for window_index, (level, ix_min, iy_min, ix_max, iy_max, tiles) in (
            enumerate(results)):
        if window_index:
            parts.append(",")
        parts.append("[")
        parts.append(",".join((str(level), str(ix_min), str(iy_min),
                               str(ix_max), str(iy_max))))
        parts.append(",[")
        for tile_index, tile in enumerate(tiles):
            if tile_index:
                parts.append(",")
            (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count) = tile
            parts.append("[")
            parts.append(",".join((str(tx), str(ty), str(ix0), str(iy0),
                                   str(ix1), str(iy1), _tiles._format_z(zmin),
                                   _tiles._format_z(zmax), str(count))))
            parts.append("]")
        parts.append("]]")
    parts.append("]}")
    return "".join(parts)


def _run_window_tiles() -> str:
    """Run the ``window-tiles`` command and return the output JSON line.

    :raises TypeError: a point/window field or ``cell_size``/``tile_cells``/
        ``levels`` has the wrong type.
    :raises ValueError: the input is not UTF-8/JSON, the top-level or array
        structure is bad, or a field value is invalid.
    """
    text = _read_stdin_text()
    points, windows, cell_size, tile_cells, levels = (
        _parse_window_tiles_document(text))
    results = _tiles.window_tiles(points, windows, cell_size=cell_size,
                                  tile_cells=tile_cells, levels=levels)
    return _format_window_tiles_text(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lidar-scan-modeling", description="LiDAR point cloud measurement modelling")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")
    sub.add_parser(
        "window-tiles",
        help="aggregate stdin JSON points into window tiles and write "
             "a compact JSON result to stdout")
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "window-tiles":
        try:
            line = _run_window_tiles()
        except (ValueError, TypeError) as exc:
            sys.stderr.write(f"{type(exc).__name__}\n{exc}\n")
            return 2
        sys.stdout.write(line + "\n")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
