"""Command line entry point for lidar-scan-modeling."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .tiles import _format_z, publish_migration_journal, window_tiles

_TOP_LEVEL_KEYS = frozenset(
    ("points", "windows", "cell_size", "tile_cells", "levels"))

_MIGRATION_TOP_LEVEL_KEYS = frozenset(
    ("journal_path", "index_path", "before", "after", "max_steps"))
_MIGRATION_REQUIRED_KEYS = ("journal_path", "index_path", "before", "after")


def _reject_constant(raw: str):
    """Reject JSON non-finite literals (``NaN``/``Infinity``) at parse time."""
    raise ValueError(f"non-finite JSON literal is not allowed: {raw}")


def _read_stdin_text() -> str:
    """Read all of stdin and decode it as UTF-8, raising ``ValueError``."""
    stream = sys.stdin
    buffer = getattr(stream, "buffer", None)
    data = buffer.read() if buffer is not None else stream.read()
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("stdin is not valid UTF-8") from exc


def _parse_window_tiles_document(text: str) -> tuple:
    """Validate the stdin JSON document and build ``window_tiles`` arguments.

    Returns ``(points, windows, cell_size, tile_cells, levels)`` where
    ``points`` is a one-shot iterator of five-item lists and ``windows`` is a
    tuple of five-tuples. Structural problems (JSON syntax, top-level keys,
    array shapes) raise ``ValueError``; field type and value problems are
    left to :func:`window_tiles`.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("stdin is not valid JSON") from exc

    if not isinstance(document, dict):
        raise ValueError("top-level value must be a JSON object")
    unknown = set(document) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError("top-level keys must be a subset of 'points', "
                         "'windows', 'cell_size', 'tile_cells' and 'levels'")
    if "points" not in document or "windows" not in document:
        raise ValueError("top-level object must contain 'points' and 'windows'")

    raw_points = document["points"]
    if not isinstance(raw_points, list):
        raise ValueError("'points' must be an array")
    points = []
    for raw_point in raw_points:
        if not isinstance(raw_point, list) or len(raw_point) != 5:
            raise ValueError("each point must be an array of five values "
                             "[x, y, z, intensity, sigma]")
        points.append(raw_point)

    raw_windows = document["windows"]
    if not isinstance(raw_windows, list):
        raise ValueError("'windows' must be an array")
    windows = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list) or len(raw_window) != 5:
            raise ValueError("each window must be an array of five values "
                             "[level, ix_min, iy_min, ix_max, iy_max]")
        windows.append(tuple(raw_window))

    return (iter(points), tuple(windows),
            document.get("cell_size", 1.0),
            document.get("tile_cells", 256),
            document.get("levels", 3))


def _format_window_tiles_text(results: tuple) -> str:
    """Build the canonical compact JSON text of a window-tiles document."""
    parts = ['{"windows":[']
    for window_index, (level, ix_min, iy_min, ix_max, iy_max, tiles) \
            in enumerate(results):
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
                                   str(ix1), str(iy1), _format_z(zmin),
                                   _format_z(zmax), str(count))))
            parts.append("]")
        parts.append("]]")
    parts.append("]}")
    return "".join(parts)


def _run_window_tiles() -> str:
    """Run the ``window-tiles`` subcommand and return the output JSON text."""
    points, windows, cell_size, tile_cells, levels = \
        _parse_window_tiles_document(_read_stdin_text())
    results = window_tiles(points, windows, cell_size=cell_size,
                           tile_cells=tile_cells, levels=levels)
    return _format_window_tiles_text(results)


def _parse_publish_migration_document(text: str) -> tuple:
    """Validate the stdin JSON document and build migration arguments.

    Returns ``(journal_path, index_path, before, after, max_steps)`` where
    ``max_steps`` is ``None`` when the key is omitted or ``null``.
    Structural problems (JSON syntax, top-level keys) raise ``ValueError``;
    field type and value problems are left to
    :func:`publish_migration_journal`.
    """
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    except ValueError as exc:
        raise ValueError("stdin is not valid JSON") from exc

    if not isinstance(document, dict):
        raise ValueError("top-level value must be a JSON object")
    unknown = set(document) - _MIGRATION_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError("top-level keys must be a subset of 'journal_path', "
                         "'index_path', 'before', 'after' and 'max_steps'")
    missing = [key for key in _MIGRATION_REQUIRED_KEYS if key not in document]
    if missing:
        raise ValueError("top-level object must contain 'journal_path', "
                         "'index_path', 'before' and 'after'")

    return (document["journal_path"], document["index_path"],
            document["before"], document["after"],
            document.get("max_steps"))


def _run_publish_migration() -> str:
    """Run the ``publish-migration`` subcommand and return the JSON text."""
    journal_path, index_path, before, after, max_steps = \
        _parse_publish_migration_document(_read_stdin_text())
    return publish_migration_journal(journal_path, index_path, before, after,
                                     max_steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lidar-scan-modeling", description="LiDAR point cloud measurement modelling")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")
    sub.add_parser("window-tiles",
                   help="aggregate stdin JSON points into window tiles")
    sub.add_parser("publish-migration",
                   help="resume a migration journal and publish the index")
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "window-tiles":
        try:
            text = _run_window_tiles()
        except (TypeError, ValueError) as exc:
            sys.stderr.write(type(exc).__name__ + "\n")
            if str(exc):
                sys.stderr.write(str(exc) + "\n")
            return 2
        sys.stdout.write(text + "\n")
        return 0

    if args.command == "publish-migration":
        try:
            text = _run_publish_migration()
        except (TypeError, ValueError, OSError) as exc:
            sys.stderr.write(type(exc).__name__ + "\n")
            if str(exc):
                sys.stderr.write(str(exc) + "\n")
            return 2
        sys.stdout.write(text + "\n")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
