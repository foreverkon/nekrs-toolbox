#!/usr/bin/env python3

import argparse
import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index
from pathlib import Path
import re

from adios2 import FileReader, ShapeID
import numpy as np


@dataclass(frozen=True)
class MonitorSegment:
    """One segment's attribute metadata; array rows follow ``point_ids``."""

    path: Path
    point_ids: np.ndarray
    coordinates: np.ndarray
    fields: tuple[str, ...]
    num_steps: int


class MonitorReader:
    """Read segmented BP histories by stable point ID.

    Prefixes discover files in segment order; path iterables keep their order.
    Missing points or fields within a segment produce NaN.
    """

    def __init__(self, prefix: str | Path | Iterable[str | Path]):
        if isinstance(prefix, (str, Path)):
            prefix = Path(prefix)
            if prefix.suffix == ".bp" and prefix.exists():
                paths = [prefix]
            else:
                pattern = re.compile(re.escape(prefix.name) + r"\.([0-9]{5}|[1-9][0-9]{5,})\.bp")
                numbered = [(int(match[1]), path) for path in prefix.parent.iterdir()
                            if (match := pattern.fullmatch(path.name))]
                paths = [path for _, path in sorted(numbered)]
        else:
            paths = [Path(path) for path in prefix]
        if not paths:
            raise FileNotFoundError(f"No monitor BP segments found for {prefix!r}")

        self._readers = []
        self._points = {}
        segments = []
        try:
            for path in paths:
                reader = FileReader(str(path))
                self._readers.append(reader)
                segment = self._metadata(reader, path)
                for point_id, coordinates in zip(segment.point_ids, segment.coordinates):
                    point_id = int(point_id)
                    coordinates = tuple(float(value) for value in coordinates)
                    if point_id in self._points and self._points[point_id] != coordinates:
                        raise ValueError(f"Coordinates changed for point ID {point_id} in {path}")
                    self._points[point_id] = coordinates
                segments.append(segment)
        except Exception:
            self.close()
            raise
        self.segments = tuple(segments)
        self.fields = tuple(dict.fromkeys(name for segment in segments for name in segment.fields))

    @staticmethod
    def _metadata(reader, path):
        if reader.read_attribute("toolbox_kind") != "monitor":
            raise ValueError(f"{path} is not a monitor BP segment")
        point_ids = np.asarray(reader.read_attribute("point_ids"))
        coordinates = np.asarray(reader.read_attribute("coordinates"))
        fields = reader.read_attribute("fields")
        fields = (fields,) if isinstance(fields, str) else tuple(fields)
        if (point_ids.ndim != 1 or point_ids.dtype != np.dtype("uint64")
                or len(np.unique(point_ids)) != point_ids.size):
            raise ValueError(f"{path}: point_ids must be unique uint64 IDs")
        if (coordinates.ndim != 1 or coordinates.dtype != np.dtype("float64")
                or coordinates.size != 3 * point_ids.size or not np.isfinite(coordinates).all()):
            raise ValueError(f"{path}: coordinates must contain finite double xyz triples")
        if (not all(isinstance(name, str) and name not in ("time", "tstep") for name in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError(f"{path}: fields must contain distinct field names")
        coordinates = coordinates.reshape(-1, 3)
        point_ids.setflags(write=False)
        coordinates.setflags(write=False)
        num_steps = reader.num_steps() if reader.available_variables() else 0
        if num_steps:
            for name, dtype in (("time", "double"), ("tstep", "int64_t")):
                variable = reader.inquire_variable(name)
                if (variable is None or variable.shape_id() != ShapeID.GlobalValue
                        or variable.type() != dtype or variable.steps() != num_steps):
                    raise ValueError(f"{path}: {name} must be a global {dtype} scalar at every step")
            for name in fields:
                variable = reader.inquire_variable(name)
                if (variable is None or variable.shape_id() != ShapeID.GlobalArray
                        or variable.type() != "double" or variable.shape() != [point_ids.size]
                        or variable.steps() != num_steps):
                    raise ValueError(f"{path}: {name} must be a global double [npoints] array at every step")
        return MonitorSegment(path, point_ids, coordinates, fields, num_steps)

    @property
    def points(self) -> dict[int, tuple[float, float, float]]:
        """Union of stable IDs and xyz coordinates, in first-appearance order."""
        return dict(self._points)

    def __enter__(self) -> "MonitorReader":
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for reader in self._readers:
            reader.close()
        self._readers.clear()

    def _ensure_open(self):
        if not self._readers:
            raise ValueError("MonitorReader is closed")

    def _selection(self, point_ids):
        if point_ids is None:
            return tuple(self._points)
        selected = tuple(index(point_id) for point_id in point_ids)
        unknown = set(selected).difference(self._points)
        if unknown:
            raise KeyError(f"Unknown monitor point IDs: {sorted(unknown)}")
        return selected

    def iter_steps(
        self,
        fields: str | Iterable[str],
        *,
        steps=None,
        point_ids=None,
        time_range: tuple[float | None, float | None] | None = None,
    ) -> Iterator[tuple[int, float, dict[str, np.ndarray]]]:
        """Yield ``(tstep, time, fields)`` in requested point order.

        ``time_range`` includes both endpoints; None leaves an endpoint open.
        """
        self._ensure_open()
        selected_steps = None if steps is None else set(steps)
        selected = self._selection(point_ids)
        fields = (fields,) if isinstance(fields, str) else tuple(fields)
        unknown = set(fields).difference(self.fields)
        if unknown:
            raise KeyError(f"Unknown monitor fields: {sorted(unknown)}")
        start_time, stop_time = (None, None) if time_range is None else time_range
        if ((start_time is not None and np.isnan(start_time))
                or (stop_time is not None and np.isnan(stop_time))
                or (start_time is not None and stop_time is not None and start_time > stop_time)):
            raise ValueError("time_range must have ordered, non-NaN endpoints")
        previous_time = None
        for segment, reader in zip(self.segments, self._readers):
            rows = {int(point_id): row for row, point_id in enumerate(segment.point_ids)}
            positions = [position for position, point_id in enumerate(selected) if point_id in rows]
            selected_rows = np.array([rows[selected[position]] for position in positions], dtype=np.intp)
            if positions:
                first_row = int(selected_rows.min())
                row_count = int(selected_rows.max()) - first_row + 1
            for step in range(segment.num_steps):
                raw_time = float(np.asarray(reader.read("time", step_selection=[step, 1])).item())
                if not np.isfinite(raw_time) or (previous_time is not None and raw_time <= previous_time):
                    raise ValueError(f"{segment.path}, step {step}: times must be finite and strictly increasing")
                previous_time = raw_time
                if ((start_time is not None and raw_time < start_time)
                        or (stop_time is not None and raw_time > stop_time)):
                    continue
                tstep = int(np.asarray(reader.read("tstep", step_selection=[step, 1])).item())
                if selected_steps is not None and tstep not in selected_steps:
                    continue
                data = {}
                for name in fields:
                    values = np.full(len(selected), np.nan)
                    if name in segment.fields and positions:
                        if reader.inquire_variable(name).shape(step) != [len(segment.point_ids)]:
                            raise ValueError(f"{segment.path}: {name} point count changed at step {step}")
                        span = np.asarray(reader.read(
                            name, start=[first_row], count=[row_count], step_selection=[step, 1]
                        )).reshape(-1)
                        values[positions] = span[selected_rows - first_row]
                    data[name] = values
                yield tstep, raw_time, data

    def iter_samples(self, field: str, *, point_ids=None, time_range=None):
        """Yield ``(time, values)`` for one field."""
        for _, raw_time, data in self.iter_steps(field, point_ids=point_ids, time_range=time_range):
            yield raw_time, data[field]

    def read_series(self, field: str, *, point_ids=None, time_range=None) -> tuple[np.ndarray, np.ndarray]:
        """Return raw times and values shaped ``(samples, requested points)``."""
        selected = self._selection(point_ids)
        times, samples = [], []
        for raw_time, values in self.iter_samples(field, point_ids=selected, time_range=time_range):
            times.append(raw_time)
            samples.append(values)
        values = np.stack(samples) if samples else np.empty((0, len(selected)))
        return np.asarray(times, dtype=np.float64), values


def main():
    parser = argparse.ArgumentParser(description="Export monitor BP data to CSV or NPZ")
    parser.add_argument("input", help="Monitor file prefix or a single .bp path")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("-o", "--output", type=Path)
    action.add_argument("--info", action="store_true", help="Print segment metadata as JSON")
    parser.add_argument("--format", choices=("csv", "npz"), default="csv")
    parser.add_argument("--steps", nargs="+", type=int, help="Stored solver tstep values (default: all)")
    parser.add_argument("--fields", nargs="+", help="Field names (default: all)")
    parser.add_argument("--points", nargs="+", type=int, help="Point IDs (default: all)")
    args = parser.parse_args()

    with MonitorReader(args.input) as reader:
        if args.info:
            metadata = [{
                "file": str(segment.path),
                "steps": segment.num_steps,
                "fields": segment.fields,
                "point_ids": segment.point_ids.tolist(),
                "coordinates": segment.coordinates.tolist(),
            } for segment in reader.segments]
            print(json.dumps(metadata, indent=2))
            return
        fields = args.fields or reader.fields
        points = reader.points
        point_ids = args.points or list(points)
        samples = reader.iter_steps(fields, steps=args.steps, point_ids=point_ids)
        if args.format == "csv":
            with args.output.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["time", "tstep", "point_id", *fields])
                for step, time, data in samples:
                    for position, point_id in enumerate(point_ids):
                        writer.writerow([time, step, point_id,
                                         *(data[name][position] for name in fields)])
        else:
            times, steps, values = [], [], []
            for step, time, data in samples:
                times.append(time)
                steps.append(step)
                values.append(np.column_stack([data[name] for name in fields]))
            coordinates = [points[point_id] for point_id in point_ids]
            with args.output.open("wb") as output:
                np.savez(output,
                         time=np.asarray(times, dtype=np.float64),
                         tstep=np.asarray(steps, dtype=np.int64),
                         point_ids=np.asarray(point_ids, dtype=np.uint64),
                         coordinates=np.asarray(coordinates),
                         fields=np.asarray(fields),
                         values=np.asarray(values).reshape(len(times), len(point_ids), len(fields)))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
