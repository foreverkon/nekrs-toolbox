#!/usr/bin/env python3

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import zlib

from adios2 import FileReader
import h5py as h5
import numpy as np


HDF_CHUNK_BYTES = 64 * 1024 * 1024


def hdf_chunks(shape, dtype):
    if shape[0] == 0:
        return None
    tuple_bytes = np.dtype(dtype).itemsize * int(np.prod(shape[1:]))
    rows = max(1, HDF_CHUNK_BYTES // tuple_bytes)
    return (min(shape[0], rows),) + tuple(shape[1:])


def compress_chunk(values, level):
    shuffled = values.view(np.uint8).reshape(-1, values.dtype.itemsize).T.copy()
    return zlib.compress(shuffled, level)


def write_chunks(dataset, arrays, executor, level, max_pending):
    offset = 0
    if not level:
        for values in arrays:
            dataset[offset:offset + len(values)] = values
            offset += len(values)
        return

    pending = deque()
    buffer = np.empty(dataset.chunks, dtype=dataset.dtype)
    used = 0

    def drain():
        start, future = pending.popleft()
        dataset.id.write_direct_chunk(
            (start,) + (0,) * (dataset.ndim - 1), future.result()
        )

    def submit(values):
        nonlocal offset
        pending.append((offset, executor.submit(compress_chunk, values, level)))
        offset += len(values)
        if len(pending) >= max_pending:
            drain()

    for values in arrays:
        start = 0
        while start < len(values):
            count = min(len(values) - start, len(buffer) - used)
            buffer[used:used + count] = values[start:start + count]
            used += count
            start += count
            if used == len(buffer):
                submit(buffer)
                buffer = np.empty(dataset.chunks, dtype=dataset.dtype)
                used = 0
    if used:
        buffer[used:] = 0
        submit(buffer)
    while pending:
        drain()


def block_map(reader, name, step):
    """Match blocks by writer ID and within-writer order."""
    blocks = reader.engine.blocks_info(name, step)
    result = {}
    counts = {}
    for block in sorted(blocks, key=lambda b: int(b["BlockID"])):
        writer = int(block["WriterID"])
        ordinal = counts.get(writer, 0)
        result[writer, ordinal] = block
        counts[writer] = ordinal + 1
    if not result:
        raise ValueError(f"No blocks for {name!r} at step {step}")
    return result


class StaticMeshBP:
    def __init__(self, reader):
        self.reader = reader
        self.n_steps = reader.num_steps()
        if self.n_steps < 1:
            raise ValueError("No steps found in the input file")
        self.times = np.asarray(
            reader.read("time", step_selection=[0, self.n_steps])
        ).reshape(-1)
        if self.times.size != self.n_steps:
            raise ValueError("Expected exactly one time value per step")

        mesh_blocks = block_map(reader, "mesh", 0)
        conn_blocks = block_map(reader, "connectivity", 0)
        if mesh_blocks.keys() != conn_blocks.keys():
            raise ValueError("Mesh and connectivity writer blocks do not match")
        self.keys = sorted(mesh_blocks)
        self.sizes = {}
        point_parts, cell_parts = [], []
        point_offset = 0
        for key in self.keys:
            points = self.read_block("mesh", mesh_blocks[key], 0)
            cells = self.read_block("connectivity", conn_blocks[key], 0)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Expected 3D points in block {key}: {points.shape}")
            if (
                cells.ndim != 2
                or cells.shape[1] != 9
                or not np.issubdtype(cells.dtype, np.integer)
                or not np.all(cells[:, 0] == 8)
            ):
                raise ValueError(f"Expected eight-node hexahedra in block {key}")
            ids = cells[:, 1:]
            if ids.size and (ids.min() < 0 or ids.max() >= len(points)):
                raise ValueError(f"Connectivity out of bounds in block {key}")
            self.sizes[key] = len(points)
            point_parts.append(points)
            cell_parts.append(ids.astype(np.int64) + point_offset)
            point_offset += len(points)
        self.points = np.concatenate(point_parts)
        self.cells = np.concatenate(cell_parts)
        self.num_points = len(self.points)
        if self.num_points == 0:
            raise ValueError("The mesh has no points")

        variables = reader.available_variables()
        for name, initial_blocks, initial_values in (
            ("mesh", mesh_blocks, point_parts),
            ("connectivity", conn_blocks, cell_parts),
        ):
            for step in range(1, int(variables[name]["AvailableStepsCount"])):
                blocks = block_map(reader, name, step)
                if blocks.keys() != initial_blocks.keys():
                    raise ValueError("Only static meshes are supported")
                point_offset = 0
                for key, initial in zip(self.keys, initial_values):
                    values = self.read_block(name, blocks[key], step)
                    if name == "connectivity":
                        values = values[:, 1:].astype(np.int64) + point_offset
                    if not np.array_equal(values, initial):
                        raise ValueError("Only static meshes are supported")
                    point_offset += self.sizes[key]
        mesh_variables = {
            "numOfCells", "numOfPoints", "types",
            "connectivity", "globalElementIds", "hRefineSchedule", "mesh", "time",
        }
        point_types = {"float": np.dtype("f4"), "double": np.dtype("f8")}
        self.fields = []
        self.layouts = {}
        for name, info in sorted(variables.items()):
            if (name in mesh_variables or name == "polynomialOrder"
                    or info.get("SingleValue", "false").lower() == "true"
                    or info["Type"] not in point_types):
                continue
            if int(info["AvailableStepsCount"]) != self.n_steps:
                raise ValueError(f"Point field {name!r} is not present at every step")
            blocks = self.field_blocks(name, 0)
            components = None
            for key in self.keys:
                shape = tuple(int(n) for n in blocks[key]["Count"].split(",") if n.strip())
                if len(shape) not in (1, 2) or shape[0] != self.sizes[key]:
                    raise ValueError(
                        f"Non-scalar variable {name!r}, block {key}: "
                        f"shape {shape} does not match {self.sizes[key]} mesh points"
                    )
                if components is not None and shape[1:] != components:
                    raise ValueError(f"Point field {name!r} has inconsistent block components")
                components = shape[1:]
            self.fields.append(name)
            self.layouts[name] = (components, point_types[info["Type"]])
        if not self.fields:
            print("No point fields found; writing mesh and time values only.")
        candidates = sorted(set(variables) - set(self.fields) - mesh_variables)
        self.metadata = {}
        skipped = []
        for name in candidates:
            info = variables[name]
            if info.get("SingleValue", "false").lower() != "true":
                skipped.append(name)
                continue
            count = int(info["AvailableStepsCount"])
            if count not in (1, self.n_steps):
                raise ValueError(f"Metadata {name!r} must be static or present at every step")
            if count == 1 and not reader.engine.blocks_info(name, 0):
                raise ValueError(f"Static metadata {name!r} is missing at the initial step")
            values = np.asarray(reader.read(name, step_selection=[0, count])).reshape(-1)
            if values.size != count or values.dtype.kind not in "fiu":
                raise ValueError(f"Expected numeric single-value metadata for {name!r}")
            self.metadata[name] = values
        if self.metadata:
            print("Saving single-value metadata:", ", ".join(self.metadata))
        if skipped:
            print("Skipping non-point arrays:", ", ".join(skipped))
        print(f"Mesh: {len(self.keys)} blocks, {self.num_points} points, "
              f"{len(self.cells)} cells; {self.n_steps} time steps")

    def read_block(self, name, block, step):
        return np.asarray(self.reader.read(
            name, block_id=int(block["BlockID"]), step_selection=[step, 1]
        ))

    def field_blocks(self, name, step):
        blocks = block_map(self.reader, name, step)
        if blocks.keys() != self.sizes.keys():
            raise ValueError(f"Writer blocks for {name!r}, step {step} do not match the mesh")
        return blocks

    def iter_field(self, name, step):
        blocks = self.field_blocks(name, step)
        tail, dtype = self.layouts[name]
        offset = 0
        for key in self.keys:
            values = self.read_block(name, blocks[key], step)
            expected = (self.sizes[key],) + tail
            if values.shape != expected or values.dtype != dtype:
                raise ValueError(
                    f"Field {name!r}, step {step}, block {key}: expected "
                    f"{expected}/{dtype}, got {values.shape}/{values.dtype}"
                )
            yield offset, values
            offset += self.sizes[key]


def to_vtkhdf(input: str | Path, output: str | Path, compression_level: int = 1,
              compression_threads: int = 4):
    if not 0 <= compression_level <= 9:
        raise ValueError("Compression level must be between 0 and 9")
    compression = (
        {"compression": "gzip", "compression_opts": compression_level, "shuffle": True}
        if compression_level else {}
    )
    started = perf_counter()
    with FileReader(str(input)) as reader:
        print("Reading shared geometry...", flush=True)
        bp = StaticMeshBP(reader)
        print(f"Mesh and metadata read: {perf_counter() - started:.3f} s", flush=True)
        with ThreadPoolExecutor(max_workers=compression_threads) as executor, \
                h5.File(output, "w") as writer:
            root = writer.create_group("VTKHDF")
            root.attrs["Version"] = np.array([2, 3], dtype="i8")
            root.attrs["Type"] = np.bytes_("UnstructuredGrid")
            steps = root.create_group("Steps")
            steps.attrs["NSteps"] = bp.n_steps
            steps.create_dataset("Values", data=bp.times)
            for name in ("PartOffsets", "PointOffsets"):
                steps.create_dataset(name, data=np.zeros(bp.n_steps, dtype="i8"))
            for name in ("CellOffsets", "ConnectivityIdOffsets"):
                steps.create_dataset(name, data=np.zeros((bp.n_steps, 1), dtype="i8"))
            steps.create_dataset("NumberOfParts", data=np.ones(bp.n_steps, dtype="i8"))

            geometry_started = perf_counter()
            print("Writing shared geometry...", flush=True)
            n_cells = len(bp.cells)
            root.create_dataset("NumberOfPoints", data=[bp.num_points], dtype="i8")
            root.create_dataset("NumberOfCells", data=[n_cells], dtype="i8")
            root.create_dataset("NumberOfConnectivityIds", data=[bp.cells.size], dtype="i8")
            root.create_dataset("Types", data=np.full(n_cells, 12, dtype="u1"))
            root.create_dataset("Offsets", data=np.arange(n_cells + 1, dtype="i8") * 8)
            for name, values in (("Points", bp.points), ("Connectivity", bp.cells.ravel())):
                dataset = root.create_dataset(
                    name, shape=values.shape, dtype=values.dtype,
                    chunks=hdf_chunks(values.shape, values.dtype), **compression
                )
                if values.size:
                    write_chunks(dataset, (values,), executor,
                                 compression_level, compression_threads)
            del values, bp.points, bp.cells
            print(f"Geometry output: {perf_counter() - geometry_started:.3f} s", flush=True)

            if bp.metadata:
                field_data = root.create_group("FieldData")
                field_offsets = steps.create_group("FieldDataOffsets")
                field_sizes = steps.create_group("FieldDataSizes")
                for name, values in bp.metadata.items():
                    field_data.create_dataset(name, data=values)
                    field_offsets.create_dataset(
                        name, data=(np.zeros(bp.n_steps, dtype="i8") if len(values) == 1
                                    else np.arange(bp.n_steps, dtype="i8"))
                    )
                    field_sizes.create_dataset(name, data=np.ones((bp.n_steps, 2), dtype="i8"))

            offsets = steps.create_group("PointDataOffsets")
            data = root.create_group("PointData")
            for name, (tail, dtype) in bp.layouts.items():
                offsets.create_dataset(
                    name, data=np.arange(bp.n_steps, dtype="i8") * bp.num_points
                )
                data.create_dataset(
                    name,
                    shape=(bp.n_steps * bp.num_points,) + tail,
                    dtype=dtype,
                    chunks=hdf_chunks((bp.num_points,) + tail, dtype),
                    **compression,
                )
            for name in bp.fields:
                field_started = perf_counter()
                read_seconds = 0.0

                def arrays():
                    nonlocal read_seconds
                    for step, time in enumerate(bp.times):
                        print(f"  {name}: step {step + 1}/{bp.n_steps} (time={time})...",
                              flush=True)
                        blocks = iter(bp.iter_field(name, step))
                        for _ in bp.keys:
                            read_started = perf_counter()
                            _, values = next(blocks)
                            read_seconds += perf_counter() - read_started
                            yield values

                write_chunks(data[name], arrays(), executor,
                             compression_level, compression_threads)
                elapsed = perf_counter() - field_started
                print(f"  {name}: total {elapsed:.3f} s, ADIOS read {read_seconds:.3f} s, "
                      f"output pipeline {elapsed - read_seconds:.3f} s", flush=True)
    print(f"Saved {output} in {perf_counter() - started:.3f} s", flush=True)


def to_xdmf(input: str | Path, output: str | Path):
    import meshio

    class AdjacentHDFWriter(meshio.xdmf.TimeSeriesWriter):
        def __enter__(self):
            # meshio 5.3 writes the sidecar in the working directory by default.
            # Keep it beside the XDMF so relative references resolve correctly.
            self.h5_filename = str(self.filename.with_suffix(".h5"))
            self.h5_file = h5.File(self.h5_filename, "w")
            return self

    with FileReader(str(input)) as reader:
        bp = StaticMeshBP(reader)
        with AdjacentHDFWriter(output) as writer:
            writer.write_points_cells(bp.points, [("hexahedron", bp.cells)])
            for step, time in enumerate(bp.times):
                print(f"Writing step {step + 1}/{bp.n_steps} (time={time})...")
                data = {
                    name: np.concatenate([values for _, values in bp.iter_field(name, step)])
                    for name in bp.fields
                }
                writer.write_data(float(time), point_data=data)
                for name, values in bp.metadata.items():
                    value = values[0 if len(values) == 1 else step]
                    info = ET.SubElement(writer.collection[-1], "Information",
                                         Name=name, Value=str(value.item()))
                    ET.SubElement(info, "Information", Name="dtype", Value=values.dtype.str)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inp", type=Path, help="Input nekRS BP file")
    parser.add_argument("-t", choices=["xdmf", "vtkhdf"], default="vtkhdf")
    parser.add_argument("-o", type=Path, help="Output path")
    parser.add_argument("-c", type=int, choices=range(10), default=1,
                        help="VTKHDF gzip level (0 disables compression; default: 1)")
    parser.add_argument("--compression-threads", type=int, default=4,
                        help="VTKHDF compression workers (default: 4)")
    args = parser.parse_args()
    if args.compression_threads < 1:
        parser.error("Compression threads must be at least 1")
    output = args.o if args.o is not None else args.inp.with_suffix("." + args.t)
    if not args.inp.exists():
        parser.error(f"Input {args.inp} does not exist")
    targets = [output]
    if args.t == "xdmf":
        targets.append(output.with_suffix(".h5"))
    if any(target.resolve() == args.inp.resolve() for target in targets):
        parser.error("Output must not overwrite the input")
    if len({target.resolve() for target in targets}) != len(targets):
        parser.error("XDMF output and its .h5 sidecar must have distinct paths")
    for target in targets:
        if target.exists():
            if not target.is_file():
                parser.error(f"Output {target} is not a regular file")
            if input(f"Output {target} exists. Overwrite? (y/N): ").strip().lower() != "y":
                print("Aborting.")
                return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {args.inp} to {output}...")
    if args.t == "xdmf":
        to_xdmf(args.inp, output)
    else:
        to_vtkhdf(args.inp, output, args.c, args.compression_threads)


if __name__ == "__main__":
    main()
