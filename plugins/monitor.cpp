#include "monitor.hpp"
#include "mlog.hpp"
#include "probe_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <set>

namespace monitor {
namespace {
    toolbox::Schedule schedule;
    std::string pointsFile;
    std::string output = "monitor";
    int bufferSamples = 32;
    mesh_t* mesh = nullptr;
    dlong fieldOffset = 0;
    std::vector<Field> fields;
    std::size_t localCount = 0;
    std::unique_ptr<pointInterpolation_t> interpolation;
    std::unique_ptr<ProbeWriter> writer;

    std::vector<Point> read_points()
    {
        const auto comm = platform->comm.mpiComm();
        const auto rank = platform->comm.mpiRank();
        std::vector<Point> points;
        std::string error;
        int count = 0;
        if (rank == 0) {
            try {
                std::ifstream input(pointsFile);
                long long requested = 0;
                if (!(input >> requested) || requested < 1
                    || requested > std::numeric_limits<int>::max() / 3) {
                    throw std::runtime_error("Expected a positive point count in " + pointsFile);
                }
                count = static_cast<int>(requested);
                points.resize(count);
                for (auto& point : points) {
                    std::string id;
                    if (!(input >> id >> point.xyz[0] >> point.xyz[1] >> point.xyz[2])
                        || id.empty() || id.find_first_not_of("0123456789") != std::string::npos
                        || !std::all_of(point.xyz.begin(), point.xyz.end(), [](double coordinate) {
                               return std::isfinite(coordinate);
                           })) {
                        throw std::runtime_error("Expected finite 'id x y z' point records in " + pointsFile);
                    }
                    point.id = std::stoull(id);
                }
                std::string trailing;
                if (input >> trailing) {
                    throw std::runtime_error("Unexpected records after the point count in " + pointsFile);
                }
                std::sort(points.begin(), points.end(), [](const Point& first, const Point& second) {
                    return first.id < second.id;
                });
                for (std::size_t position = 1; position < points.size(); ++position) {
                    if (points[position - 1].id == points[position].id) {
                        throw std::runtime_error("Duplicate monitor point ID in " + pointsFile);
                    }
                }
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        toolbox::check(!error.empty(), error);
        MPI_Bcast(&count, 1, MPI_INT, 0, comm);
        points.resize(count);
        std::vector<std::uint64_t> ids(count);
        std::vector<double> coordinates(3 * count);
        if (rank == 0) {
            for (int position = 0; position < count; ++position) {
                ids[position] = points[position].id;
                std::copy(points[position].xyz.begin(), points[position].xyz.end(), coordinates.begin() + 3 * position);
            }
        }
        MPI_Bcast(ids.data(), count, MPI_UINT64_T, 0, comm);
        MPI_Bcast(coordinates.data(), 3 * count, MPI_DOUBLE, 0, comm);
        for (int position = 0; position < count; ++position) {
            points[position].id = ids[position];
            std::copy_n(coordinates.begin() + 3 * position, 3, points[position].xyz.begin());
        }
        return points;
    }
}

void setup0(MPI_Comm, setupAide&)
{
    schedule.read("monitor");
    pointsFile.clear();
    output = "monitor";
    bufferSamples = 32;
    if (!schedule.enabled) {
        return;
    }
    platform->par->extract("monitor", "pointsfile", pointsFile);
    platform->par->extract("monitor", "output", output);
    platform->par->extract("monitor", "bufferSamples", bufferSamples);
    toolbox::check(pointsFile.empty() || output.empty() || bufferSamples < 1,
        "monitor: pointsfile and output are required; bufferSamples must be positive");
}

void set_fields(mesh_t* meshIn, dlong fieldOffsetIn, const std::vector<Field>& fieldsIn)
{
    toolbox::check(writer != nullptr, "Register monitor fields before toolbox::setup()");
    mesh = meshIn;
    fieldOffset = fieldOffsetIn;
    fields = fieldsIn;
}

bool enabled() { return schedule.enabled; }
bool automatic() { return schedule.enabled && schedule.automatic; }

void setup(double restartTime)
{
    if (!enabled()) {
        return;
    }
    toolbox::check(writer != nullptr, "monitor::setup called twice");
    toolbox::check(!mesh || fields.empty(), "Register a mesh and monitor fields in UDF_Setup before toolbox::setup()");
    toolbox::check(fieldOffset < mesh->Nlocal, "Monitor fieldOffset is smaller than the mesh");
    std::set<std::string> names;
    for (const auto& field : fields) {
        toolbox::check(field.name.empty() || field.name == "time" || field.name == "tstep"
                || !names.insert(field.name).second || field.values.size() < static_cast<std::size_t>(fieldOffset),
            "Monitor field names must be unique, not time/tstep, and buffers must span fieldOffset");
    }
    const auto points = read_points();
    const auto comm = platform->comm.mpiComm();
    const std::size_t rank = platform->comm.mpiRank();
    const std::size_t ranks = platform->comm.mpiCommSize();
    const std::size_t localBegin = points.size() * rank / ranks;
    localCount = points.size() * (rank + 1) / ranks - localBegin;
    std::vector<dfloat> pointsX(localCount), pointsY(localCount), pointsZ(localCount);
    for (std::size_t position = 0; position < localCount; ++position) {
        const auto& point = points[localBegin + position];
        pointsX[position] = point.xyz[0];
        pointsY[position] = point.xyz[1];
        pointsZ[position] = point.xyz[2];
    }
    interpolation = std::make_unique<pointInterpolation_t>(mesh, comm);
    interpolation->setPoints(pointsX, pointsY, pointsZ);
    interpolation->find(pointInterpolation_t::VerbosityLevel::Detailed);
    const auto& codes = interpolation->data().code;
    toolbox::check(std::find(codes.begin(), codes.end(), pointInterpolation_t::CODE_NOT_FOUND) != codes.end(),
        "A monitor point lies outside the interpolation mesh");
    std::vector<std::string> fieldNames;
    for (const auto& field : fields) {
        fieldNames.push_back(field.name);
    }
    writer = std::make_unique<ProbeWriter>(comm, output, points, fieldNames,
        localBegin, localCount, restartTime, static_cast<std::size_t>(bufferSamples));
    schedule.reset();
    if (std::isfinite(writer->last_time())) {
        schedule.mark(writer->last_time(), 0);
    }
    mlog::cout() << "Monitor: " << points.size() << " points, " << fields.size()
                 << " fields, output " << writer->path() << std::endl;
}

bool needs_sample(double time, int tstep)
{
    return schedule.due(time, tstep) && (!writer || time > writer->last_time());
}

void run(double time, int tstep)
{
    if (!needs_sample(time, tstep)) {
        return;
    }
    toolbox::check(!writer, "Call toolbox::setup() before monitor::run()");
    poolDeviceMemory<dfloat> samples(std::max<std::size_t>(1, localCount));
    std::vector<dfloat> host(localCount);
    std::vector<double> values(fields.size() * localCount);
    for (std::size_t field = 0; field < fields.size(); ++field) {
        interpolation->eval(1, fieldOffset, fields[field].values, localCount, samples);
        if (localCount > 0) {
            samples.copyTo(host, localCount);
            std::copy(host.begin(), host.end(), values.begin() + field * localCount);
        }
    }
    writer->sample(time, tstep, values);
    schedule.mark(time, tstep);
}

void flush()
{
    if (writer) {
        writer->flush();
    }
}

void finalize()
{
    if (writer) {
        writer->close();
        writer.reset();
    }
    interpolation.reset();
    fields.clear();
    mesh = nullptr;
}

}
