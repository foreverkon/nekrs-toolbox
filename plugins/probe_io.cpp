#include "probe_io.hpp"

#include <adios2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace monitor {
namespace {

constexpr double noTime = -std::numeric_limits<double>::infinity();

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error("monitor BP5: " + message);
}

void broadcastString(std::string& value, int root, MPI_Comm comm)
{
    uint64_t length = value.size();
    MPI_Bcast(&length, 1, MPI_UINT64_T, root, comm);
    value.resize(length);
    size_t offset = 0;
    while (offset < value.size()) {
        const int count = static_cast<int>(std::min(value.size() - offset,
                                                  size_t(std::numeric_limits<int>::max())));
        MPI_Bcast(value.data() + offset, count, MPI_CHAR, root, comm);
        offset += count;
    }
}

void collectiveCheck(MPI_Comm comm, const std::string& message)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int failedRank = message.empty() ? std::numeric_limits<int>::max() : rank;
    MPI_Allreduce(MPI_IN_PLACE, &failedRank, 1, MPI_INT, MPI_MIN, comm);
    if (failedRank != std::numeric_limits<int>::max()) {
        std::string error = message;
        broadcastString(error, failedRank, comm);
        throw std::runtime_error("monitor BP5: " + error);
    }
}

struct Metadata {
    std::vector<uint64_t> ids;
    std::vector<double> coordinates;
    std::vector<std::string> fields;

    bool operator==(const Metadata& other) const
    {
        return ids == other.ids && coordinates == other.coordinates && fields == other.fields;
    }
};

void validateMetadata(const Metadata& metadata)
{
    require(!metadata.ids.empty(), "the global point set must not be empty");
    require(metadata.ids.size() <= std::numeric_limits<size_t>::max() / 3 &&
                metadata.coordinates.size() == 3 * metadata.ids.size(),
            "coordinates must contain three values per point ID");
    require(std::set<uint64_t>(metadata.ids.begin(), metadata.ids.end()).size() == metadata.ids.size(),
            "point IDs must be unique within a segment");
    for (double coordinate : metadata.coordinates)
        require(std::isfinite(coordinate), "point coordinates must be finite");
    require(!metadata.fields.empty(), "at least one field is required");
    std::set<std::string> names;
    for (const auto& field : metadata.fields)
        require(!field.empty() && field != "time" && field != "tstep" && names.insert(field).second,
                "field names must be nonempty, unique, and distinct from time/tstep");
}

template <typename Value>
std::vector<Value> attribute(adios2::IO& io, const std::string& name)
{
    const auto value = io.InquireAttribute<Value>(name);
    require(bool(value), "missing or incorrectly typed attribute " + name);
    return value.Data();
}

std::string segmentPath(const std::string& prefix, uint64_t index)
{
    std::ostringstream path;
    path << prefix << '.' << std::setfill('0') << std::setw(5) << index << ".bp";
    return path.str();
}

struct Segment {
    uint64_t index;
    std::string path;
    Metadata metadata;
    size_t steps = 0;
    size_t retained = 0;
    double last = noTime;
};

std::vector<Segment> discover(const std::string& prefix)
{
    namespace fs = std::filesystem;
    const fs::path base(prefix);
    const auto directory = base.has_parent_path() ? base.parent_path() : fs::path(".");
    std::vector<Segment> segments;
    if (!fs::exists(directory))
        return segments;
    const std::string stem = base.filename().string() + '.';
    for (const auto& entry : fs::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.compare(0, stem.size(), stem) != 0 || name.size() < stem.size() + 8 ||
            name.compare(name.size() - 3, 3, ".bp") != 0)
            continue;
        const auto digits = name.substr(stem.size(), name.size() - stem.size() - 3);
        if (digits.find_first_not_of("0123456789") != std::string::npos)
            continue;
        const uint64_t index = std::stoull(digits);
        if (segmentPath(base.filename().string(), index) != name)
            continue;
        require(!entry.is_symlink(), "refusing symlink segment " + entry.path().string());
        Segment segment;
        segment.index = index;
        segment.path = segmentPath(prefix, index);
        segments.push_back(std::move(segment));
    }
    std::sort(segments.begin(), segments.end(), [](const Segment& left, const Segment& right) {
        return left.index < right.index;
    });
    return segments;
}

void inspect(Segment& segment, double restartTime)
{
    adios2::ADIOS adios;
    auto io = adios.DeclareIO("inspect");
    io.SetEngine("BP5");
    auto reader = io.Open(segment.path, adios2::Mode::ReadRandomAccess);
    require(attribute<std::string>(io, "toolbox_kind") == std::vector<std::string>{"monitor"},
            "refusing non-monitor segment " + segment.path);
    segment.metadata.ids = attribute<uint64_t>(io, "point_ids");
    segment.metadata.coordinates = attribute<double>(io, "coordinates");
    segment.metadata.fields = attribute<std::string>(io, "fields");
    validateMetadata(segment.metadata);
    segment.steps = reader.Steps();
    if (segment.steps != 0) {
        const auto variables = io.AvailableVariables();
        require(variables.count("time") != 0 && variables.count("tstep") != 0,
                "missing time/tstep variables in " + segment.path);
        auto time = io.InquireVariable<double>("time");
        auto step = io.InquireVariable<int64_t>("tstep");
        require(time && step && time.ShapeID() == adios2::ShapeID::GlobalValue &&
                    step.ShapeID() == adios2::ShapeID::GlobalValue &&
                    time.Steps() == segment.steps && step.Steps() == segment.steps,
                "missing or invalid time/tstep variables in " + segment.path);
        for (const auto& name : segment.metadata.fields) {
            auto field = io.InquireVariable<double>(name);
            require(field && field.ShapeID() == adios2::ShapeID::GlobalArray &&
                        field.Steps() == segment.steps,
                    "missing or invalid field " + name + " in " + segment.path);
            for (size_t index = 0; index < segment.steps; ++index)
                require(field.Shape(index) == adios2::Dims{segment.metadata.ids.size()},
                        "field shape changed within " + segment.path);
        }
        double previous = noTime;
        for (size_t index = 0; index < segment.steps; ++index) {
            double value;
            time.SetStepSelection({index, 1});
            reader.Get(time, value, adios2::Mode::Sync);
            require(std::isfinite(value) && value > previous,
                    "sample times must be finite and strictly increasing in " + segment.path);
            previous = value;
            if (value <= restartTime) {
                ++segment.retained;
                segment.last = value;
            }
        }
    }
    reader.Close();
}

void checkCoordinates(const Metadata& metadata, std::map<uint64_t, std::array<double, 3>>& known)
{
    for (size_t point = 0; point < metadata.ids.size(); ++point) {
        const std::array<double, 3> xyz{metadata.coordinates[3 * point],
                                      metadata.coordinates[3 * point + 1],
                                      metadata.coordinates[3 * point + 2]};
        const auto inserted = known.emplace(metadata.ids[point], xyz);
        require(inserted.second || inserted.first->second == xyz,
                "point ID " + std::to_string(metadata.ids[point]) +
                    " changed coordinates; assign a new ID to a moved point");
    }
}

void prepare(const std::string& prefix, const Metadata& metadata, double restartTime,
             std::string& path, double& last, bool& append)
{
    auto segments = discover(prefix);
    std::map<uint64_t, std::array<double, 3>> known;
    const Segment* latest = nullptr;
    for (auto& segment : segments) {
        try {
            inspect(segment, restartTime);
            checkCoordinates(segment.metadata, known);
        } catch (const std::exception& error) {
            throw std::runtime_error(segment.path + ": " + error.what());
        }
        if (segment.retained != 0) {
            latest = &segment;
            last = std::max(last, segment.last);
        }
    }
    checkCoordinates(metadata, known);
    append = latest && latest->metadata == metadata;
    if (append) {
        path = latest->path;
    } else {
        require(segments.empty() || segments.back().index < std::numeric_limits<uint64_t>::max(),
                "segment index exhausted");
        path = segmentPath(prefix, segments.empty() ? 0 : segments.back().index + 1);
    }
    for (const auto& segment : segments) {
        require(segment.retained <= size_t(std::numeric_limits<int>::max()),
                "too many retained steps for BP5 AppendAfterSteps in " + segment.path);
    }
    for (const auto& segment : segments) {
        if (segment.retained == 0) {
            std::filesystem::remove_all(segment.path);
        } else if (segment.retained < segment.steps) {
            adios2::ADIOS adios;
            auto io = adios.DeclareIO("truncate");
            io.SetEngine("BP5");
            io.SetParameter("AppendAfterSteps", std::to_string(segment.retained));
            io.SetParameter("AsyncOpen", "false");
            auto writer = io.Open(segment.path, adios2::Mode::Append);
            writer.Close();
        }
    }
}

}

struct ProbeWriter::Impl {
    struct Sample {
        double time;
        int64_t step;
        std::vector<double> values;
    };

    MPI_Comm comm;
    int rank = 0;
    size_t localBegin;
    size_t localCount;
    size_t bufferSamples;
    Metadata metadata;
    std::string outputPath;
    double lastTime = noTime;
    bool append = false;
    bool closed = false;
    adios2::ADIOS adios;
    adios2::IO io;
    adios2::Engine engine;
    adios2::Variable<double> timeVariable;
    adios2::Variable<int64_t> stepVariable;
    std::vector<adios2::Variable<double>> fieldVariables;
    std::vector<Sample> pending;

    Impl(MPI_Comm communicator, const std::string& prefix, const std::vector<Point>& points,
         const std::vector<std::string>& fields, size_t begin, size_t count,
         double restartTime, size_t capacity)
        : comm(communicator), localBegin(begin), localCount(count), bufferSamples(capacity),
          adios(communicator)
    {
        MPI_Comm_rank(comm, &rank);
        metadata.fields = fields;
        for (const auto& point : points) {
            metadata.ids.push_back(point.id);
            metadata.coordinates.insert(metadata.coordinates.end(), point.xyz.begin(), point.xyz.end());
        }
        std::string error;
        try {
            validateMetadata(metadata);
            require(!prefix.empty() && !std::filesystem::path(prefix).filename().empty(),
                    "a nonempty filename prefix is required");
            require(std::isfinite(restartTime), "restart time must be finite");
            require(capacity != 0, "bufferSamples must be positive");
            require(begin <= points.size() && count <= points.size() - begin,
                    "local point selection is out of bounds");
            require(count <= std::numeric_limits<size_t>::max() / fields.size(),
                    "local field size overflow");
        } catch (const std::exception& exception) {
            error = exception.what();
        }
        collectiveCheck(comm, error);
        int ranks;
        MPI_Comm_size(comm, &ranks);
        const std::array<uint64_t, 2> selection{begin, count};
        std::vector<uint64_t> selections(2 * size_t(ranks));
        MPI_Allgather(selection.data(), 2, MPI_UINT64_T, selections.data(), 2, MPI_UINT64_T, comm);
        std::vector<std::pair<uint64_t, uint64_t>> intervals;
        for (int peer = 0; peer < ranks; ++peer)
            if (selections[2 * size_t(peer) + 1] != 0)
                intervals.emplace_back(selections[2 * size_t(peer)], selections[2 * size_t(peer) + 1]);
        std::sort(intervals.begin(), intervals.end());
        uint64_t end = 0;
        for (const auto& interval : intervals) {
            require(interval.first == end, "rank selections must cover every point exactly once");
            end += interval.second;
        }
        require(end == points.size(), "rank selections do not cover the global point set");
        if (rank == 0) {
            try {
                prepare(prefix, metadata, restartTime, outputPath, lastTime, append);
            } catch (const std::exception& exception) {
                error = exception.what();
            }
        }
        collectiveCheck(comm, error);
        broadcastString(outputPath, 0, comm);
        MPI_Bcast(&lastTime, 1, MPI_DOUBLE, 0, comm);
        int appendFlag = append;
        MPI_Bcast(&appendFlag, 1, MPI_INT, 0, comm);
        append = appendFlag != 0;
    }

    void open()
    {
        io = adios.DeclareIO("monitor");
        io.SetEngine("BP5");
        io.SetParameter("AsyncOpen", "false");
        io.DefineAttribute<std::string>("toolbox_kind", "monitor");
        io.DefineAttribute<uint64_t>("point_ids", metadata.ids.data(), metadata.ids.size());
        io.DefineAttribute<double>("coordinates", metadata.coordinates.data(), metadata.coordinates.size());
        io.DefineAttribute<std::string>("fields", metadata.fields.data(), metadata.fields.size());
        timeVariable = io.DefineVariable<double>("time");
        stepVariable = io.DefineVariable<int64_t>("tstep");
        for (const auto& field : metadata.fields)
            fieldVariables.push_back(io.DefineVariable<double>(
                field, {metadata.ids.size()}, {localBegin}, {localCount}, adios2::ConstantDims));
        engine = io.Open(outputPath, append ? adios2::Mode::Append : adios2::Mode::Write);
    }

    void flush()
    {
        if (pending.empty())
            return;
        if (!engine)
            open();
        for (const auto& sample : pending) {
            engine.BeginStep();
            if (rank == 0) {
                engine.Put(timeVariable, &sample.time, adios2::Mode::Deferred);
                engine.Put(stepVariable, &sample.step, adios2::Mode::Deferred);
            }
            if (localCount != 0)
                for (size_t field = 0; field < fieldVariables.size(); ++field)
                    engine.Put(fieldVariables[field], sample.values.data() + field * localCount,
                               adios2::Mode::Deferred);
            engine.EndStep();
        }
        pending.clear();
    }

    void close()
    {
        if (closed)
            return;
        flush();
        if (engine)
            engine.Close();
        closed = true;
    }
};

ProbeWriter::ProbeWriter(MPI_Comm comm, const std::string& prefix,
                         const std::vector<Point>& allPoints, const std::vector<std::string>& fields,
                         size_t localBegin, size_t localCount, double restartTime, size_t bufferSamples)
    : impl(std::make_unique<Impl>(comm, prefix, allPoints, fields, localBegin, localCount,
                                  restartTime, bufferSamples))
{
}

ProbeWriter::~ProbeWriter()
{
    try {
        impl->close();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "monitor BP5 close failed: %s\n", error.what());
    }
}

void ProbeWriter::sample(double time, int step, const std::vector<double>& fieldMajorValues)
{
    std::string error;
    if (impl->closed)
        error = "sample called after close";
    else if (!std::isfinite(time) || time <= impl->lastTime)
        error = "sample time must be finite and greater than last_time()";
    else if (fieldMajorValues.size() != impl->metadata.fields.size() * impl->localCount)
        error = "sample values must have fields.size() * localCount entries";
    collectiveCheck(impl->comm, error);
    impl->pending.push_back({time, int64_t(step), fieldMajorValues});
    impl->lastTime = time;
    if (impl->pending.size() >= impl->bufferSamples)
        impl->flush();
}

void ProbeWriter::flush()
{
    impl->flush();
}

void ProbeWriter::close()
{
    impl->close();
}

double ProbeWriter::last_time() const
{
    return impl->lastTime;
}

std::string ProbeWriter::path() const
{
    return impl->outputPath;
}

}
