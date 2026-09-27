#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace monitor {

struct Point {
    uint64_t id;
    std::array<double, 3> xyz;
};

class ProbeWriter {
public:
    ProbeWriter(MPI_Comm comm, const std::string& prefix,
                const std::vector<Point>& allPoints, const std::vector<std::string>& fields,
                size_t localBegin, size_t localCount, double restartTime, size_t bufferSamples);
    ~ProbeWriter();

    void sample(double time, int step, const std::vector<double>& fieldMajorValues);
    void flush();
    void close();
    double last_time() const;
    std::string path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}
