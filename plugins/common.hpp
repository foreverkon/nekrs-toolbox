#pragma once

#include "nrs.hpp"

#include <limits>
#include <string>

namespace toolbox {

nrs_t* solver();
void check(bool condition, const std::string& message);
std::string kernels_dir();

struct Schedule {
    bool enabled = false;
    bool automatic = true;
    double start = 0;
    int sampleEvery = 1;

    void read(const std::string& section);
    bool due(double time, int step) const;
    void mark(double time, int step);
    void reset();

private:
    double lastTime = -std::numeric_limits<double>::infinity();
};

}
