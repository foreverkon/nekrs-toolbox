#pragma once

#include "nrs.hpp"
#include "udf.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mlog {

void setup0(MPI_Comm comm, setupAide& options);

void printf(const char* format, ...);

void metric(int step, const std::string& group,
    const std::vector<std::pair<std::string, double>>& values);

std::ostream& cout();

}
