#pragma once

#include "nrs.hpp"
#include "udf.hpp"

namespace noise {

void setup0(MPI_Comm comm, setupAide& options);
bool enabled();
bool automatic();
void load_kernels(deviceKernelProperties& kernelInfo);
void run(double time);

}
