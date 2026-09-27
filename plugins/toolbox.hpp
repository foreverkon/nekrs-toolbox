#pragma once

#include "alg.hpp"
#include "diagnostics.hpp"
#include "mlog.hpp"
#include "monitor.hpp"
#include "noise.hpp"
#include "statistics.hpp"

namespace toolbox {

void setup0(MPI_Comm comm, setupAide& options);
void load_kernels(deviceKernelProperties& kernelInfo);
void setup();
void setup(double restartTime);
void run(double time, int tstep);
void source(double time);
void finalize();

}
