#include "toolbox.hpp"
#include "common.hpp"

namespace toolbox {

void setup0(MPI_Comm comm, setupAide& options)
{
    mlog::setup0(comm, options);
    monitor::setup0(comm, options);
    diagnostics::setup0(comm, options);
    statistics::setup0(comm, options);
    noise::setup0(comm, options);
}

void load_kernels(deviceKernelProperties& kernelInfo)
{
    alg::load_kernels(kernelInfo);
    noise::load_kernels(kernelInfo);
}

void setup()
{
    double restartTime = 0;
    platform->options.getArgs("START TIME", restartTime);
    setup(restartTime);
}

void setup(double restartTime)
{
    check(solver() == nullptr, "The toolbox requires the nekRS NRS application");
    monitor::setup(restartTime);
    diagnostics::setup(restartTime);
    statistics::setup(restartTime);
}

void run(double time, int tstep)
{
    if (monitor::automatic()) {
        monitor::run(time, tstep);
    }
    if (diagnostics::automatic()) {
        diagnostics::run(time, tstep);
    }
    if (statistics::automatic()) {
        statistics::run(time, tstep);
    }
}

void source(double time)
{
    if (noise::automatic()) {
        noise::run(time);
    }
}

void finalize()
{
    monitor::finalize();
    diagnostics::finalize();
    statistics::finalize();
}

}
