#include "common.hpp"

#include <cmath>
#include <cstdlib>

namespace toolbox {

nrs_t* solver()
{
    return dynamic_cast<nrs_t*>(platform->app);
}

void check(bool condition, const std::string& message)
{
    nekrsCheck(condition, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", message.c_str());
}

std::string kernels_dir()
{
#ifdef NEKRS_TOOLBOX_KERNEL_DIR
    return std::string(NEKRS_TOOLBOX_KERNEL_DIR) + "/";
#else
    const char* directory = std::getenv("NEKRS_PLUGINS_DIR");
    check(!directory, "Set NEKRS_PLUGINS_DIR or include plugins.cmake before loading toolbox kernels");
    return std::string(directory) + "/kernels/";
#endif
}

void Schedule::read(const std::string& section)
{
    *this = Schedule {};
    auto& parameters = *platform->par;
    parameters.extract(section, "enable", enabled);
    if (!enabled) {
        return;
    }
    std::string execution = "auto";
    parameters.extract(section, "execution", execution);
    parameters.extract(section, "start", start);
    parameters.extract(section, "sampleEvery", sampleEvery);
    execution = lowerCase(execution);
    check(execution != "auto" && execution != "manual", section + ": execution must be auto or manual");
    check(!std::isfinite(start) || sampleEvery < 1,
        section + ": start must be finite, sampleEvery positive");
    automatic = execution == "auto";
}

bool Schedule::due(double time, int step) const
{
    return enabled && solver()->timeStepConverged && time >= start && time > lastTime
        && step % sampleEvery == 0;
}

void Schedule::mark(double time, int)
{
    lastTime = time;
}

void Schedule::reset()
{
    lastTime = -std::numeric_limits<double>::infinity();
}

}
