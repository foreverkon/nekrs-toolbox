#include "noise.hpp"
#include "utils.hpp"
#include "common.hpp"

#include <cmath>
#include <iomanip>

namespace noise {

namespace {
    bool enable = false;
    bool autoExecute = false;
    dfloat xmin = 0.0;
    dfloat xmax = 0.0;
    dfloat ymin = 0.0;
    dfloat ymax = 0.0;
    dfloat start = 0.0;
    dfloat duration = 0.0;
    dfloat amplitude = 0.0;
    dfloat lengthZ = 0.0;
    dfloat lengthXY = 0.0;
    dfloat velocity = 0.0;
    dfloat forcingScale = 0.0;
    dfloat frequency1X = 0.0;
    dfloat frequency1Y = 0.0;
    dfloat frequency1Z = 0.0;
    dfloat frequency1T = 0.0;
    dfloat frequency2X = 0.0;
    dfloat frequency2Y = 0.0;
    dfloat frequency2Z = 0.0;
    dfloat frequency2T = 0.0;
    dfloat frequency3X = 0.0;
    dfloat frequency3Y = 0.0;
    dfloat frequency3Z = 0.0;
    dfloat frequency3T = 0.0;
}

bool enabled() { return enable; }
bool automatic() { return enable && autoExecute; }

void setup0(MPI_Comm comm, setupAide& options)
{
    platform->par->extract("noise", "enable", enable);
    if (!enable) {
        mlog::printf("\n========== Noise plugin disabled\n\n");
        return;
    }

    std::string execution = "manual";
    platform->par->extract("noise", "execution", execution);
    execution = lowerCase(execution);
    toolbox::check(execution != "manual" && execution != "auto", "noise: execution must be auto or manual");
    autoExecute = execution == "auto";
    dfloat channelLength = 0;
    dfloat hydraulicDiameter = 0;
    dfloat bulkVelocity = 0;
    bool complete = true;
    complete &= platform->par->extract("noise", "channelLength", channelLength);
    complete &= platform->par->extract("noise", "hydraulicDiameter", hydraulicDiameter);
    complete &= platform->par->extract("noise", "bulkVelocity", bulkVelocity);
    complete &= platform->par->extract("noise", "start", start);
    complete &= platform->par->extract("noise", "duration", duration);
    complete &= platform->par->extract("noise", "amplitude", amplitude);
    complete &= platform->par->extract("noise", "xmin", xmin);
    complete &= platform->par->extract("noise", "xmax", xmax);
    complete &= platform->par->extract("noise", "ymin", ymin);
    complete &= platform->par->extract("noise", "ymax", ymax);
    complete &= platform->par->extract("noise", "frequency1X", frequency1X);
    complete &= platform->par->extract("noise", "frequency1Y", frequency1Y);
    complete &= platform->par->extract("noise", "frequency1Z", frequency1Z);
    complete &= platform->par->extract("noise", "frequency1T", frequency1T);
    complete &= platform->par->extract("noise", "frequency2X", frequency2X);
    complete &= platform->par->extract("noise", "frequency2Y", frequency2Y);
    complete &= platform->par->extract("noise", "frequency2Z", frequency2Z);
    complete &= platform->par->extract("noise", "frequency2T", frequency2T);
    complete &= platform->par->extract("noise", "frequency3X", frequency3X);
    complete &= platform->par->extract("noise", "frequency3Y", frequency3Y);
    complete &= platform->par->extract("noise", "frequency3Z", frequency3Z);
    complete &= platform->par->extract("noise", "frequency3T", frequency3T);

    nekrsCheck(!complete,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "Enabled [NOISE] section is missing one or more required parameters.\n");
    nekrsCheck(channelLength <= 0.0 || hydraulicDiameter <= 0.0 || bulkVelocity <= 0.0,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "Noise scaling requires positive channelLength, hydraulicDiameter and bulkVelocity.\n");
    nekrsCheck(duration <= 0.0,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "NOISE:duration must be positive (seconds).\n");
    nekrsCheck(amplitude < 0.0,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "NOISE:amplitude must be non-negative.\n");
    nekrsCheck(xmin >= xmax || ymin >= ymax,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "NOISE bounds must satisfy xmin < xmax and ymin < ymax.\n");
    const bool periodicZ =
        std::abs(frequency1Z - std::round(frequency1Z)) < 1e-12 &&
        std::abs(frequency2Z - std::round(frequency2Z)) < 1e-12 &&
        std::abs(frequency3Z - std::round(frequency3Z)) < 1e-12;
    nekrsCheck(!periodicZ,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "NOISE frequency1Z, frequency2Z and frequency3Z must be integers for axial periodicity.\n");

    dfloat rho = 0.0;
    options.getArgs("FLUID DENSITY", rho);
    nekrsCheck(rho <= 0.0,
               comm,
               EXIT_FAILURE,
               "%s\n",
               "Noise scaling requires a positive fluid density.\n");

    lengthZ = channelLength;
    lengthXY = hydraulicDiameter;
    velocity = bulkVelocity;
    forcingScale = amplitude * rho * bulkVelocity * bulkVelocity / hydraulicDiameter;

    mlog::cout()
        << std::setprecision(12)
        << "\n========== Noise parameters ==========\n"
        << "start                         [s]     : " << start << "\n"
        << "duration                      [s]     : " << duration << "\n"
        << "dimensionless amplitude       [-]     : " << amplitude << "\n"
        << "momentum-source scale         [N/m^3] : " << forcingScale << "\n"
        << "x bounds                      [m]     : " << xmin << ", " << xmax << "\n"
        << "y bounds                      [m]     : " << ymin << ", " << ymax << "\n"
        << "frequency 1 (x*,y*,z*,t*)     [-]     : "
        << frequency1X << ", " << frequency1Y << ", " << frequency1Z << ", " << frequency1T << "\n"
        << "frequency 2 (x*,y*,z*,t*)     [-]     : "
        << frequency2X << ", " << frequency2Y << ", " << frequency2Z << ", " << frequency2T << "\n"
        << "frequency 3 (x*,y*,z*,t*)     [-]     : "
        << frequency3X << ", " << frequency3Y << ", " << frequency3Z << ", " << frequency3T << "\n"
        << "======================================\n"
        << std::endl;
}

void load_kernels(deviceKernelProperties& kernelInfo)
{
    if (enable) {
        occa::json prop = kernelInfo;
        prop["defines/p_noise_forcing_scale"] = forcingScale;
        prop["defines/p_noise_xmin"] = xmin;
        prop["defines/p_noise_xmax"] = xmax;
        prop["defines/p_noise_ymin"] = ymin;
        prop["defines/p_noise_ymax"] = ymax;
        prop["defines/p_noise_length_z"] = lengthZ;
        prop["defines/p_noise_length_xy"] = lengthXY;
        prop["defines/p_noise_velocity"] = velocity;
        prop["defines/p_noise_start"] = start;
        prop["defines/p_noise_duration"] = duration;
        prop["defines/p_noise_frequency_1x"] = frequency1X;
        prop["defines/p_noise_frequency_1y"] = frequency1Y;
        prop["defines/p_noise_frequency_1z"] = frequency1Z;
        prop["defines/p_noise_frequency_1t"] = frequency1T;
        prop["defines/p_noise_frequency_2x"] = frequency2X;
        prop["defines/p_noise_frequency_2y"] = frequency2Y;
        prop["defines/p_noise_frequency_2z"] = frequency2Z;
        prop["defines/p_noise_frequency_2t"] = frequency2T;
        prop["defines/p_noise_frequency_3x"] = frequency3X;
        prop["defines/p_noise_frequency_3y"] = frequency3Y;
        prop["defines/p_noise_frequency_3z"] = frequency3Z;
        prop["defines/p_noise_frequency_3t"] = frequency3T;
        platform->kernelRequests.add("noise::add_noise", toolbox::kernels_dir() + "add_noise.okl", prop);
    }
}

void run(double time)
{
    if (!enable) {
        return;
    }
    const auto nrs = toolbox::solver();
    auto mesh = nrs->meshV;

    if (time >= start && time <= (start + duration)) {
        LaunchKernel("noise::add_noise",
            mesh->Nlocal,
            time,
            mesh->o_x,
            mesh->o_y,
            mesh->o_z,
            nrs->fluid->o_explicitTerms("x"),
            nrs->fluid->o_explicitTerms("y"),
            nrs->fluid->o_explicitTerms("z"));
    }
}
}
