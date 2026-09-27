#ifndef __okl__
#include "toolbox.hpp"

namespace {
    deviceMemory<dfloat> speedSquared;
}

void UDF_Setup0(MPI_Comm comm, setupAide& options)
{
    toolbox::setup0(comm, options);
}

void UDF_LoadKernels(deviceKernelProperties& kernelInfo)
{
    toolbox::load_kernels(kernelInfo);
}

void UDF_Setup()
{
    auto mesh = nrs->meshV;
    const auto offset = nrs->fieldOffset;
    const auto velocityX = nrs->fluid->o_solution("x");
    const auto velocityY = nrs->fluid->o_solution("y");
    const auto velocityZ = nrs->fluid->o_solution("z");
    const deviceMemory<dfloat> pressure(nrs->fluid->o_P);

    monitor::set_fields(mesh, offset, {
        {"u", velocityX}, {"v", velocityY}, {"w", velocityZ}, {"p", pressure}
    });
    statistics::set_fields(mesh, offset, {
        {"u", {velocityX}}, {"v", {velocityY}}, {"w", {velocityZ}}, {"p", {pressure}},
        {"uu", {velocityX, velocityX}}, {"vv", {velocityY, velocityY}},
        {"ww", {velocityZ, velocityZ}}, {"uv", {velocityX, velocityY}},
        {"uw", {velocityX, velocityZ}}, {"vw", {velocityY, velocityZ}}
    });
    if (diagnostics::enabled()) {
        speedSquared.resize(offset);
        diagnostics::set_fields({
            {"mean_u", [mesh, velocityX]() { return diagnostics::volume_mean(mesh, velocityX); }},
            {"kinetic_energy", [mesh]() { return 0.5 * diagnostics::volume_mean(mesh, speedSquared); }}
        });
    }
    toolbox::setup();
}

void UDF_ExecuteStep(double time, int tstep)
{
    if (diagnostics::needs_sample(time, tstep)) {
        const auto velocityX = nrs->fluid->o_solution("x");
        const auto velocityY = nrs->fluid->o_solution("y");
        const auto velocityZ = nrs->fluid->o_solution("z");
        alg::vec_dot(nrs->meshV->Nlocal, velocityX, velocityY, velocityZ,
            velocityX, velocityY, velocityZ, speedSquared);
    }
    toolbox::run(time, tstep);
}

void UDF_Finalize()
{
    toolbox::finalize();
    speedSquared.clear();
}
#endif
