#include "alg.hpp"
#include "utils.hpp"
#include "common.hpp"

namespace alg {

void load_kernels(deviceKernelProperties& kernelInfo)
{
    platform->kernelRequests.add("alg::vec_dot", toolbox::kernels_dir() + "vec_dot.okl", kernelInfo);
}

void vec_dot(const dlong N,
    const occa::memory& a1,
    const occa::memory& a2,
    const occa::memory& a3,
    const occa::memory& b1,
    const occa::memory& b2,
    const occa::memory& b3,
    occa::memory& z)
{
    LaunchKernel("alg::vec_dot", N, a1, a2, a3, b1, b2, b3, z);
}

}
