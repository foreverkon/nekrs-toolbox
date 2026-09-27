#pragma once

#include "nrs.hpp"
#include "udf.hpp"

namespace alg {

void load_kernels(deviceKernelProperties& kernelInfo);

void vec_dot(const dlong N,
    const occa::memory& a1,
    const occa::memory& a2,
    const occa::memory& a3,
    const occa::memory& b1,
    const occa::memory& b2,
    const occa::memory& b3,
    occa::memory& z);

}
