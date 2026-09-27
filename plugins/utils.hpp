#pragma once

#include "mlog.hpp"

#define LaunchKernel(name, ...)                           \
    do {                                                  \
        static occa::kernel kernel;                       \
        if (!kernel.isInitialized())                      \
            kernel = platform->kernelRequests.load(name); \
        kernel(__VA_ARGS__);                              \
    } while (0)
