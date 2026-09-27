#pragma once

#include "nrs.hpp"

#include <functional>
#include <string>
#include <vector>

namespace diagnostics {

/** Evaluated on every rank in registration order; reductions belong to the UDF. */
struct Metric {
    std::string name;
    std::function<dfloat()> evaluate;
};

void setup0(MPI_Comm comm, setupAide& options);

/**
 * Register the same ordered metrics collectively before setup. Names must be
 * unique, nonempty, without whitespace or '='. The UDF keeps captured fields alive and current.
 */
void set_fields(std::vector<Metric> metrics);

/**
 * [diagnostics] output defaults to diagnostics.csv. Rank zero writes columns
 * time,step followed by metric names, with round-trip scalar precision.
 * Existing CSV schemas must match; rows newer than restartTime are discarded
 * and retained times are not sampled again. All lifecycle calls are collective.
 */
void setup(double restartTime);
void run(double time, int step);

/** Close the CSV and release callbacks. */
void finalize();

bool enabled();
bool automatic();
bool needs_sample(double time, int step);

/** Collective volume mean using the supplied mesh's current quadrature weights. */
dfloat volume_mean(mesh_t* mesh, const deviceMemory<dfloat>& value);

/**
 * Collective integral(value * weight) / integral(weight), using current UDF
 * coefficients and mesh quadrature weights. A zero denominator is an error.
 */
dfloat weighted_mean(mesh_t* mesh,
    const deviceMemory<dfloat>& value,
    const deviceMemory<dfloat>& weight);

}
