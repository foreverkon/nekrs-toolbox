#pragma once

#include "nrs.hpp"
#include "tavg.hpp"

namespace statistics {

void setup0(MPI_Comm comm, setupAide& options);

/**
 * Register fields collectively before setup. The UDF owns their device storage
 * and updates derived fields before each due run. Including this header makes
 * tavg.hpp visible to nekRS UDF plugin autoload.
 */
void set_fields(mesh_t* mesh, dlong fieldOffset, const std::vector<tavg::field>& fields);

/**
 * Start a fresh averaging window on the first due sample when enabled.
 * Compression is configured in adios.yaml.
 */
void setup(double restartTime);
void run(double time, int step);
void write();

/** Flush an accumulated partial window and release it. */
void finalize();

bool enabled();
bool automatic();

/** Query before updating UDF-derived fields. */
bool needs_sample(double time, int step);

}
