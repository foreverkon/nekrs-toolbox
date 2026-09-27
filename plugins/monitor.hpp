#pragma once

#include "common.hpp"

#include <string>
#include <vector>

namespace monitor {

void setup0(MPI_Comm comm, setupAide& options);
struct Field {
    std::string name;
    deviceMemory<dfloat> values;
};

void set_fields(mesh_t* mesh, dlong fieldOffset, const std::vector<Field>& fields);
void setup(double restartTime);
bool enabled();
bool automatic();
bool needs_sample(double time, int tstep);
void run(double time, int tstep);
void flush();
void finalize();

}
