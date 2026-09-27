#include "statistics.hpp"
#include "common.hpp"

#include <memory>

namespace statistics {

namespace {
    toolbox::Schedule schedule;
    mesh_t* fieldMesh = nullptr;
    dlong fieldOffset = 0;
    std::vector<tavg::field> fields;
    std::unique_ptr<tavg> avg;
}

void setup0(MPI_Comm, setupAide&)
{
    schedule = toolbox::Schedule {};
    schedule.read("statistics");
}

void set_fields(mesh_t* mesh, dlong offset, const std::vector<tavg::field>& registeredFields)
{
    toolbox::check(static_cast<bool>(avg), "statistics: set_fields must precede setup");
    fieldMesh = mesh;
    fieldOffset = offset;
    fields = registeredFields;
}

void setup(double)
{
    schedule.reset();
    if (!schedule.enabled) {
        return;
    }

    toolbox::check(fields.empty(), "statistics: enabled plugin requires registered fields");
    toolbox::check(!fieldMesh || fieldOffset < fieldMesh->Nlocal || fieldOffset <= 0,
        "statistics: registered mesh and field offset are invalid");
    avg = std::make_unique<tavg>(fieldOffset, fields);
}

bool enabled()
{
    return schedule.enabled;
}

bool automatic()
{
    return schedule.automatic;
}

bool needs_sample(double time, int step)
{
    return avg && schedule.due(time, step);
}

void run(double time, int step)
{
    if (!needs_sample(time, step)) {
        return;
    }

    avg->run(time);
    schedule.mark(time, step);
}

void write()
{
    if (avg && avg->time() > 0) {
        avg->writeToFile(fieldMesh, true);
    }
}

void finalize()
{
    write();
    avg.reset();
    fields.clear();
    fieldMesh = nullptr;
    fieldOffset = 0;
}

}
