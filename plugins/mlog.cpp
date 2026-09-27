#include "mlog.hpp"

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

namespace mlog {
namespace {
    bool should_output = false;
}
void setup0(MPI_Comm comm, setupAide& options)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    should_output = (rank == 0);
}

void printf(const char* format, ...)
{
    if (!should_output)
        return;
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}

void metric(int step, const std::string& group,
    const std::vector<std::pair<std::string, double>>& values)
{
    if (!should_output)
        return;
    std::ostringstream line;
    line.imbue(std::locale::classic());
    line << "METRIC step=" << step << " group=" << group
         << std::scientific << std::setprecision(8);
    for (const auto& [name, value] : values) {
        line << ' ' << name << '=' << value;
    }
    line << '\n';
    std::fputs(line.str().c_str(), stdout);
    std::fflush(stdout);
}

std::ostream& cout()
{
    if (should_output) {
        return std::cout;
    } else {
        // https://stackoverflow.com/a/46455079
        static std::ofstream ofs;
        return ofs;
    }
}

}
