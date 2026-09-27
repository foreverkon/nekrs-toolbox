#include "diagnostics.hpp"
#include "common.hpp"
#include "mlog.hpp"
#include "linAlg.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace diagnostics {

namespace {
    toolbox::Schedule schedule;
    std::vector<Metric> metrics;
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    std::string output = "diagnostics.csv";
    std::ofstream csv;
    double lastWritten = -std::numeric_limits<double>::infinity();
    bool ready = false;

    std::string header()
    {
        std::string result = "time,step";
        for (const auto& metric : metrics) {
            result += ',';
            if (metric.name.find_first_of(",\"") == std::string::npos) {
                result += metric.name;
            } else {
                result += '"';
                for (const char character : metric.name) {
                    result += character;
                    if (character == '"') {
                        result += '"';
                    }
                }
                result += '"';
            }
        }
        return result;
    }

    void open_csv(double restartTime)
    {
        const auto expectedHeader = header();
        std::vector<std::string> retained;
        if (std::filesystem::exists(output)) {
            std::ifstream previous(output);
            if (!previous) {
                throw std::runtime_error("cannot read existing CSV");
            }

            std::string line;
            if (std::getline(previous, line) && line != expectedHeader) {
                throw std::runtime_error("CSV schema mismatch; choose another output or restore registered metrics");
            }
            double previousTime = -std::numeric_limits<double>::infinity();
            while (std::getline(previous, line)) {
                std::istringstream row(line);
                row.imbue(std::locale::classic());
                double rowTime;
                char delimiter;
                if (!(row >> rowTime >> delimiter) || delimiter != ',' || !std::isfinite(rowTime)
                    || rowTime <= previousTime) {
                    throw std::runtime_error("CSV contains an invalid or non-increasing time");
                }
                int rowStep;
                if (!(row >> rowStep >> delimiter) || delimiter != ',') {
                    throw std::runtime_error("CSV row requires an integer step followed by metric values");
                }
                std::string value;
                std::size_t count = 0;
                while (std::getline(row, value, ',')) {
                    std::size_t parsed;
                    std::stod(value, &parsed);
                    if (parsed != value.size()) {
                        throw std::runtime_error("CSV metric value must be numeric");
                    }
                    ++count;
                }
                if (count != metrics.size() || line.back() == ',') {
                    throw std::runtime_error("CSV row requires " + std::to_string(metrics.size()) + " metric values");
                }
                previousTime = rowTime;
                if (rowTime <= restartTime) {
                    retained.push_back(line);
                    lastWritten = rowTime;
                }
            }
            if (previous.bad()) {
                throw std::runtime_error("cannot read existing CSV");
            }
        }

        csv.clear();
        csv.open(output, std::ios::out | std::ios::trunc);
        csv.imbue(std::locale::classic());
        csv << std::setprecision(std::max(std::numeric_limits<double>::max_digits10,
            std::numeric_limits<dfloat>::max_digits10));
        csv << expectedHeader << '\n';
        for (const auto& row : retained) {
            csv << row << '\n';
        }
        csv.flush();
        if (!csv) {
            throw std::runtime_error("cannot write CSV");
        }
    }
}

void setup0(MPI_Comm comm, setupAide&)
{
    communicator = comm;
    MPI_Comm_rank(communicator, &rank);
    schedule = toolbox::Schedule {};
    schedule.read("diagnostics");
    output = "diagnostics.csv";
    platform->par->extract("diagnostics", "output", output);
}

void set_fields(std::vector<Metric> registeredMetrics)
{
    toolbox::check(ready, "diagnostics: set_fields must precede setup");
    metrics = std::move(registeredMetrics);
}

void setup(double restartTime)
{
    schedule.reset();
    if (!schedule.enabled) {
        return;
    }

    toolbox::check(metrics.empty(), "diagnostics: enabled plugin requires registered metrics");
    toolbox::check(output.empty(), "diagnostics: output must name a CSV file");
    std::set<std::string> names;
    for (const auto& metric : metrics) {
        toolbox::check(metric.name.empty() || metric.name.find_first_of(" \t\r\n\f\v=") != std::string::npos
                || !metric.evaluate,
            "diagnostics: each metric requires a name without whitespace or '=' and an evaluator");
        toolbox::check(!names.insert(metric.name).second,
            "diagnostics: duplicate metric name: " + metric.name);
    }

    lastWritten = -std::numeric_limits<double>::infinity();
    std::string error;
    if (rank == 0) {
        try {
            open_csv(restartTime);
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    }
    toolbox::check(!error.empty(), "diagnostics: " + output + ": " + error);
    MPI_Bcast(&lastWritten, 1, MPI_DOUBLE, 0, communicator);
    if (std::isfinite(lastWritten)) {
        schedule.mark(lastWritten, 0);
    }
    ready = true;
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
    return ready && time > lastWritten && schedule.due(time, step);
}

void run(double time, int step)
{
    if (!needs_sample(time, step)) {
        return;
    }

    std::vector<std::pair<std::string, double>> values;
    values.reserve(metrics.size());
    for (const auto& metric : metrics) {
        values.emplace_back(metric.name, metric.evaluate());
    }
    if (rank == 0) {
        csv << time << ',' << step;
        for (const auto& value : values) {
            csv << ',' << value.second;
        }
        csv << '\n';
        csv.flush();
        mlog::metric(step, "diagnostics", values);
    }
    toolbox::check(rank == 0 && !csv, "diagnostics: cannot write " + output);
    lastWritten = time;
    schedule.mark(time, step);
}

void finalize()
{
    if (ready) {
        if (rank == 0) {
            csv.close();
        }
        toolbox::check(rank == 0 && csv.fail(), "diagnostics: cannot close " + output);
    }
    ready = false;
    metrics.clear();
}

dfloat volume_mean(mesh_t* mesh, const deviceMemory<dfloat>& value)
{
    return platform->linAlg->innerProd(mesh->Nlocal, value, mesh->o_Jw,
               platform->comm.mpiComm())
        / mesh->volume;
}

dfloat weighted_mean(mesh_t* mesh,
    const deviceMemory<dfloat>& value,
    const deviceMemory<dfloat>& weight)
{
    const auto denominator = platform->linAlg->innerProd(mesh->Nlocal, weight,
        mesh->o_Jw, platform->comm.mpiComm());
    toolbox::check(denominator == 0, "diagnostics: weighted_mean has zero integrated weight");
    return platform->linAlg->weightedInnerProd(mesh->Nlocal, mesh->o_Jw, value,
               weight, platform->comm.mpiComm())
        / denominator;
}

}
