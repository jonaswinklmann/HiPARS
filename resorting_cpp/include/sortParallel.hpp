#define AOD_TOTAL_LIMIT 5
#define AOD_ROW_LIMIT 16
#define AOD_COL_LIMIT 16

#define MOVE_COST_OFFSET 3
#define MOVE_COST_OFFSET_SUBMOVE 0
#define MOVE_COST_SCALING_SQRT 0
#define MOVE_COST_SCALING_LINEAR 1

#define ALLOW_MOVES_BETWEEN_ROWS true
#define ALLOW_MOVES_BETWEEN_COLS true

#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"
#include "spdlog/spdlog.h"

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

class ParallelMove
{
    public:
        struct Step
        {
            std::vector<double> colSelection;
            std::vector<double> rowSelection;
        };
        std::vector<Step> steps;
        ParallelMove() : steps() {};
        double cost();
        bool execute(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, std::shared_ptr<spdlog::logger> logger);
};

std::vector<ParallelMove> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);