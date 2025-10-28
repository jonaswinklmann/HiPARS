#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "sort.hpp"

#define MAX_MULTI_ITER_COUNT 100

#define M_4TH_ROOT_2 1.1892071150027210667
#define M_4TH_ROOT_1_2 1 / M_4TH_ROOT_2
/*#define HALF_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_SQRT1_2 + MOVE_COST_SCALING_LINEAR / 2)
#define HALF_DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_1_2 + MOVE_COST_SCALING_LINEAR * M_SQRT1_2)
#define DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_2 + MOVE_COST_SCALING_LINEAR * M_SQRT2)*/

#define NUM_THREADS 8

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
    static ParallelMove fromStartAndEnd(
        ArrayAccessor& stateArray, ParallelMove::Step start, ParallelMove::Step end, std::shared_ptr<spdlog::logger> logger);
    double cost();
    bool execute(ArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger,
        std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved = std::nullopt) const;
};

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);