#include "sort.hpp"

#define AOD_TOTAL_LIMIT 256
#define AOD_ROW_LIMIT 16
#define AOD_COL_LIMIT 16

#define MOVE_COST_OFFSET 150
#define MOVE_COST_OFFSET_SUBMOVE 0
#define MOVE_COST_SCALING_SQRT 0
#define MOVE_COST_SCALING_LINEAR 0

#define MAX_MULTI_ITER_COUNT 100

#define M_4TH_ROOT_2 1.1892071150027210667
#define M_4TH_ROOT_1_2 1 / M_4TH_ROOT_2
#define HALF_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_SQRT1_2 + MOVE_COST_SCALING_LINEAR / 2)
#define HALF_DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_1_2 + MOVE_COST_SCALING_LINEAR * M_SQRT1_2)
#define DIAG_STEP_COST (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_SQRT * M_4TH_ROOT_2 + MOVE_COST_SCALING_LINEAR * M_SQRT2)

#define ALLOW_MOVES_BETWEEN_ROWS true
#define ALLOW_MOVES_BETWEEN_COLS true
#define ALLOW_MULTIPLE_MOVES_PER_ATOM false

#define ALLOW_MOVING_EMPTY_TRAP_ONTO_OCCUPIED false

#define ALLOW_DIAGONAL_MOVEMENT false

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
        StateArrayAccessor& stateArray,
        ParallelMove::Step start, ParallelMove::Step end, std::shared_ptr<spdlog::logger> logger);
    double cost();
    bool execute(StateArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger,
        std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved = std::nullopt);
};

double inline costPerSubMove(double dist)
{
    return dist > DOUBLE_EQUIVALENCE_THRESHOLD ? (MOVE_COST_OFFSET_SUBMOVE + MOVE_COST_SCALING_LINEAR * dist + (MOVE_COST_SCALING_SQRT != 0 ? MOVE_COST_SCALING_SQRT * sqrt(dist) : 0)) : 0;
}

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);

std::optional<std::vector<ParallelMove>> sortLatticeGeometriesParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& targetGeometry);