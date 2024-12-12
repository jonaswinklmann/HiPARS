#include "sort.hpp"

#define AOD_TOTAL_LIMIT 256
#define AOD_ROW_LIMIT 16
#define AOD_COL_LIMIT 16

#define MOVE_COST_OFFSET 10
#define MOVE_COST_OFFSET_SUBMOVE 0
#define MOVE_COST_SCALING_SQRT 0
#define MOVE_COST_SCALING_LINEAR 1

#define MAX_MULTI_ITER_COUNT 10000

#define ALLOW_MOVES_BETWEEN_ROWS true
#define ALLOW_MOVES_BETWEEN_COLS true
#define ALLOW_MULTIPLE_MOVES_PER_ATOM false

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
    static ParallelMove fromStartAndEnd(ParallelMove::Step start, ParallelMove::Step end, 
        std::shared_ptr<spdlog::logger> logger);
    double cost();
    bool execute(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, std::shared_ptr<spdlog::logger> logger,
        std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>);
};

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);