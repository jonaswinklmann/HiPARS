#include "sort.hpp"

enum class Direction {
    HOR = 0,
    VER = 1,
    NONE = 2,
    DIAG = 3
};

typedef struct Move {
    std::vector<std::pair<double,double>> sites_list;
    double distance;
    Direction init_dir;
} Move;

std::optional<std::vector<Move>> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);

bool sortSequentiallyByRowCA(std::vector<Move>& ml, size_t rows, size_t cols, bool** stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger);
