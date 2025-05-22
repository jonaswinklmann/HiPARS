#include "sort.hpp"

typedef struct Move {
    std::vector<std::pair<double,double>> sites_list;
    double distance;
} Move;

std::optional<std::vector<Move>> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd);

bool sortSequentiallyByRowC1D(std::vector<Move>& ml, size_t rows, size_t cols, bool* stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger);
bool sortSequentiallyByRowC2D(std::vector<Move>& ml, size_t rows, size_t cols, bool** stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger);