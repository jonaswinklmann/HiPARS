#pragma once

#include "sortParallel.hpp"

// 0 means never, 1 always, 2 every second, ...
#define CHECK_FOR_DIRECT_REMOVAL_EVERY_X_MOVES 20

#define VALUE_FILLED_DESIRED 1.
#define VALUE_USED_UNDESIRED 0.5
#define VALUE_CLEARED_UNDESIRED 0.2
#define VALUE_CLEARED_UNDESIRED_UNUSABLE 0.8
#define VALUE_CLEARED_OUTSIDE_UNUSABLE 0.1

std::optional<std::vector<ParallelMove>> sortLatticeGreedyParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& targetGeometry);

std::optional<std::vector<ParallelMove>> sortLatticeByRowParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& targetGeometry);

Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> generateMask(double distance, double spacingFraction = 1);