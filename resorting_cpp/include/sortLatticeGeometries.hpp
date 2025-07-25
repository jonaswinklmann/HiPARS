#include "sort.hpp"

#define RECOMMENDED_DIST_FROM_OCC_SITES 1
#define RECOMMENDED_DIST_FROM_EMPTY_SITES 0.1
#define MIN_DIST_FROM_OCC_SITES 1
#define MAX_SUBMOVE_DIST_IN_PENALIZED_AREA 1.5
#define COL_SPACING 0.5
#define ROW_SPACING 1
#define HALF_COL_SPACING ((double)(COL_SPACING) / 2)
#define HALF_ROW_SPACING ((double)(ROW_SPACING) / 2)

// 0 means never, 1 always, 2 every second, ...
#define CHECK_FOR_DIRECT_REMOVAL_EVERY_X_MOVES 20

#define VALUE_FILLED_DESIRED 1.
#define VALUE_USED_UNDESIRED 0.5
#define VALUE_CLEARED_UNDESIRED 0.2
#define VALUE_CLEARED_UNDESIRED_UNUSABLE 0.8
#define VALUE_CLEARED_OUTSIDE_UNUSABLE 0.1


Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> generateMask(double distance, double spacingFraction = 1);