#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

struct Move
{
    unsigned int x, y;
    int xDir, yDir;
};

std::vector<Move> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    Eigen::Array2i filledShape);
