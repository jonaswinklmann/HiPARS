#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"
#include "spdlog/spdlog.h"

#ifndef sort_hpp_included
#define sort_hpp_included

#define DOUBLE_EQUIVALENCE_THRESHOLD 0.00001

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

class StateArrayAccessor
{
public:
    virtual bool& operator()(std::size_t row, std::size_t col) = 0;
    virtual std::size_t rows() = 0;
    virtual std::size_t cols() = 0;
};

class EigenArrayStateArrayAccessor : public StateArrayAccessor
{
private:
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> stateArray;
public:
    EigenArrayStateArrayAccessor(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> array) : stateArray(array) {}
    bool& operator()(std::size_t row, std::size_t col)
    {
        return this->stateArray(row,col);
    }
    std::size_t rows()
    {
        return this->stateArray.rows();
    }
    std::size_t cols()
    {
        return this->stateArray.cols();
    }
};

class CStyleStateArrayAccessor : public StateArrayAccessor
{
private:
    bool **stateArray;
    std::size_t rowCount, colCount;
public:
    CStyleStateArrayAccessor(bool **array, std::size_t rows, std::size_t cols) : stateArray(array), rowCount(rows), colCount(cols) {}
    bool& operator()(std::size_t row, std::size_t col)
    {
        return this->stateArray[row][col];
    }
    std::size_t rows()
    {
        return this->rowCount;
    }
    std::size_t cols()
    {
        return this->colCount;
    }
};
#endif