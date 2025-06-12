#pragma once

#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"
#include "spdlog/spdlog.h"
#include <stdexcept>
#include <sstream>

#define DOUBLE_EQUIVALENCE_THRESHOLD 0.00001

namespace py = pybind11;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

class StateArrayAccessor
{
public:
    virtual std::unique_ptr<StateArrayAccessor> copy() const = 0;
    virtual void operator=(const StateArrayAccessor& other) = 0;
    virtual const bool& operator()(std::size_t row, std::size_t col) const = 0;
    virtual bool& operator()(std::size_t row, std::size_t col) = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
};

class RowBitMask
{
public:
    std::vector<size_t> indices;
    size_t count;
private:
    uint64_t *bitMask;
public:
    RowBitMask() : indices(), count(0), bitMask(nullptr) {};
    RowBitMask(size_t count, std::optional<size_t> index = std::nullopt) : indices(), 
        count(count), bitMask(new uint64_t[this->count / 64 + 1]()) 
    {
        if(index.has_value())
        {
            this->indices.push_back(index.value());
        }
    }
    RowBitMask(std::vector<size_t> indices, size_t count, uint64_t *bitMask) : indices(indices), 
        count(count), bitMask(bitMask) {}
    RowBitMask(RowBitMask&& other) : indices(std::exchange(other.indices, std::vector<size_t>())), 
        count(std::exchange(other.count, 0)), 
        bitMask(std::exchange(other.bitMask, nullptr)) {}
    RowBitMask(const RowBitMask& other) : indices(other.indices), count(other.count), 
        bitMask(new uint64_t[this->count / 64 + 1]()) 
    {
        for(size_t i = 0; i <= this->count / 64; i++)
        {
            this->bitMask[i] = other.bitMask[i];
        }
    }
    ~RowBitMask()
    {
        delete[] this->bitMask;
    }
    static RowBitMask fromOr(const RowBitMask& l, const RowBitMask& r)
    {
        size_t c = l.count < r.count ? l.count : r.count;
        uint64_t *bitMask = new uint64_t[c / 64 + 1];
        for(size_t i = 0; i <= c / 64; i++)
        {
            bitMask[i] = l.bitMask[i] | r.bitMask[i];
        }
        std::vector<size_t> indices;
        indices.insert(indices.end(), l.indices.begin(), l.indices.end());
        indices.insert(indices.end(), r.indices.begin(), r.indices.end());
        return RowBitMask(indices, c, bitMask);
    }
    static RowBitMask fromAnd(const RowBitMask& l, const RowBitMask& r)
    {
        size_t c = l.count < r.count ? l.count : r.count;
        uint64_t *bitMask = new uint64_t[c / 64 + 1];
        for(size_t i = 0; i <= c / 64; i++)
        {
            bitMask[i] = l.bitMask[i] & r.bitMask[i];
        }
        std::vector<size_t> indices;
        indices.insert(indices.end(), l.indices.begin(), l.indices.end());
        indices.insert(indices.end(), r.indices.begin(), r.indices.end());
        return RowBitMask(indices, c, bitMask);
    }
    void set(size_t index, bool value)
    {
        if(index < this->count)
        {
            if(value)
            {
                this->bitMask[index / 64] |= UINT64_C(1) << (index % 64);
            }
            else
            {
                this->bitMask[index / 64] &= ~(UINT64_C(1) << (index % 64));
            }
        }
        else
        {
            std::stringstream msg;
            msg << "BitMask index " << index << " out of bounds (size: " << this->count << ") in set";
            throw std::invalid_argument(msg.str());
        }
    }
    bool operator[](size_t index)
    {
        if(index < this->count)
        {
            return this->bitMask[index / 64] & (UINT64_C(1) << (index % 64));
        }
        else
        {
            std::stringstream msg;
            msg << "BitMask index " << index << " out of bounds (size: " << this->count << ") in operator[]";
            throw std::invalid_argument(msg.str());
        }
    }
    RowBitMask& operator|=(const RowBitMask& other)
    {
        size_t c = count < other.count ? count : other.count;
        for(size_t i = 0; i <= c / 64; i++)
        {
            this->bitMask[i] = this->bitMask[i] | other.bitMask[i];
        }
        this->indices.insert(this->indices.end(), other.indices.begin(), other.indices.end());
        return *this;
    }
    RowBitMask& operator&=(const RowBitMask& other)
    {
        size_t c = count < other.count ? count : other.count;
        for(size_t i = 0; i <= c / 64; i++)
        {
            this->bitMask[i] = this->bitMask[i] & other.bitMask[i];
        }
        this->indices.insert(this->indices.end(), other.indices.begin(), other.indices.end());
        return *this;
    }
    RowBitMask operator=(const RowBitMask&) = delete;
    RowBitMask& operator=(RowBitMask&& other)
    {
        delete[] this->bitMask;
        this->bitMask = std::exchange(other.bitMask, nullptr);
        this->count = std::exchange(other.count, 0);
        this->indices = std::exchange(other.indices, std::vector<size_t>());

        return *this;
    };
    unsigned int bitsSet()
    {
        unsigned int popCount = 0;
        for(size_t i = 0; i <= this->count / 64; i++)
        {
            popCount += std::__popcount(this->bitMask[i]);
        }
        return popCount;
    }
};

class EigenArrayStateArrayAccessor : public StateArrayAccessor
{
private:
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> stateArray;
public:
    EigenArrayStateArrayAccessor(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> array) : stateArray(array) {}
    std::unique_ptr<StateArrayAccessor> copy() const
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> arrayCopy = stateArray;
        return std::unique_ptr<EigenArrayStateArrayAccessor>(new EigenArrayStateArrayAccessor(arrayCopy));
    }
    void operator=(const StateArrayAccessor& other)
    {
        if(this->rows() != other.rows() || this->cols() != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->stateArray(r,c) = other(r,c);
                }
            }
        }
    }
    const bool& operator()(std::size_t row, std::size_t col) const
    {
        return this->stateArray(row,col);
    }
    bool& operator()(std::size_t row, std::size_t col)
    {
        return this->stateArray(row,col);
    }
    std::size_t rows() const
    {
        return this->stateArray.rows();
    }
    std::size_t cols() const
    {
        return this->stateArray.cols();
    }
};

class CStyle2DStateArrayAccessor : public StateArrayAccessor
{
private:
    const bool dataOwned;
    bool **stateArray;
    std::size_t rowCount, colCount;
    CStyle2DStateArrayAccessor(const CStyle2DStateArrayAccessor& other) : dataOwned(true), stateArray(nullptr), rowCount(other.rowCount), colCount(other.colCount)
    {
        stateArray = new bool*[rowCount];
        for(size_t i = 0; i < rowCount; i++)
        {
            stateArray[i] = new bool[colCount];
        }
    }
public:
    CStyle2DStateArrayAccessor(bool **array, std::size_t rows, std::size_t cols) : dataOwned(false), stateArray(array), rowCount(rows), colCount(cols) {}
    std::unique_ptr<StateArrayAccessor> copy() const
    {
        CStyle2DStateArrayAccessor *accessorCopy = new CStyle2DStateArrayAccessor(*this);
        return std::unique_ptr<CStyle2DStateArrayAccessor>(accessorCopy);
    }
    ~CStyle2DStateArrayAccessor()
    {
        if(dataOwned)
        {
            for(size_t i = 0; i < rowCount; i++)
            {
                delete[] stateArray[i];
            }
            delete[] stateArray;
        }
    }
    void operator=(const StateArrayAccessor& other)
    {
        if(this->rowCount != other.rows() || this->colCount != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->stateArray[r][c] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(std::size_t row, std::size_t col) const
    {
        return this->stateArray[row][col];
    }
    bool& operator()(std::size_t row, std::size_t col)
    {
        return this->stateArray[row][col];
    }
    std::size_t rows() const
    {
        return this->rowCount;
    }
    std::size_t cols() const
    {
        return this->colCount;
    }
};

class CStyle1DStateArrayAccessor : public StateArrayAccessor
{
private:
    const bool dataOwned;
    bool *stateArray;
    std::size_t rowCount, colCount;
    CStyle1DStateArrayAccessor(const CStyle1DStateArrayAccessor& other) : dataOwned(true), stateArray(nullptr), rowCount(other.rowCount), colCount(other.colCount)
    {
        stateArray = new bool[rowCount * colCount];
    }
public:
    CStyle1DStateArrayAccessor(bool *array, std::size_t rows, std::size_t cols) : dataOwned(false), stateArray(array), rowCount(rows), colCount(cols) {}
    std::unique_ptr<StateArrayAccessor> copy() const
    {
        CStyle1DStateArrayAccessor *accessorCopy = new CStyle1DStateArrayAccessor(*this);
        return std::unique_ptr<CStyle1DStateArrayAccessor>(accessorCopy);
    }
    ~CStyle1DStateArrayAccessor()
    {
        if(dataOwned)
        {
            delete[] stateArray;
        }
    }
    void operator=(const StateArrayAccessor& other)
    {
        if(this->rowCount != other.rows() || this->colCount != other.cols())
        {
            throw std::invalid_argument("Array cannot be assigned due to different sizes");
        }
        else
        {
            for(size_t r = 0; r < this->rows(); r++)
            {
                for(size_t c = 0; c < this->cols(); c++)
                {
                    this->stateArray[r * this->colCount + c] = other(r,c);
                }
            }
        }
    }
    const bool& operator()(std::size_t row, std::size_t col) const
    {
        return this->stateArray[row * this->colCount + col];
    }
    bool& operator()(std::size_t row, std::size_t col)
    {
        return this->stateArray[row * this->colCount + col];
    }
    std::size_t rows() const
    {
        return this->rowCount;
    }
    std::size_t cols() const
    {
        return this->colCount;
    }
};