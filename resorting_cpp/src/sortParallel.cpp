#include "sortParallel.hpp"

#include <optional>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "config.hpp"
#include "spdlog/sinks/basic_file_sink.h"

bool orderedDblVecContainsElem(const std::vector<double>& vec, double elem)
{
    const auto& firstOccurrence = std::lower_bound(vec.begin(), vec.end(), elem - DOUBLE_EQUIVALENCE_THRESHOLD);
    if(firstOccurrence == vec.end())
    {
        return false;
    }
    else
    {
        return abs(*firstOccurrence - elem) < DOUBLE_EQUIVALENCE_THRESHOLD;
    }
}

size_t roundCoordDown(double coord)
{
    return (size_t)(coord + 0.25);
}

bool& accessStateArray(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray,
    size_t i, size_t j, bool rowFirst)
{
    if(rowFirst)
    {
        return stateArray(i,j);
    }
    else
    {
        return stateArray(j,i);
    }
}

void fillDimensionDependantData(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], bool rowFirst, size_t outerDimCompZone[2], size_t innerDimCompZone[2],
    size_t& outerSize, size_t& innerSize, unsigned int& outerAODLimit, unsigned int& innerAODLimit)
{
    if(rowFirst)
    {
        outerDimCompZone[0] = compZone[0];
        outerDimCompZone[1] = compZone[1];
        innerDimCompZone[0] = compZone[2];
        innerDimCompZone[1] = compZone[3];
        outerSize = stateArray.rows();
        innerSize = stateArray.cols();
        outerAODLimit = AOD_ROW_LIMIT;
        innerAODLimit = AOD_COL_LIMIT;
    }
    else
    {
        outerDimCompZone[0] = compZone[2];
        outerDimCompZone[1] = compZone[3];
        innerDimCompZone[0] = compZone[0];
        innerDimCompZone[1] = compZone[1];
        outerSize = stateArray.cols();
        innerSize = stateArray.rows();
        outerAODLimit = AOD_COL_LIMIT;
        innerAODLimit = AOD_ROW_LIMIT;
    }
}

double approxCostPerMove(double dist1, double dist2)
{
    if(ALLOW_DIAGONAL_MOVEMENT)
    {
        if(dist1 <= 1 + DOUBLE_EQUIVALENCE_THRESHOLD && dist2 <= 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            if(dist1 >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD && dist2 >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                return MOVE_COST_OFFSET + DIAG_STEP_COST;
            }
            else
            {
                // If not both are 1 then at least one has to be 0 so dist1+dist2 = sqrt(dist1^2 + dist2^2)
                return MOVE_COST_OFFSET + costPerSubMove(dist1 + dist2);
            }
        }
        else
        {
            double cost = MOVE_COST_OFFSET + 2 * HALF_DIAG_STEP_COST;
            if(dist1 > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                cost += costPerSubMove(dist1 - 1);
            }
            if(dist2 > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                cost += costPerSubMove(dist2 - 1);
            }
            return cost;
        }
    }
    else
    {
        if(dist1 < 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            return MOVE_COST_OFFSET + 2 * HALF_STEP_COST + costPerSubMove(dist2);
        }
        else if(dist2 < 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            return MOVE_COST_OFFSET + 2 * HALF_STEP_COST + costPerSubMove(dist1);
        }
        else
        {
            return MOVE_COST_OFFSET + 2 * HALF_STEP_COST + costPerSubMove(dist1 - 0.5) + costPerSubMove(dist2 - 0.5);
        }
    }
}

bool fillSteps(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    ParallelMove::Step& start, ParallelMove::Step& end, ParallelMove::Step& step1, ParallelMove::Step& step2, 
    bool rowFirst, double& maxDist, std::shared_ptr<spdlog::logger> logger)
{
    bool completelyDirectMove = true;
    std::vector<double> *step1Selection, *step2Selection, *selStart, *selEnd, *otherStart, *otherEnd;
    if(rowFirst)
    {
        step1Selection = &step1.rowSelection;
        step2Selection = &step2.rowSelection;
        selStart = &start.rowSelection;
        selEnd = &end.rowSelection;
        otherStart = &start.colSelection;
        otherEnd = &end.colSelection;
    }
    else
    {
        step1Selection = &step1.colSelection;
        step2Selection = &step2.colSelection;
        selStart = &start.colSelection;
        selEnd = &end.colSelection;
        otherStart = &start.rowSelection;
        otherEnd = &end.rowSelection;
    }
    for(size_t i = 0; i < selStart->size(); i++)
    {
        double startRow = (*selStart)[i];
        double endRow = (*selEnd)[i];
        if(endRow > startRow + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            completelyDirectMove = false;
            (*step1Selection)[i] += 0.5;
            (*step2Selection)[i] -= 0.5;
        }
        else if(endRow < startRow - DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            completelyDirectMove = false;
            (*step1Selection)[i] -= 0.5;
            (*step2Selection)[i] += 0.5;
        }
        else
        {
            bool directMove = true;

            for(size_t j = 0; j < otherStart->size() && directMove; j++)
            {
                size_t row = roundCoordDown(startRow);
                double col = (*otherStart)[j];
                if(accessStateArray(stateArray, row, roundCoordDown(col), rowFirst))
                {
                    double endCol = (*otherEnd)[j];
                    int colStep = abs(endCol - col) > DOUBLE_EQUIVALENCE_THRESHOLD ? (signbit(endCol - col) ? -1 : 1) : 0;
                    col += colStep;

                    // If either has at least one more step to go (the other will be zero)
                    for(; (endCol - col) * colStep >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD; col += colStep)
                    {
                        if(accessStateArray(stateArray, row, roundCoordDown(col), rowFirst) && 
                            !orderedDblVecContainsElem(*otherStart, col))
                        {
                            directMove = false;
                            break;
                        }
                    }
                }
            }
            if(!directMove)
            {
                completelyDirectMove = false;
                (*step1Selection)[i] += 0.5;
                (*step2Selection)[i] += 0.5;
            }
        }
        double dist = abs(endRow - startRow);
        if(dist > maxDist)
        {
            maxDist = dist;
        }
    }
    return completelyDirectMove;
}

ParallelMove ParallelMove::fromStartAndEnd(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray,
    ParallelMove::Step start, ParallelMove::Step end, std::shared_ptr<spdlog::logger> logger)
{
    ParallelMove move;
    ParallelMove::Step step1;
    ParallelMove::Step step2;

    step1.rowSelection = start.rowSelection;
    step1.colSelection = start.colSelection;
    step2.rowSelection = end.rowSelection;
    step2.colSelection = end.colSelection;

    double maxRowDist = 0;
    double maxColDist = 0;

    bool completelyDirectMove = fillSteps(stateArray, start, end, step1, step2, true, maxRowDist, logger);
    if(!completelyDirectMove)
    {
        completelyDirectMove = fillSteps(stateArray, start, end, step1, step2, false, maxColDist, logger);
    }
       
    move.steps.push_back(start);
    if(!completelyDirectMove && (!ALLOW_DIAGONAL_MOVEMENT ||
        (maxRowDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD || maxColDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)))
    {
        bool require4Steps = maxRowDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD && maxColDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD;
        if(!ALLOW_DIAGONAL_MOVEMENT)
        {
            if(maxRowDist > maxColDist)
            {
                step1.rowSelection = start.rowSelection;
                if(require4Steps)
                {
                    step2.colSelection = end.colSelection;
                }
                else
                {
                    step2.rowSelection = end.rowSelection;
                }
            }
            else
            {
                step1.colSelection = start.colSelection;
                if(require4Steps)
                {
                    step2.rowSelection = end.rowSelection;
                }
                else
                {
                    step2.colSelection = end.colSelection;
                }
            }
        }
        move.steps.push_back(step1);
        if(require4Steps)
        {
            ParallelMove::Step intermediateStep;
            if(maxRowDist > maxColDist)
            {
                intermediateStep.rowSelection = step2.rowSelection;
                intermediateStep.colSelection = step1.colSelection;
            }
            else
            {
                intermediateStep.rowSelection = step1.rowSelection;
                intermediateStep.colSelection = step2.colSelection;
            }
            move.steps.push_back(std::move(intermediateStep));
        }
        move.steps.push_back(std::move(step2));
    }
    move.steps.push_back(end);
    return move;
}

double ParallelMove::cost()
{
    double cost = MOVE_COST_OFFSET;
    const ParallelMove::Step *lastStep = &this->steps[0];
    for(size_t step = 1; step < this->steps.size(); step++)
    {
        const ParallelMove::Step *newStep = &this->steps[step];
        double longestColDist = 0;
        for(size_t tone = 0; tone < lastStep->colSelection.size() && tone < newStep->colSelection.size(); tone++)
        {
            double dist = abs((double)(lastStep->colSelection[tone]) - (double)(newStep->colSelection[tone]));
            if(dist > longestColDist)
            {
                longestColDist = dist;
            }
        }
        double longestRowDist = 0;
        for(size_t tone = 0; tone < lastStep->rowSelection.size() && tone < newStep->rowSelection.size(); tone++)
        {
            double dist = abs(lastStep->rowSelection[tone] - newStep->rowSelection[tone]);
            if(dist > longestRowDist)
            {
                longestRowDist = dist;
            }
        }
        if(longestColDist == 0 || longestRowDist == 0)
        {
            cost += costPerSubMove(longestRowDist + longestColDist);
        }
        else if(abs(longestColDist - longestRowDist) < DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            cost += costPerSubMove(longestRowDist * M_SQRT2);
        }
        else
        {
            cost += costPerSubMove(sqrt(longestRowDist * longestRowDist + longestColDist * longestColDist));
        }
        lastStep = newStep;
    }
    return cost;
}

bool ParallelMove::execute(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    const auto& firstStep = this->steps.front();
    const auto& lastStep = this->steps.back();

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrayCopy = stateArray;

    std::stringstream startCols;
    std::stringstream endCols;
    std::stringstream startRows;
    std::stringstream endRows;

    for(const auto& col : firstStep.colSelection)
    {
        startCols << col << " ";
    }
    for(const auto& col : lastStep.colSelection)
    {
        endCols << col << " ";
    }
    for(const auto& row : firstStep.rowSelection)
    {
        startRows << row << " ";
    }
    for(const auto& row : lastStep.rowSelection)
    {
        endRows << row << " ";
    }
    logger->info("Executing move, rows: ({})->({}), cols: ({})->({})", 
        startRows.str(), endRows.str(), startCols.str(), endCols.str());

    if(firstStep.colSelection.size() != lastStep.colSelection.size() || firstStep.rowSelection.size() != lastStep.rowSelection.size())
    {
        logger->error("Move does not have equally many tones at beginning and end");
        return false;
    }
    if(firstStep.colSelection.size() == 0 || firstStep.rowSelection.size() == 0)
    {
        logger->error("No tone selected!");
        return false;
    }

    for(const auto& step : this->steps)
    {
        double lastTone = step.rowSelection[0];
        for(size_t i = 1; i < step.rowSelection.size(); i++)
        {
            if(step.rowSelection[i] <= lastTone + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                logger->error("Move tones not in correct order! {} <= {}", step.rowSelection[i], lastTone);
                return false;
            }
            lastTone = step.rowSelection[i];
        }
        lastTone = step.colSelection[0];
        for(size_t i = 1; i < step.colSelection.size(); i++)
        {
            if(step.colSelection[i] <= lastTone + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                logger->error("Move tones not in correct order! {} <= {}", step.colSelection[i], lastTone);
                return false;
            }
            lastTone = step.colSelection[i];
        }
    }

    for(size_t rowTone = 0; rowTone < firstStep.rowSelection.size(); rowTone++)
    {
        for(size_t colTone = 0; colTone < firstStep.colSelection.size(); colTone++)
        {
            stateArrayCopy(roundCoordDown(firstStep.rowSelection[rowTone]),
                roundCoordDown(firstStep.colSelection[colTone])) = false;
        }
    }
    for(size_t rowTone = 0; rowTone < firstStep.rowSelection.size(); rowTone++)
    {
        for(size_t colTone = 0; colTone < firstStep.colSelection.size(); colTone++)
        {
            if(stateArray(roundCoordDown(firstStep.rowSelection[rowTone]),
                roundCoordDown(firstStep.colSelection[colTone])) && 
                lastStep.rowSelection[rowTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                lastStep.rowSelection[rowTone] < stateArray.rows() && 
                lastStep.colSelection[colTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                lastStep.colSelection[colTone] < stateArray.cols())
            {
                stateArrayCopy(roundCoordDown(lastStep.rowSelection[rowTone]),
                    roundCoordDown(lastStep.colSelection[colTone])) = true;
                if(alreadyMoved.has_value())
                {
                    alreadyMoved.value()(roundCoordDown(lastStep.rowSelection[rowTone]),
                        roundCoordDown(lastStep.colSelection[colTone])) = true;
                }
            }
        }
    }
    stateArray = stateArrayCopy;
    return true;
}

std::tuple<std::optional<ParallelMove>,int,double> improveMoveByAddingIndependentAtom(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger, ParallelMove move, 
    std::optional<double> cost, std::optional<unsigned int> filledVacancies,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    if(AOD_COL_LIMIT <= 1 || AOD_ROW_LIMIT <= 1 || !ALLOW_MOVES_BETWEEN_COLS || !ALLOW_MOVES_BETWEEN_ROWS)
    {
        logger->info("Move limitations prevent method for improving move by adding independent atoms");
        return std::tuple(std::nullopt, 0, __DBL_MAX__);
    }

    ParallelMove::Step start = move.steps[0];
    ParallelMove::Step end = move.steps.back();
    
    if(!cost.has_value())
    {
        cost = move.cost();
    }
    if(!filledVacancies.has_value())
    {
        filledVacancies = 0;
        for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
        {
            for(size_t j = 0; j < start.colSelection.size() && j < end.colSelection.size(); j++)
            {
                if(stateArray(roundCoordDown(start.rowSelection[i]), roundCoordDown(start.colSelection[j])) && 
                    end.rowSelection[i] >= compZone[0] && end.rowSelection[i] < compZone[1] && 
                    end.colSelection[i] >= compZone[2] && end.colSelection[i] < compZone[3])
                {
                    filledVacancies.value()++;
                }
            }
        }
    }

    double rowDiff = 0;
    for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
    {
        double diff = abs(start.rowSelection[i] - end.rowSelection[i]);
        if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            rowDiff = diff;
        }
    }
    double colDiff = 0;
    for(size_t i = 0; i < start.colSelection.size() && i < end.colSelection.size(); i++)
    {
        double diff = abs(start.colSelection[i] - end.colSelection[i]);
        if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            colDiff = diff;
        }
    }
    double baseCost = approxCostPerMove(rowDiff, colDiff);
    double rowCost = 0;
    if(rowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
    {
        rowCost = costPerSubMove(rowDiff - 1);
    }
    double colCost = 0;
    if(colDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
    {
        colCost = costPerSubMove(colDiff - 1);
    }
    
    unsigned int selectedRows = start.rowSelection.size();
    unsigned int selectedCols = start.colSelection.size();

    double bestCostPerFilledVacancy = cost.value() / (double)filledVacancies.value();
    while(selectedRows < AOD_ROW_LIMIT && selectedCols < AOD_COL_LIMIT && (selectedCols + 1) * (selectedRows + 1) < AOD_TOTAL_LIMIT)
    {
        std::optional<std::tuple<size_t,size_t,size_t,size_t>> addedOuterAndInnerRowAndCol = std::nullopt;

        std::vector<std::tuple<size_t,size_t,size_t>> viableColsWithInnerStartAndEnd;
        for(size_t colIndex = 0; colIndex <= selectedCols; colIndex++)
        {
            size_t outerStartIndex = colIndex > 0 ? start.colSelection[colIndex - 1] + 1 : 0;
            size_t outerEndIndex = colIndex < selectedCols ? start.colSelection[colIndex] : stateArray.cols();
            size_t innerStartIndex = colIndex > 0 ? end.colSelection[colIndex - 1] + 1 : compZone[2];
            size_t innerEndIndex = colIndex < selectedCols ? end.colSelection[colIndex] : compZone[3];
            if(innerEndIndex <= innerStartIndex)
            {
                continue;
            }
            for(size_t outerCol = outerStartIndex; outerCol < outerEndIndex; outerCol++)
            {
                bool allowed = true;
                for(double row : start.rowSelection)
                {
                    if(stateArray(roundCoordDown(row), outerCol))
                    {
                        allowed = false;
                        break;
                    }
                }
                if(allowed)
                {
                    viableColsWithInnerStartAndEnd.push_back(std::tuple(outerCol, innerStartIndex, innerEndIndex));
                }
            }
        }
        for(size_t rowIndex = 0; rowIndex <= selectedRows; rowIndex++)
        {
            size_t outerStartIndex = rowIndex > 0 ? start.rowSelection[rowIndex - 1] + 1 : 0;
            size_t outerEndIndex = rowIndex < selectedRows ? start.rowSelection[rowIndex] : stateArray.rows();
            size_t innerStartIndex = rowIndex > 0 ? end.rowSelection[rowIndex - 1] + 1 : compZone[0];
            size_t innerEndIndex = rowIndex < selectedRows ? end.rowSelection[rowIndex] : compZone[1];
            if(innerEndIndex <= innerStartIndex)
            {
                continue;
            }
            for(size_t outerRow = outerStartIndex; outerRow < outerEndIndex; outerRow++)
            {
                bool allowed = true;
                for(double col : start.colSelection)
                {
                    if(stateArray(outerRow,roundCoordDown(col)))
                    {
                        allowed = false;
                        break;
                    }
                }
                if(allowed)
                {
                    for(const auto& [outerCol, innerColStart, innerColEnd] : viableColsWithInnerStartAndEnd)
                    {
                        if(stateArray(outerRow, outerCol) && (outerRow < compZone[0] || outerRow >= compZone[1] || 
                            outerCol < compZone[2] || outerCol >= compZone[3]))
                        {
                            for(size_t innerRow = innerStartIndex; innerRow < innerEndIndex; innerRow++)
                            {
                                bool allowed = true;
                                if(!ALLOW_MOVING_EMPTY_TRAP_ONTO_OCCUPIED)
                                {
                                    for(double col : end.colSelection)
                                    {
                                        if(stateArray(innerRow, roundCoordDown(col)))
                                        {
                                            allowed = false;
                                            break;
                                        }
                                    }
                                }
                                for(size_t innerCol = innerColStart; innerCol < innerColEnd; innerCol++)
                                {
                                    if(!ALLOW_MOVING_EMPTY_TRAP_ONTO_OCCUPIED)
                                    {
                                        for(double row : end.rowSelection)
                                        {
                                            if(stateArray(roundCoordDown(row), innerCol))
                                            {
                                                allowed = false;
                                                break;
                                            }
                                        }
                                    }
                                    if(allowed && !stateArray(innerRow, innerCol))
                                    {
                                        double newCost = baseCost;
                                        double newRowDiff = abs((double)innerRow - (double)outerRow);
                                        if(newRowDiff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD && newRowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                        {
                                            newCost += costPerSubMove(newRowDiff - 1) - rowCost;
                                        }
                                        double newColDiff = abs((double)innerCol - (double)outerCol);
                                        if(newColDiff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD && newColDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                        {
                                            newCost += costPerSubMove(newColDiff - 1) - colCost;
                                        }
                                        if(newCost / (filledVacancies.value() + 1) < bestCostPerFilledVacancy)
                                        {
                                            bestCostPerFilledVacancy = newCost / (filledVacancies.value() + 1);
                                            addedOuterAndInnerRowAndCol = std::tuple(outerRow, innerRow, outerCol, innerCol);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if(addedOuterAndInnerRowAndCol.has_value())
        {
            double outerRow = std::get<0>(addedOuterAndInnerRowAndCol.value());
            double innerRow = std::get<1>(addedOuterAndInnerRowAndCol.value());
            double outerCol = std::get<2>(addedOuterAndInnerRowAndCol.value());
            double innerCol = std::get<3>(addedOuterAndInnerRowAndCol.value());
            filledVacancies.value()++;
            start.rowSelection.insert(std::upper_bound(start.rowSelection.begin(), start.rowSelection.end(), 
                outerRow), outerRow);
            end.rowSelection.insert(std::upper_bound(end.rowSelection.begin(), end.rowSelection.end(), 
                innerRow), innerRow);
            start.colSelection.insert(std::upper_bound(start.colSelection.begin(), start.colSelection.end(), 
                outerCol), outerCol);
            end.colSelection.insert(std::upper_bound(end.colSelection.begin(), end.colSelection.end(), 
                innerCol), innerCol);
            selectedRows++;
            selectedCols++;
            double newRowDiff = abs((double)innerRow - (double)outerRow);
            if(newRowDiff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                rowDiff = newRowDiff;
                rowCost = 0;
                if(newRowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                {
                    rowCost = costPerSubMove(newRowDiff - 1);
                }
            }
            double newColDiff = abs((double)innerCol - (double)outerCol);
            if(newColDiff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                colDiff = newColDiff;
                colCost = 0;
                if(newColDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                {
                    colCost = costPerSubMove(newColDiff - 1);
                }
            }

            logger->debug("Adding row {}->{} and col {}->{} for 1 additional vacancy filling", 
                outerRow, innerRow, outerCol, innerCol);
        }
        else
        {
            break;
        }
    }
    
    ParallelMove improvedMove = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
    bestCostPerFilledVacancy = improvedMove.cost() / filledVacancies.value();
    return std::tuple(std::move(improvedMove), filledVacancies.value(), bestCostPerFilledVacancy);
}

std::tuple<std::optional<ParallelMove>,int,double> improveComplexMove(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger, ParallelMove move, 
    std::optional<double> cost, std::optional<unsigned int> filledVacancies,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    if(AOD_COL_LIMIT <= 1 || AOD_ROW_LIMIT <= 1 || !ALLOW_MOVES_BETWEEN_COLS || !ALLOW_MOVES_BETWEEN_ROWS)
    {
        logger->info("Move limitations prevent method for improving complex row and col move");
        return std::tuple(std::nullopt, 0, __DBL_MAX__);
    }

    ParallelMove::Step start = move.steps[0];
    ParallelMove::Step end = move.steps.back();

    if(!cost.has_value())
    {
        cost = move.cost();
    }
    if(!filledVacancies.has_value())
    {
        filledVacancies = 0;
        for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
        {
            for(size_t j = 0; j < start.colSelection.size() && j < end.colSelection.size(); j++)
            {
                if(stateArray(roundCoordDown(start.rowSelection[i]), roundCoordDown(start.colSelection[j])) && 
                    end.rowSelection[i] >= compZone[0] && end.rowSelection[i] < compZone[1] && 
                    end.colSelection[i] >= compZone[2] && end.colSelection[i] < compZone[3])
                {
                    filledVacancies.value()++;
                }
            }
        }
    }

    logger->debug("Move to optimize already fills {} vacancies", filledVacancies.value());

    double rowDiff = 0;
    for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
    {
        double diff = abs(start.rowSelection[i] - end.rowSelection[i]);
        if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            rowDiff = diff;
        }
    }
    double colDiff = 0;
    for(size_t i = 0; i < start.colSelection.size() && i < end.colSelection.size(); i++)
    {
        double diff = abs(start.colSelection[i] - end.colSelection[i]);
        if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            colDiff = diff;
        }
    }
    double baseCost = approxCostPerMove(rowDiff, colDiff);
    logger->debug("Base cost: {} for r {} c {}", baseCost, rowDiff, colDiff);
    double costWithoutRowMove = approxCostPerMove(0, colDiff);
    double costWithoutColMove = approxCostPerMove(rowDiff, 0);

    unsigned int selectedRows = start.rowSelection.size();
    unsigned int selectedCols = start.colSelection.size();

    unsigned int bestAdditionalFilledVacancies = 0;
    double bestCostPerFilledVacancy = cost.value() / (double)filledVacancies.value();
    unsigned int bestAdditionalDiff = 0;
    while(true)
    {
        std::optional<std::tuple<size_t,size_t,bool>> addedOuterAndInnerIndexAndIsRow = std::nullopt;

        if(selectedRows < AOD_ROW_LIMIT && selectedCols * (selectedRows + 1) < AOD_TOTAL_LIMIT)
        {
            for(size_t rowIndex = 0; rowIndex <= selectedRows; rowIndex++)
            {
                size_t outerStartIndex = rowIndex > 0 ? start.rowSelection[rowIndex - 1] + 1 : 0;
                size_t outerEndIndex = rowIndex < selectedRows ? start.rowSelection[rowIndex] : stateArray.rows();
                size_t innerStartIndex = rowIndex > 0 ? end.rowSelection[rowIndex - 1] + 1 : compZone[0];
                size_t innerEndIndex = rowIndex < selectedRows ? end.rowSelection[rowIndex] : compZone[1];
                if(innerEndIndex <= innerStartIndex)
                {
                    continue;
                }
                for(size_t outerRow = outerStartIndex; outerRow < outerEndIndex; outerRow++)
                {
                    if(!ALLOW_MULTIPLE_MOVES_PER_ATOM && alreadyMoved.has_value())
                    {
                        bool allowed = true;
                        for(size_t colIndex = 0; colIndex < selectedCols; colIndex++)
                        {
                            if(alreadyMoved.value()(outerRow, roundCoordDown(start.colSelection[colIndex])))
                            {
                                allowed = false;
                            }
                        }
                        if(!allowed)
                        {
                            continue;
                        }
                    }
                    for(size_t innerRow = innerStartIndex; innerRow < innerEndIndex; innerRow++)
                    {
                        unsigned int additionalFilledVacancies = 0;
                        bool allowed = true;
                        for(size_t colIndex = 0; colIndex < selectedCols; colIndex++)
                        {
                            size_t startCol = roundCoordDown(start.colSelection[colIndex]);
                            if(stateArray(outerRow, startCol))
                            {
                                if(stateArray(innerRow, roundCoordDown(end.colSelection[colIndex])))
                                {
                                    allowed = false;
                                    break;
                                }
                                else if(outerRow < compZone[0] || outerRow >= compZone[1] ||
                                    startCol < compZone[2] || startCol >= compZone[3])
                                {
                                    additionalFilledVacancies++;
                                }
                            }
                            else if(stateArray(innerRow, roundCoordDown(end.colSelection[colIndex])) && !ALLOW_MOVING_EMPTY_TRAP_ONTO_OCCUPIED)
                            {
                                allowed = false;
                            }
                        }
                        if(allowed && additionalFilledVacancies > 0)
                        {
                            double newCost = baseCost;
                            double diff = abs((double)outerRow - (double)innerRow);
                            if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD && diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                            {
                                newCost = costWithoutRowMove + costPerSubMove(diff - 1);
                            }
                            double costPerFilledVacancy = newCost / (filledVacancies.value() + additionalFilledVacancies);
                            if(costPerFilledVacancy < bestCostPerFilledVacancy)
                            {
                                if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
                                {
                                    bestAdditionalDiff = diff - colDiff;
                                }
                                else
                                {
                                    bestAdditionalDiff = 0;
                                }
                                bestAdditionalFilledVacancies = additionalFilledVacancies;
                                bestCostPerFilledVacancy = costPerFilledVacancy;
                                addedOuterAndInnerIndexAndIsRow = std::tuple(outerRow, innerRow, true);
                            }
                        }
                    }
                }
            }
        }
        if(selectedCols < AOD_COL_LIMIT && selectedRows * (selectedCols + 1) < AOD_TOTAL_LIMIT)
        {
            for(size_t colIndex = 0; colIndex <= selectedCols; colIndex++)
            {
                size_t outerStartIndex = colIndex > 0 ? start.colSelection[colIndex - 1] + 1 : 0;
                size_t outerEndIndex = colIndex < selectedCols ? start.colSelection[colIndex] : stateArray.cols();
                size_t innerStartIndex = colIndex > 0 ? end.colSelection[colIndex - 1] + 1 : compZone[2];
                size_t innerEndIndex = colIndex < selectedCols ? end.colSelection[colIndex] : compZone[3];
                if(innerEndIndex <= innerStartIndex)
                {
                    continue;
                }
                for(size_t outerCol = outerStartIndex; outerCol < outerEndIndex; outerCol++)
                {
                    if(!ALLOW_MULTIPLE_MOVES_PER_ATOM && alreadyMoved.has_value())
                    {
                        bool allowed = true;
                        for(size_t rowIndex = 0; rowIndex < selectedRows; rowIndex++)
                        {
                            if(alreadyMoved.value()(roundCoordDown(start.rowSelection[rowIndex]), outerCol))
                            {
                                allowed = false;
                            }
                        }
                        if(!allowed)
                        {
                            continue;
                        }
                    }
                    for(size_t innerCol = innerStartIndex; innerCol < innerEndIndex; innerCol++)
                    {
                        unsigned int additionalFilledVacancies = 0;
                        bool allowed = true;
                        for(size_t rowIndex = 0; rowIndex < selectedRows; rowIndex++)
                        {
                            size_t startRow = roundCoordDown(start.rowSelection[rowIndex]);
                            if(stateArray(startRow, outerCol))
                            {
                                if(stateArray(roundCoordDown(end.rowSelection[rowIndex]), innerCol))
                                {
                                    allowed = false;
                                    break;
                                }
                                else if(startRow < compZone[0] || startRow >= compZone[1] ||
                                    outerCol < compZone[2] || outerCol >= compZone[3])
                                {
                                    additionalFilledVacancies++;
                                }
                            }
                            else if(stateArray(roundCoordDown(end.rowSelection[rowIndex]), innerCol) && !ALLOW_MOVING_EMPTY_TRAP_ONTO_OCCUPIED)
                            {
                                allowed = false;
                            }
                        }
                        if(allowed && additionalFilledVacancies > 0)
                        {
                            double newCost = baseCost;
                            double diff = abs((double)outerCol - (double)innerCol);
                            if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD && diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                            {
                                newCost = costWithoutColMove + costPerSubMove(diff - 1);
                            }
                            double costPerFilledVacancy = newCost / (filledVacancies.value() + additionalFilledVacancies);
                            if(costPerFilledVacancy < bestCostPerFilledVacancy)
                            {
                                if(diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                {
                                    bestAdditionalDiff = diff - rowDiff;
                                }
                                else
                                {
                                    bestAdditionalDiff = 0;
                                }
                                bestAdditionalFilledVacancies = additionalFilledVacancies;
                                bestCostPerFilledVacancy = costPerFilledVacancy;
                                addedOuterAndInnerIndexAndIsRow = std::tuple(outerCol, innerCol, false);
                            }
                        }
                    }
                }
            }
        }

        if(addedOuterAndInnerIndexAndIsRow.has_value())
        {
            double startIndex = std::get<0>(addedOuterAndInnerIndexAndIsRow.value());
            double endIndex = std::get<1>(addedOuterAndInnerIndexAndIsRow.value());
            filledVacancies.value() += bestAdditionalFilledVacancies;
            if(std::get<2>(addedOuterAndInnerIndexAndIsRow.value()))
            {
                start.rowSelection.insert(std::upper_bound(start.rowSelection.begin(), start.rowSelection.end(), 
                    startIndex), startIndex);
                end.rowSelection.insert(std::upper_bound(end.rowSelection.begin(), end.rowSelection.end(), 
                    endIndex), endIndex);
                selectedRows++;
                rowDiff += bestAdditionalDiff;
                logger->debug("Adding row {}->{} for {} additional vacancy fillings", 
                    startIndex, endIndex, bestAdditionalFilledVacancies);
            }
            else
            {
                start.colSelection.insert(std::upper_bound(start.colSelection.begin(), start.colSelection.end(), 
                    startIndex), startIndex);
                end.colSelection.insert(std::upper_bound(end.colSelection.begin(), end.colSelection.end(), 
                    endIndex), endIndex);
                selectedCols++;
                colDiff += bestAdditionalDiff;
                logger->debug("Adding col {}->{} for {} additional vacancy fillings", 
                    startIndex, endIndex, bestAdditionalFilledVacancies);
            }
            break;
        }
        else
        {
            break;
        }
    }

    ParallelMove improvedMove = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
    bestCostPerFilledVacancy = improvedMove.cost() / filledVacancies.value();
    return std::tuple(std::move(improvedMove), filledVacancies.value(), bestCostPerFilledVacancy);
}

std::pair<double, std::optional<ParallelMove>> checkComplexMoveCost(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger, RowBitMask& sourceBitMask, RowBitMask& targetBitMask, 
    bool rowFirst, size_t colDimCompZone[2], size_t cols, unsigned int maxColCount, std::optional<double> bestCostPerFilledGap)
{
    if(sourceBitMask.indices.size() != targetBitMask.indices.size())
    {
        return std::pair(0,std::nullopt);
    }
    else
    {
        unsigned int fillableCols = targetBitMask.bitsSet();
        if(maxColCount < fillableCols)
        {
            fillableCols = maxColCount;
        }

        double cost = MOVE_COST_OFFSET + 2 * HALF_DIAG_STEP_COST;
        double maxDist = 0;
        for(size_t i = 0; i < sourceBitMask.indices.size(); i++)
        {
            double dist = abs((int)(sourceBitMask.indices[i]) - (int)(targetBitMask.indices[i]));
            if(dist > maxDist)
            {
                maxDist = dist;
            }
        }
        if(maxDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            cost += costPerSubMove(maxDist - 1);
        }

        std::vector<unsigned int> outerDistancesLeft;
        for(int j = colDimCompZone[0] - 1; j >= 0; j--)
        {
            if(sourceBitMask[(size_t)j])
            {
                outerDistancesLeft.push_back(colDimCompZone[0] - (size_t)j);
            }
        }

        std::vector<unsigned int> outerDistancesRight;
        for(size_t j = colDimCompZone[1]; j < cols; j++)
        {
            if(sourceBitMask[j - colDimCompZone[1] + colDimCompZone[0]])
            {
                outerDistancesRight.push_back(j - colDimCompZone[1]);
            }
        }

        std::optional<unsigned int> bestAtomsFromLeft = std::nullopt;
        unsigned int bestMaxDist = 0;
        int minFromLeft = fillableCols - outerDistancesRight.size();
        if(minFromLeft < 0)
        {
            minFromLeft = 0;
        }
        int maxFromLeft = fillableCols > outerDistancesLeft.size() ? outerDistancesLeft.size() : fillableCols;
        std::vector<size_t> possibleTargetCols;
        for(size_t j = colDimCompZone[0]; j < colDimCompZone[1]; j++)
        {
            if(targetBitMask[j - colDimCompZone[0]])
            {
                possibleTargetCols.push_back(j);
            }
        }

        for(unsigned int atomsFromLeft = minFromLeft; atomsFromLeft <= maxFromLeft; atomsFromLeft++)
        {
            unsigned int maxDist = 0;
            for(size_t atomIndex = 0; atomIndex < atomsFromLeft; atomIndex++)
            {
                unsigned int dist = possibleTargetCols[atomIndex] - colDimCompZone[0] + outerDistancesLeft[atomsFromLeft - atomIndex - 1];
                if(dist > maxDist)
                {
                    maxDist = dist;
                }
            }
            for(size_t atomIndex = 0; atomIndex < fillableCols - atomsFromLeft; atomIndex++)
            {
                unsigned int dist = colDimCompZone[1] - possibleTargetCols[possibleTargetCols.size() - atomIndex - 1] + outerDistancesRight[fillableCols - atomsFromLeft - atomIndex - 1];
                if(dist > maxDist)
                {
                    maxDist = dist;
                }
            }
            if(!bestAtomsFromLeft.has_value() || maxDist < bestMaxDist)
            {
                bestAtomsFromLeft = atomsFromLeft;
                bestMaxDist = maxDist;
            }
        }
        if(bestMaxDist > 1)
        {
            cost += costPerSubMove(bestMaxDist - 1);
        }

        // If cost per filled gap is better than previous best, return new best move
        if(bestAtomsFromLeft.has_value() && (!bestCostPerFilledGap.has_value() ||
            (cost / (fillableCols * targetBitMask.indices.size()) < bestCostPerFilledGap.value())))
        {
            ParallelMove::Step start;
            ParallelMove::Step end;

            if(rowFirst)
            {
                start.colSelection.reserve(fillableCols);
                start.rowSelection = std::vector<double>(sourceBitMask.indices.begin(), sourceBitMask.indices.end());
                end.colSelection.reserve(fillableCols);
                end.rowSelection = std::vector<double>(targetBitMask.indices.begin(), targetBitMask.indices.end());
            }
            else
            {
                start.rowSelection.reserve(fillableCols);
                start.colSelection = std::vector<double>(sourceBitMask.indices.begin(), sourceBitMask.indices.end());
                end.rowSelection.reserve(fillableCols);
                end.colSelection = std::vector<double>(targetBitMask.indices.begin(), targetBitMask.indices.end());
            }

            auto& outerDimStartVector = rowFirst ? start.colSelection : start.rowSelection;
            auto& outerDimEndVector = rowFirst ? end.colSelection : end.rowSelection;

            for(size_t atomIndex = 0; atomIndex < bestAtomsFromLeft.value(); atomIndex++)
            {
                outerDimStartVector.push_back(colDimCompZone[0] - outerDistancesLeft[bestAtomsFromLeft.value() - atomIndex - 1]);
                outerDimEndVector.push_back(possibleTargetCols[atomIndex]);
            }
            for(int atomIndex = fillableCols - bestAtomsFromLeft.value() - 1; atomIndex >= 0; atomIndex--)
            {
                outerDimStartVector.push_back(colDimCompZone[1] + outerDistancesRight[fillableCols - bestAtomsFromLeft.value() - (size_t)atomIndex - 1]);
                outerDimEndVector.push_back(possibleTargetCols[possibleTargetCols.size() - (size_t)atomIndex - 1]);
            }

            ParallelMove move = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
            return std::pair(cost, move);
        }
    }
    return std::pair(0,std::nullopt);
}

bool stopFurtherMoveInvestigationAtRowCount(unsigned int rowCount, unsigned int maxRows, 
    unsigned int currIter, unsigned int iterCountRowCountStart, size_t availableElems, size_t alreadyInvElems, 
    std::vector<std::pair<unsigned int,std::shared_ptr<RowBitMask>>> *bitMasksPerInnerRowSet)
{
    if(bitMasksPerInnerRowSet == nullptr || bitMasksPerInnerRowSet->empty() || rowCount == maxRows)
    {
        return false;
    }
    else
    {
        // Skew towards earlier by giving 150% to the first and 50% to the last and interpolating linearly inbetween
        unsigned int targetIterCount = MAX_MULTI_ITER_COUNT / (maxRows - rowCount + 1) * 
            (1.5 - (rowCount - 2) / (maxRows - 2));
        double alreadyInvFrac = (double)(currIter - iterCountRowCountStart) / (double)targetIterCount;
        if(alreadyInvFrac > 1)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

std::tuple<std::optional<ParallelMove>,int,double> moveSeveralRowsAndCols(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    if(AOD_COL_LIMIT <= 1 || AOD_ROW_LIMIT <= 1 || !ALLOW_MOVES_BETWEEN_COLS || !ALLOW_MOVES_BETWEEN_ROWS)
    {
        logger->info("Move limitations prevent method for complicated row and col move");
        return std::tuple(std::nullopt, 0, __DBL_MAX__);
    }

    unsigned int roundedDownSqrtTotalAOD = sqrt(AOD_TOTAL_LIMIT);
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestFillableGaps = 0;
    double bestCostPerFilledGap = 0;
    for(bool rowFirst : {true, false})
    {
        size_t rowDimCompZone[2];
        size_t colDimCompZone[2];
        size_t rows;
        size_t cols;
        unsigned int rowAODLimit = 0;
        unsigned int colAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, rowDimCompZone, colDimCompZone, rows, cols, rowAODLimit, colAODLimit);

        if(rowAODLimit <= 1)
        {
            continue;
        }

        unsigned int maxRows = rowAODLimit;
        if(maxRows > roundedDownSqrtTotalAOD)
        {
            maxRows = roundedDownSqrtTotalAOD;
        }

        std::vector<std::shared_ptr<RowBitMask>> bitMaskByInnerRow;
        std::vector<RowBitMask> bitMaskByOuterRow;
        std::vector<std::pair<unsigned int,std::shared_ptr<RowBitMask>>> bitMaskInnerVec1;
        std::vector<RowBitMask> bitMaskOuterVec1;
        for(size_t i = 0; i < rows; i++)
        {
            RowBitMask outerRowBitMask(cols - (colDimCompZone[1] - colDimCompZone[0]), i);
            for(size_t j = 0; j < cols - (colDimCompZone[1] - colDimCompZone[0]); j++)
            {
                size_t index = j;
                if(index >= colDimCompZone[0])
                {
                    index += colDimCompZone[1] - colDimCompZone[0];
                }
                outerRowBitMask.set(j, accessStateArray(stateArray, i, index, rowFirst));
            }
            bitMaskByOuterRow.push_back(outerRowBitMask);
            bitMaskOuterVec1.push_back(std::move(outerRowBitMask));
            if(i >= rowDimCompZone[0] && i < rowDimCompZone[1])
            {
                RowBitMask innerRowBitMask(colDimCompZone[1] - colDimCompZone[0], i);
                for(size_t j = colDimCompZone[0]; j < colDimCompZone[1]; j++)
                {
                    innerRowBitMask.set(j - colDimCompZone[0], !accessStateArray(stateArray, i, j, rowFirst));
                }
                bitMaskByInnerRow.push_back(std::make_shared<RowBitMask>(innerRowBitMask));
                bitMaskInnerVec1.push_back(std::pair(innerRowBitMask.bitsSet(), std::make_shared<RowBitMask>(innerRowBitMask)));
            }
        }

        // Calculate maximum target sizes
        std::vector<std::pair<unsigned int,std::shared_ptr<RowBitMask>>> bitMaskInnerVec2;
        std::vector<std::pair<unsigned int,std::shared_ptr<RowBitMask>>> *prevBitMasksPerInnerRowSet = &bitMaskInnerVec1;
        std::vector<std::pair<unsigned int,std::shared_ptr<RowBitMask>>> *bitMasksPerInnerRowSet = &bitMaskInnerVec2;

        unsigned int bestTotalSize = 0;

        std::map<unsigned int,std::pair<unsigned int, std::shared_ptr<RowBitMask>>> bestSizePerInnerRowCount;

        unsigned int iterCount = 0;
        for(unsigned int rowCount = 2; rowCount <= maxRows && iterCount < MAX_MULTI_ITER_COUNT; rowCount++)
        {
            unsigned int iterCountAtRowStart = iterCount;
            bestSizePerInnerRowCount[rowCount] = std::pair(0,nullptr);
            unsigned int i = 0;
            for(const auto& prevBitMask : *prevBitMasksPerInnerRowSet)
            {
                if(stopFurtherMoveInvestigationAtRowCount(rowCount, maxRows, iterCount, 
                    iterCountAtRowStart, prevBitMasksPerInnerRowSet->size(), i, bitMasksPerInnerRowSet))
                {
                    break;
                }
                for(size_t i = prevBitMask.second->indices.back() + 1; i < rowDimCompZone[1] && 
                    iterCount < MAX_MULTI_ITER_COUNT; i++)
                {
                    iterCount++;
                    RowBitMask combBitMask = RowBitMask::fromAnd(*prevBitMask.second, *bitMaskByInnerRow[i - rowDimCompZone[0]]);
                    unsigned int overlap = combBitMask.bitsSet();
                    if(overlap > bestSizePerInnerRowCount[rowCount].first)
                    {
                        bestSizePerInnerRowCount[rowCount] = std::pair(
                            overlap, std::make_shared<RowBitMask>(combBitMask));
                    }
                    if(overlap * rowCount > bestTotalSize)
                    {
                        bestTotalSize = overlap * rowCount;
                    }
                    if(overlap >= bestTotalSize / (rowCount + 1) && overlap >= 2)
                    {
                        bitMasksPerInnerRowSet->push_back(std::pair(overlap,std::make_shared<RowBitMask>(combBitMask)));
                    }
                }
                if(iterCount > MAX_MULTI_ITER_COUNT)
                {
                    break;
                }
                i++;
            }
            auto *tmpRef = prevBitMasksPerInnerRowSet;
            prevBitMasksPerInnerRowSet = bitMasksPerInnerRowSet;
            bitMasksPerInnerRowSet = tmpRef;
            bitMasksPerInnerRowSet->clear();
        }

        for(unsigned int rowCount = 2; rowCount <= maxRows; rowCount++)
        {
            if(bestSizePerInnerRowCount[rowCount].second == nullptr || bestSizePerInnerRowCount[rowCount].first < 2)
            {
                break;
            }
            logger->debug("Max empty columns for {} rows: {}", rowCount, bestSizePerInnerRowCount[rowCount].first);
        }

        // Calculate maximum sizes for border
        std::vector<RowBitMask> bitMaskOuterVec2;
        std::vector<RowBitMask> *prevBitMasksPerOuterRowSet = &bitMaskOuterVec1;
        std::vector<RowBitMask> *bitMasksPerOuterRowSet = &bitMaskOuterVec2;

        iterCount = 0;
        for(unsigned int rowCount = 2; rowCount <= maxRows && iterCount < MAX_MULTI_ITER_COUNT; rowCount++)
        {
            if(bestSizePerInnerRowCount[rowCount].second == nullptr || bestSizePerInnerRowCount[rowCount].first < 2)
            {
                continue;
            }
            unsigned int maxCols = colAODLimit;
            if(AOD_TOTAL_LIMIT / rowCount < maxCols)
            {
                maxCols = AOD_TOTAL_LIMIT / rowCount;
            }
            for(const auto& prevBitMask : *prevBitMasksPerOuterRowSet)
            {
                for(size_t i = prevBitMask.indices.back() + 1; i < rows && iterCount < MAX_MULTI_ITER_COUNT; i++)
                {
                    iterCount++;
                    RowBitMask combBitMask = RowBitMask::fromAnd(prevBitMask, bitMaskByOuterRow[i]);
                    unsigned int overlap = combBitMask.bitsSet();
                    if(overlap >= bestSizePerInnerRowCount[rowCount].first)
                    {
                        auto [cost,move] = checkComplexMoveCost(stateArray, compZone, logger, combBitMask, 
                            *bestSizePerInnerRowCount[rowCount].second, rowFirst, colDimCompZone, cols, maxCols, 
                            bestMove.has_value() ? std::optional(bestCostPerFilledGap) : std::nullopt);
                        if(move.has_value())
                        {
                            unsigned int filledVacancies = move.value().steps[0].rowSelection.size() * move.value().steps[0].colSelection.size();
                            double costPerFilledVacancy = move.value().cost() / filledVacancies;
                            if(!bestMove.has_value() || costPerFilledVacancy < bestCostPerFilledGap)
                            {
                                bestCostPerFilledGap = costPerFilledVacancy;
                                bestFillableGaps = filledVacancies;
                                bestMove = move;
                            }
                        }
                    }
                    // Take all row sets into account that have at least as much overlap as is needed for one more row
                    if(rowCount < maxRows && overlap >= bestSizePerInnerRowCount[rowCount + 1].first)
                    {
                        bitMasksPerOuterRowSet->push_back(combBitMask);
                    }
                }
                if(iterCount > MAX_MULTI_ITER_COUNT)
                {
                    break;
                }
            }
            std::vector<RowBitMask> *tmpRef = prevBitMasksPerOuterRowSet;
            prevBitMasksPerOuterRowSet = bitMasksPerOuterRowSet;
            bitMasksPerOuterRowSet = tmpRef;
            bitMasksPerOuterRowSet->clear();
        }
    }

    if(bestMove.has_value())
    {
        auto [improvedMove, filledVacancies, costPerFilledVacancy] = 
            improveComplexMove(stateArray, compZone, logger, bestMove.value(), std::nullopt, bestFillableGaps, alreadyMoved);
        if(improvedMove.has_value())
        {
            logger->info("Best multi row and column move can fill {} at a cost of {} per gap", filledVacancies, costPerFilledVacancy);
            return std::tuple(improvedMove, filledVacancies, costPerFilledVacancy);
        }
        else
        {
            logger->info("Best multi row and column move can fill {} at a cost of {} per gap", bestFillableGaps, bestCostPerFilledGap);
            return std::tuple(bestMove.value(), bestFillableGaps, bestCostPerFilledGap);
        }
    }
    else
    {
        logger->info("No multi row and column move could be found");
        return std::tuple(std::nullopt, 0, __DBL_MAX__);
    }
}

std::tuple<std::optional<ParallelMove>,int,double> fillColumnHorizontally(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>)
{
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestFillableGaps = 0;
    double bestCostPerFilledGap = 0;
    for(bool rowFirst : {true, false})
    {
        if((rowFirst && !ALLOW_MOVES_BETWEEN_ROWS) || (!rowFirst && !ALLOW_MOVES_BETWEEN_COLS))
        {
            continue;
        }
        size_t rowDimCompZone[2];
        size_t colDimCompZone[2];
        size_t rows;
        size_t cols;
        unsigned int rowAODLimit = 0;
        unsigned int colAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, rowDimCompZone, colDimCompZone, rows, cols, rowAODLimit, colAODLimit);

        unsigned int aodLimit = rowAODLimit < AOD_TOTAL_LIMIT ? rowAODLimit : AOD_TOTAL_LIMIT;

        std::vector<std::vector<size_t>> emptyCompZoneLocations;
        for(size_t j = colDimCompZone[0]; j < colDimCompZone[1]; j++)
        {
            std::vector<size_t> emptySitesInCol;
            for(size_t i = rowDimCompZone[0]; i < rowDimCompZone[1]; i++)
            {
                if(!accessStateArray(stateArray, i, j, rowFirst))
                {
                    emptySitesInCol.push_back(i);
                }
            }
            emptyCompZoneLocations.push_back(std::move(emptySitesInCol));
        }

        for(size_t borderColTemp = 0; borderColTemp < cols - (colDimCompZone[1] - colDimCompZone[0]); borderColTemp++)
        {
            size_t borderCol = borderColTemp;
            size_t borderAtomsIndex = borderColTemp;
            if(borderColTemp >= colDimCompZone[0])
            {
                borderCol += (colDimCompZone[1] - colDimCompZone[0]);
                borderAtomsIndex -= colDimCompZone[0];
            }
            std::vector<size_t> atomLocations;

            ParallelMove::Step start;
            if(rowFirst)
            {
                start.colSelection.push_back(borderCol);
            }
            else
            {
                start.rowSelection.push_back(borderCol);
            }

            for(size_t i = 0; i < rows; i++)
            {
                if(accessStateArray(stateArray, i, borderCol, rowFirst))
                {
                    atomLocations.push_back(i);
                }
            }

            for(size_t targetCol = colDimCompZone[0]; targetCol < colDimCompZone[1]; targetCol++)
            {
                ParallelMove::Step end;
                if(rowFirst)
                {
                    end.colSelection.push_back(targetCol);
                }
                else
                {
                    end.rowSelection.push_back(targetCol);
                }

                std::vector<size_t> atomLocationsCopy = atomLocations;
                const auto& emptySitesInCol = emptyCompZoneLocations[targetCol - colDimCompZone[0]];
                size_t *sourceSiteMapping = new size_t[emptySitesInCol.size()]();
                unsigned int filledSites = 0;
                size_t dist = 1;
                for(; dist < cols; dist++)
                {
                    if(rowFirst)
                    {
                        start.rowSelection.clear();
                        end.rowSelection.clear();
                    }
                    else
                    {
                        start.colSelection.clear();
                        end.colSelection.clear();
                    }
                    size_t atomLocationIndex = 0;
                    filledSites = 0;
                    for(const auto& emptySite : emptySitesInCol)
                    {
                        while(atomLocationIndex < atomLocations.size() && (int)atomLocations[atomLocationIndex] < (int)emptySite - (int)dist)
                        {
                            atomLocationIndex++;
                        }
                        if(atomLocationIndex >= atomLocations.size() || filledSites == aodLimit)
                        {
                            break;
                        }
                        else if((int)atomLocations[atomLocationIndex] >= (int)emptySite - (int)dist && 
                            atomLocations[atomLocationIndex] <= emptySite + dist)
                        {
                            if(rowFirst)
                            {
                                start.rowSelection.push_back(atomLocations[atomLocationIndex]);
                                end.rowSelection.push_back(emptySite);
                            }
                            else
                            {
                                start.colSelection.push_back(atomLocations[atomLocationIndex]);
                                end.colSelection.push_back(emptySite);
                            }
                            filledSites++;
                            atomLocationIndex++;
                        }
                    }
                    double approxCost = MOVE_COST_OFFSET + costPerSubMove(dist - 1) + 
                        costPerSubMove(abs((int)borderCol - (int)targetCol) - 1) + 2 * HALF_DIAG_STEP_COST;
                    if(!bestMove.has_value() || approxCost / filledSites < bestCostPerFilledGap)
                    {
                        ParallelMove move = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
                        double costPerGap = move.cost() / filledSites;
                        if(!bestMove.has_value() || costPerGap < bestCostPerFilledGap)
                        {
                            bestMove = move;
                            bestCostPerFilledGap = costPerGap;
                            bestFillableGaps = filledSites;
                        }
                    }
                    if(filledSites == emptySitesInCol.size() || filledSites == atomLocations.size() || filledSites >= aodLimit)
                    {
                        break;
                    }
                }
                delete[] sourceSiteMapping;
            }
        }
    }
    logger->info("Best horizontal column move can fill {} at a cost of {} per gap", bestFillableGaps, bestCostPerFilledGap);
    return std::tuple(bestMove, bestFillableGaps, bestCostPerFilledGap);
}

std::tuple<std::optional<ParallelMove>,int,double> fillRowThroughSubspace(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>)
{
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestFillableGaps = 0;
    double bestCostPerFilledGap = 0;
    for(bool rowFirst : {true, false})
    {
        if((rowFirst && !ALLOW_MOVES_BETWEEN_ROWS) || (!rowFirst && !ALLOW_MOVES_BETWEEN_COLS))
        {
            continue;
        }
        size_t outerDimCompZone[2];
        size_t innerDimCompZone[2];
        size_t outerSize;
        size_t innerSize;
        unsigned int innerAODLimit = 0;
        unsigned int outerAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, outerDimCompZone, innerDimCompZone, outerSize, innerSize, outerAODLimit, innerAODLimit);

        unsigned int *borderAtomsLeft = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(borderAtomsLeft == 0)
        {
            logger->error("Could not allocate memory");
            return std::tuple(std::nullopt, 0, __DBL_MAX__);
        }
        unsigned int *borderAtomsRight = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(borderAtomsRight == 0)
        {
            logger->error("Could not allocate memory");
            return std::tuple(std::nullopt, 0, __DBL_MAX__);
        }
        unsigned int *emptyCompZoneLocations = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(emptyCompZoneLocations == 0)
        {
            logger->error("Could not allocate memory");
            return std::tuple(std::nullopt, 0, __DBL_MAX__);
        }
        for(size_t i = outerDimCompZone[0]; i < outerDimCompZone[1]; i++)
        {
            for(size_t j = 0; j < innerSize; j++)
            {
                if(accessStateArray(stateArray, i, j, rowFirst))
                {
                    if(j < innerDimCompZone[0])
                    {
                        borderAtomsLeft[i - outerDimCompZone[0]]++;
                    }
                    else if(j >= innerDimCompZone[1])
                    {
                        borderAtomsRight[i - outerDimCompZone[0]]++;
                    }
                }
                else if(j >= innerDimCompZone[0] && j < innerDimCompZone[1])
                {
                    emptyCompZoneLocations[i - outerDimCompZone[0]]++;
                }
            }
        }
        
        for(size_t iBorder = outerDimCompZone[0]; iBorder < outerDimCompZone[1]; iBorder++)
        {
            const unsigned int& currentBorderAtomsLeft = borderAtomsLeft[iBorder - outerDimCompZone[0]];
            const unsigned int& currentBorderAtomsRight = borderAtomsRight[iBorder - outerDimCompZone[0]];
            for(size_t iTarget = outerDimCompZone[0]; iTarget < outerDimCompZone[1]; iTarget++)
            {
                unsigned int outerDist = abs((int)iBorder - (int)iTarget);
                if(outerDist > 0)
                {
                    outerDist--;
                }
                const unsigned int& currentEmptyCompZoneLocations = emptyCompZoneLocations[iTarget - outerDimCompZone[0]];

                unsigned int fillableGaps = (currentBorderAtomsLeft + currentBorderAtomsRight) < currentEmptyCompZoneLocations ? 
                    (currentBorderAtomsLeft + currentBorderAtomsRight) : currentEmptyCompZoneLocations;
                if(fillableGaps > innerAODLimit)
                {
                    fillableGaps = innerAODLimit;
                }
                if(fillableGaps > AOD_TOTAL_LIMIT)
                {
                    fillableGaps = AOD_TOTAL_LIMIT;
                }
                
                double minCost = MOVE_COST_OFFSET + costPerSubMove(outerDist) + costPerSubMove((fillableGaps - 1) / 2) + 2 * HALF_DIAG_STEP_COST;
                if(fillableGaps > 0 && (!bestMove.has_value() || minCost / fillableGaps < bestCostPerFilledGap))
                {
                    unsigned int minFromLeft = fillableGaps < currentBorderAtomsRight ? 0 : fillableGaps - currentBorderAtomsRight;
                    unsigned int maxFromLeft = currentBorderAtomsLeft < fillableGaps ? currentBorderAtomsLeft : fillableGaps;

                    std::optional<ParallelMove> bestMoveInRow = std::nullopt;
                    unsigned int bestDistInRow = 0;

                    unsigned int *distances = new (std::nothrow) unsigned int[fillableGaps];
                    if(distances == 0)
                    {
                        logger->error("Could not allocate memory");
                        return std::tuple(std::nullopt, 0, __DBL_MAX__);
                    }

                    for(unsigned int atomsFromLeft = minFromLeft; atomsFromLeft <= maxFromLeft; atomsFromLeft++)
                    {
                        ParallelMove::Step start;
                        ParallelMove::Step end;
                        if(rowFirst)
                        {
                            start.colSelection.resize(fillableGaps);
                            start.rowSelection.push_back(iBorder);
                            end.colSelection.resize(fillableGaps);
                            end.rowSelection.push_back(iTarget);
                        }
                        else
                        {
                            start.rowSelection.resize(fillableGaps);
                            start.colSelection.push_back(iBorder);
                            end.rowSelection.resize(fillableGaps);
                            end.colSelection.push_back(iTarget);
                        }

                        auto& outerDimStartVector = rowFirst ? start.colSelection : start.rowSelection;
                        auto& outerDimEndVector = rowFirst ? end.colSelection : end.rowSelection;

                        // Fill distances array first with distances from border atoms to compZone[2]
                        unsigned int positionsFound = 0;
                        int currentPosition = innerDimCompZone[0] - 1;
                        for(;currentPosition >= 0 && positionsFound < atomsFromLeft; currentPosition--)
                        {
                            if(accessStateArray(stateArray, iBorder, currentPosition, rowFirst))
                            {
                                distances[atomsFromLeft - positionsFound - 1] = innerDimCompZone[0] - currentPosition;
                                outerDimStartVector[atomsFromLeft - positionsFound - 1] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[1];
                        for(;currentPosition < innerSize && positionsFound < fillableGaps - atomsFromLeft; currentPosition++)
                        {
                            if(accessStateArray(stateArray, iBorder, currentPosition, rowFirst))
                            {
                                distances[atomsFromLeft + positionsFound] = currentPosition - (innerDimCompZone[1] - 1);
                                outerDimStartVector[atomsFromLeft + positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        // Now add distances to gaps (from compZone[2])
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[0];
                        for(;currentPosition < innerDimCompZone[1] && positionsFound < atomsFromLeft; currentPosition++)
                        {
                            if(!accessStateArray(stateArray, iTarget, currentPosition, rowFirst))
                            {
                                distances[positionsFound] += currentPosition - innerDimCompZone[0];
                                outerDimEndVector[positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[1] - 1;
                        for(;currentPosition >= innerDimCompZone[0] && positionsFound < fillableGaps - atomsFromLeft; currentPosition--)
                        {
                            if(!accessStateArray(stateArray, iTarget, currentPosition, rowFirst))
                            {
                                distances[fillableGaps - positionsFound - 1] += innerDimCompZone[1] - 1 - currentPosition;
                                outerDimEndVector[fillableGaps - positionsFound - 1] = (double)currentPosition;
                                positionsFound++;
                            }
                        }

                        unsigned int dist = *std::max_element(distances, distances + fillableGaps);

                        if(!bestMoveInRow.has_value() || dist < bestDistInRow)
                        {
                            bestMoveInRow = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
                            bestDistInRow = dist;
                        }
                    }

                    if(bestMoveInRow.has_value())
                    {
                        double costPerGap = bestMoveInRow.value().cost() / (double)fillableGaps;
                        if(!bestMove.has_value() || costPerGap < bestCostPerFilledGap)
                        {
                            bestMove = bestMoveInRow.value();
                            bestFillableGaps = fillableGaps;
                            bestCostPerFilledGap = costPerGap;
                        }
                    }

                    delete[] distances;
                }
            }
        }
        delete[] borderAtomsLeft;
        delete[] borderAtomsRight;
        delete[] emptyCompZoneLocations;
    }

    logger->info("Best subspace linear move can fill {} at a cost of {} per gap", bestFillableGaps, bestCostPerFilledGap);
    return std::tuple(bestMove, bestFillableGaps, bestCostPerFilledGap);
}

std::tuple<std::optional<ParallelMove>,int,double> fillRowSidesDirectly(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestFillableGaps = 0;
    double bestCostPerFilledGap = 0;
    for(bool rowFirst : {true, false})
    {
        size_t outerDimCompZone[2];
        size_t innerDimCompZone[2];
        size_t outerSize;
        size_t innerSize;
        unsigned int innerAODLimit = 0;
        unsigned int outerAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, outerDimCompZone, innerDimCompZone, outerSize, innerSize, outerAODLimit, innerAODLimit);

        for(size_t i = outerDimCompZone[0]; i < outerDimCompZone[1]; i++)
        {
            unsigned int borderAtomsLeft = 0, borderAtomsRight = 0;
            for(size_t j = 0; j < innerDimCompZone[0]; j++)
            {
                if(accessStateArray(stateArray, i, j, rowFirst))
                {
                    borderAtomsLeft++;
                }
            }
            for(size_t j = innerDimCompZone[1]; j < innerSize; j++)
            {
                if(accessStateArray(stateArray, i, j, rowFirst))
                {
                    borderAtomsRight++;
                }
            }

            size_t leftPosition = innerDimCompZone[0], rightPosition = innerDimCompZone[1] - 1;
            unsigned int aodsUsed = 0, fillableGaps = 0;
            while(leftPosition <= rightPosition && (borderAtomsLeft > 0 || borderAtomsRight > 0))
            {
                std::optional<unsigned int> lowestDistToNextGap = std::nullopt;
                bool fromLeft = true;
                size_t newPosition = 0;
                if(borderAtomsLeft > 0)
                {
                    for(size_t j = leftPosition; j <= rightPosition; j++)
                    {
                        if(!accessStateArray(stateArray, i, j, rowFirst))
                        {
                            lowestDistToNextGap = j - leftPosition + 1;
                            newPosition = j + 1;
                            break;
                        }
                    }
                }
                if(borderAtomsRight > 0)
                {
                    for(size_t j = rightPosition; j >= leftPosition; j--)
                    {
                        if(!accessStateArray(stateArray, i, j, rowFirst))
                        {
                            if(!lowestDistToNextGap.has_value() || (rightPosition - j < lowestDistToNextGap.value()))
                            {
                                lowestDistToNextGap = rightPosition - j + 1;
                                fromLeft = false;
                                newPosition = j - 1;
                            }
                            break;
                        }
                    }
                }

                if(!lowestDistToNextGap.has_value() || (aodsUsed + lowestDistToNextGap.value() > AOD_TOTAL_LIMIT) || 
                    (aodsUsed + lowestDistToNextGap.value() > innerAODLimit))
                {
                    break;
                }
                else
                {
                    fillableGaps++;
                    aodsUsed += lowestDistToNextGap.value();
                    if(fromLeft)
                    {
                        leftPosition = newPosition;
                        borderAtomsLeft--;
                    }
                    else
                    {
                        rightPosition = newPosition;
                        borderAtomsRight--;
                    }
                }
            }
            double minCost = MOVE_COST_OFFSET + costPerSubMove((fillableGaps + 1) / 2);
            if(!bestMove.has_value() || (minCost / fillableGaps < bestCostPerFilledGap))
            {
                ParallelMove::Step start;
                ParallelMove::Step end;
                int atomsFromLeft = leftPosition - innerDimCompZone[0];
                int atomsLeftRemaining = atomsFromLeft;
                int atomsFromRight = innerDimCompZone[1] - 1 - rightPosition;
                int atomsRightRemaining = atomsFromRight;

                if(rowFirst)
                {
                    start.colSelection.resize(atomsFromLeft + atomsFromRight);
                    start.rowSelection.push_back(i);
                    end.colSelection.resize(atomsFromLeft + atomsFromRight);
                    end.rowSelection.push_back(i);
                }
                else
                {
                    start.rowSelection.resize(atomsFromLeft + atomsFromRight);
                    start.colSelection.push_back(i);
                    end.rowSelection.resize(atomsFromLeft + atomsFromRight);
                    end.colSelection.push_back(i);
                }

                auto& outerDimStartVector = rowFirst ? start.colSelection : start.rowSelection;
                auto& outerDimEndVector = rowFirst ? end.colSelection : end.rowSelection;

                int j = leftPosition - 1;
                for(; j >= 0 && atomsLeftRemaining > 0; j--)
                {
                    if(j >= innerDimCompZone[0])
                    {
                        outerDimEndVector[j - innerDimCompZone[0]] = j;
                    }
                    if(accessStateArray(stateArray, i, j, rowFirst))
                    {
                        atomsLeftRemaining--;
                        outerDimStartVector[atomsLeftRemaining] = j;
                    }
                }
                int distanceLeft = (int)innerDimCompZone[0] - (int)j;
                j = rightPosition + 1;
                for(; j < innerSize && atomsRightRemaining > 0; j++)
                {
                    if(j < innerDimCompZone[1])
                    {
                        outerDimEndVector[atomsFromLeft + atomsFromRight - (innerDimCompZone[1] - j)] = j;
                    }
                    if(accessStateArray(stateArray, i, j, rowFirst))
                    {
                        atomsRightRemaining--;
                        outerDimStartVector[atomsFromLeft + atomsFromRight - atomsRightRemaining - 1] = j;
                    }
                }
                int distanceRight = (int)j - (int)(innerDimCompZone[1] - 1);
                if(atomsLeftRemaining > 0 || atomsRightRemaining > 0 || distanceLeft < 0 || distanceRight < 0)
                {
                    logger->error("This point in fillRowSidesDirectly should never be reached!");
                    continue;
                }

                if(!ALLOW_MULTIPLE_MOVES_PER_ATOM && alreadyMoved.has_value())
                {
                    bool moveAllowed = true;
                    for(size_t row : start.rowSelection)
                    {
                        for(size_t col : start.colSelection)
                        {
                            if(alreadyMoved.value()(row,col))
                            {
                                moveAllowed = false;
                                break;
                            }
                        }
                        if(!moveAllowed)
                        {
                            break;
                        }
                    }
                    if(!moveAllowed)
                    {
                        continue;
                    }
                }
                
                ParallelMove move;
                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(end));
                double costPerGap = move.cost() / (double)fillableGaps;
                if(!bestMove.has_value() || costPerGap < bestCostPerFilledGap)
                {
                    bestMove = move;
                    bestFillableGaps = fillableGaps;
                    bestCostPerFilledGap = costPerGap;
                }
            }
        }
    }

    logger->info("Best direct move can fill {} at a cost of {} per gap", bestFillableGaps, bestCostPerFilledGap);
    return std::tuple(bestMove, bestFillableGaps, bestCostPerFilledGap);
}

bool analyzeArray(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], unsigned int& vacancies, std::shared_ptr<spdlog::logger> logger, bool printArray)
{
    vacancies = 0;
    for(size_t row = 0; row < (size_t)stateArray.rows(); row++)
    {
        for(size_t col = 0; col < (size_t)stateArray.cols(); col++)
        {
            if(!stateArray(row,col) && row >= compZone[0] && row < compZone[1] && col >= compZone[2] && col < compZone[3])
            {
                vacancies++;
            }
        }
    }
    if(printArray)
    {
        std::stringstream arrayString;
        arrayString << stateArray;
        logger->debug("\n{}", arrayString.str());
    }
    logger->debug("{} vacancies (analyzeArray)", vacancies);
    return true;
}

bool findNextMove(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZone[4], std::vector<ParallelMove>& moves, bool& sorted, unsigned int& vacancies, std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved)
{
    logger->debug("Finding next best move");
    std::optional<ParallelMove> bestMove = std::nullopt;
    double bestCostPerFilledSites = 0;
    int bestFilledSites = 0;
    for(const auto& function : {fillRowSidesDirectly, fillRowThroughSubspace, fillColumnHorizontally, moveSeveralRowsAndCols})
    {
        auto [move, filledSites, costPerFilledGap] = function(stateArray, compZone, logger, alreadyMoved);
        if(move.has_value() && filledSites > 0)
        {
            if(!bestMove.has_value() || costPerFilledGap < bestCostPerFilledSites)
            {
                bestCostPerFilledSites = costPerFilledGap;
                bestFilledSites = filledSites;
                bestMove = move.value();
            }
        }
    }
    if(!bestMove.has_value())
    {
        logger->error("Finding next best move");
        return false;
    }
    else
    {
        for(const auto& improvingFunction : {improveComplexMove, improveMoveByAddingIndependentAtom})
        {
            auto [improvedMove, improvedFilledVacancies, improvedCostPerVacancy] = improvingFunction(stateArray, 
                compZone, logger, bestMove.value(), bestMove.value().cost(), bestFilledSites, alreadyMoved);
            if(improvedMove.has_value())
            {
                bestMove = improvedMove;
                bestFilledSites = improvedFilledVacancies;
            }
        }

        if(!bestMove.value().execute(stateArray, logger, alreadyMoved))
        {
            logger->error("Error when executing move");
            return false;
        }
        else
        {
            vacancies -= bestFilledSites;
            moves.push_back(std::move(bestMove.value()));
        }
    }
    logger->debug("{} vacancies", vacancies);
    if(vacancies == 0)
    {
        sorted = true;
        logger->debug("Sorting done");
    }
    return true;
}

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd)
{
    std::shared_ptr<spdlog::logger> logger;
    Config& config = Config::getInstance();
    if((logger = spdlog::get(config.parallelLoggerName)) == nullptr)
    {
        logger = spdlog::basic_logger_mt(config.parallelLoggerName, config.logFileName);
    }

    std::optional<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> alreadyMoved = std::nullopt;
    if(!ALLOW_MULTIPLE_MOVES_PER_ATOM)
    {
        alreadyMoved = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(
            stateArray.rows(), stateArray.cols(), false);
    }

    size_t compZone[4] = {compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd};
    std::vector<ParallelMove> moves;
    if(compZoneRowStart >= compZoneRowEnd || compZoneRowEnd > (size_t)stateArray.rows() || 
        compZoneColStart >= compZoneColEnd || compZoneColEnd > (size_t)stateArray.cols())
    {
        logger->error("No suitable arguments");
        return std::nullopt;
    }
    unsigned int vacancies = 0;
    analyzeArray(stateArray, compZone, vacancies, logger, true);
    bool sorted = false;
    while(!sorted)
    {
        if(alreadyMoved.has_value())
        {
            if(!findNextMove(stateArray, compZone, moves, sorted, vacancies, logger, alreadyMoved.value()))
            {
                logger->error("No move could be found");
                analyzeArray(stateArray, compZone, vacancies, logger, true);
                return std::nullopt;
            }
        }
        else
        {
            if(!findNextMove(stateArray, compZone, moves, sorted, vacancies, logger, std::nullopt))
            {
                logger->error("No move could be found");
                analyzeArray(stateArray, compZone, vacancies, logger, true);
                return std::nullopt;
            }
        }
    }
    return moves;
}
