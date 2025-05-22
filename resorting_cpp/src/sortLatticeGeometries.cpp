#include "sortLatticeGeometries.hpp"
#include "sortParallel.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <set>

#include "config.hpp"
#include "spdlog/sinks/basic_file_sink.h"

struct compareOnlyTones
{
    bool operator()(const std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>& lhs, 
        const std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>& rhs) const
    {
        // https://en.cppreference.com/w/cpp/utility/tuple/operator_cmp
        if (std::get<0>(lhs) < std::get<0>(rhs)) return true;
        if (std::get<0>(rhs) < std::get<0>(lhs)) return false;
        return std::get<1>(lhs) < std::get<1>(rhs);
    }
};

double pythagorasDist(double d1, double d2)
{
    return sqrt(d1 * d1 + d2 * d2);
}

double moveCost(const std::vector<std::tuple<bool,size_t,int>>& path)
{
    double cost = MOVE_COST_OFFSET;
    bool verticalSegment = false;
    std::map<size_t,double> segmentDist;
    bool first = true;
    for(const auto& [vertical, index, dist] : path)
    {
        if(first)
        {
            first = false;
            verticalSegment = vertical;
        }
        else if(vertical != verticalSegment)
        {
            for(auto& v : segmentDist)
            {
                v.second = abs(v.second);
            }
            cost += costPerSubMove(std::max_element(segmentDist.begin(), segmentDist.end(), 
                [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->second);
            verticalSegment = vertical;
            segmentDist.clear();
        }
        segmentDist[index] += (double)dist / 2 * (vertical ? ROW_SPACING : COL_SPACING); // / 2 Because the path is expressed in units of half steps
    }
    for(auto& v : segmentDist)
    {
        v.second = abs(v.second);
    }
    cost += costPerSubMove(std::max_element(segmentDist.begin(), segmentDist.end(), 
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->second);
    return cost;
}

Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> generateMask(double distance, double spacingFraction = 1)
{
    int maskRowDist = distance / (ROW_SPACING * spacingFraction);
    int maskRows = 2 * maskRowDist + 1;
    int maskColDist = distance / (COL_SPACING * spacingFraction);
    int maskCols = 2 * maskColDist + 1;
    Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> mask(maskRows, maskCols);
    
    for(int r = 0; r < maskRows; r++)
    {
        for(int c = 0; c < maskCols; c++)
        {
            mask(r,c) = pythagorasDist((r - maskRowDist) * (ROW_SPACING * spacingFraction), 
                (c - maskColDist) * (COL_SPACING * spacingFraction)) < distance;
        }
    }

    return mask;
}

void removeUnusableAtomsFromList(const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const ParallelMove& move, std::set<std::tuple<size_t,size_t>>& unusableAtoms)
{
    for(size_t rIndex = 0; rIndex < move.steps[0].rowSelection.size(); rIndex++)
    {
        for(size_t cIndex = 0; cIndex < move.steps[0].colSelection.size(); cIndex++)
        {
            if(unusableAtoms.erase(std::tuple((int)(move.steps[0].rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD), 
                (int)(move.steps[0].colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD))))
            {
                int newRow = move.steps.back().rowSelection[rIndex];
                int newCol = move.steps.back().colSelection[cIndex];
                if(newRow >= 0 && newRow < stateArray.rows() && newCol >= 0 && newCol < stateArray.cols())
                {
                    unusableAtoms.insert(std::tuple(newRow, newCol));
                }
            }
        }
    }
}

Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> generatePathway(size_t borderRows, size_t borderCols, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &occupancy,
    double distFromOcc = RECOMMENDED_DIST_FROM_OCC_SITES, double distFromEmpty = RECOMMENDED_DIST_FROM_EMPTY_SITES)
{
    size_t pathwayRows = 2 * occupancy.rows() - 1 + 2 * borderRows;
    size_t pathwayCols = 2 * occupancy.cols() - 1 + 2 * borderCols;

    Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> pathway = 
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathwayRows, pathwayCols);

    auto occMask = generateMask(distFromOcc, 0.5);
    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;
    auto emptyMask = generateMask(distFromEmpty, 0.5);
    Eigen::Index halfEmptyRows = emptyMask.rows() / 2;
    Eigen::Index halfEmptyCols = emptyMask.cols() / 2;

    for(size_t r = 0; r < (size_t)occupancy.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)occupancy.cols(); c++)
        {
            if(occupancy(r,c))
            {
                pathway(Eigen::seqN(2 * r + borderRows - halfOccRows, occMask.rows()), 
                    Eigen::seqN(2 * c + borderCols - halfOccCols, occMask.cols())) += occMask.cast<unsigned int>();
            }
            else
            {
                pathway(Eigen::seqN(2 * r + borderRows - halfEmptyRows, emptyMask.rows()), 
                    Eigen::seqN(2 * c + borderCols - halfEmptyCols, emptyMask.cols())) += emptyMask.cast<unsigned int>();
            }
        }
    }

    return pathway;
}

std::tuple<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>, unsigned int> labelPathway(
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& pathway)
{
    Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> labelledPathway = 
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathway.rows(), pathway.cols());
    
    unsigned int pathwayIndex = 1;
    std::vector<std::tuple<Eigen::Index,Eigen::Index>> unvisitedLocs;
    for(Eigen::Index r = 0; r < pathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < pathway.cols(); c++)
        {
            if(pathway(r, c) == 0 && labelledPathway(r, c) == 0)
            {
                unvisitedLocs.push_back(std::tuple(r,c));
                while(!unvisitedLocs.empty())
                {
                    const auto [rP, cP] = unvisitedLocs.back();
                    unvisitedLocs.pop_back();
                    labelledPathway(rP, cP) = pathwayIndex;

                    for(int dir = 0; dir < 4; dir++)
                    {
                        int shiftedRow = dir % 2 == 0 ? rP + dir - 1 : rP;
                        int shiftedCol = dir % 2 == 1 ? cP + dir - 2 : cP;
                        if(shiftedRow >= 0 && shiftedRow < pathway.rows() && shiftedCol >= 0 && 
                            shiftedCol < pathway.cols() && pathway(shiftedRow, shiftedCol) == 0 && 
                            labelledPathway(shiftedRow, shiftedCol) == 0)
                        {
                            unvisitedLocs.push_back(std::tuple(shiftedRow,shiftedCol));
                        }
                    }
                }
                pathwayIndex++;
            }
        }
    }
    return std::tuple(labelledPathway, pathwayIndex - 1);
}

std::vector<std::set<size_t>> findTargetSitesPerPathway(
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway, 
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t borderRows, size_t borderCols, unsigned int labelCount,  size_t compZoneRowStart, size_t compZoneRowEnd,
    size_t compZoneColStart, size_t compZoneColEnd,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::shared_ptr<spdlog::logger> logger)
{
    auto [labelledPenalizedPathway, pLabelCount] = labelPathway(penalizedPathway);
    std::vector<std::set<size_t>> targetSitesPerPPathway;
    targetSitesPerPPathway.resize(pLabelCount);

    std::vector<std::set<unsigned int>> pathwaysPerPPathway;
    pathwaysPerPPathway.resize(pLabelCount);

    for(size_t r = 0; r < 2 * (size_t)stateArray.rows() - 1; r++)
    {
        Eigen::Index pathwayRowIndex = r + borderRows;
        for(size_t c = 0; c < 2 * (size_t)stateArray.cols() - 1; c++)
        {
            Eigen::Index pathwayColIndex = c + borderRows;
            auto pPathwayIndex = labelledPenalizedPathway(pathwayRowIndex, pathwayColIndex);
            if(pPathwayIndex != 0)
            {
                if(r % 2 == 0 && c % 2 == 0 && r / 2 >= compZoneRowStart && r / 2 < compZoneRowEnd && 
                    c / 2 >= compZoneColEnd && c / 2 < compZoneColEnd && 
                    targetGeometry(r / 2 - compZoneRowStart, c / 2 - compZoneColStart) &&
                    !stateArray(r / 2, c / 2) && usableTargetSites.contains(std::tuple(r / 2, c / 2)))
                {
                    targetSitesPerPPathway[pPathwayIndex - 1].insert(
                        (r / 2 - compZoneRowStart) * targetGeometry.cols() + c / 2 - compZoneColStart);
                }
                auto pathwayIndex = labelledPathway(pathwayRowIndex, pathwayColIndex);
                if(pathwayIndex != 0)
                {
                    pathwaysPerPPathway[pPathwayIndex - 1].insert(pathwayIndex);
                }
            }
        }
    }

    std::vector<std::set<size_t>> targetSitesPerPathway;
    targetSitesPerPathway.resize(labelCount);
    for(size_t pPathwayIndex = 0; pPathwayIndex < pLabelCount; pPathwayIndex++)
    {
        for(const auto& pathwayIndex : pathwaysPerPPathway[pPathwayIndex])
        {
            targetSitesPerPathway[pathwayIndex] = targetSitesPerPPathway[pPathwayIndex];
        }
    }

    return targetSitesPerPathway;
}

bool isDirectMove(std::set<std::tuple<size_t,size_t>> rows, std::set<std::tuple<size_t,size_t>> cols, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int maxRowDist = 0;
    unsigned int maxColDist = 0;
    for(const auto& [start,end] : rows)
    {
        unsigned int dist = abs((int)end - (int)start);
        if(dist > maxRowDist)
        {
            maxRowDist = dist;
        }
    }
    for(const auto& [start,end] : cols)
    {
        unsigned int dist = abs((int)end - (int)start);
        if(dist > maxColDist)
        {
            maxColDist = dist;
        }
    }
    return (maxRowDist == 0 && maxColDist <= 1) || (maxRowDist <= 1 && maxColDist == 0);
}

std::optional<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>,double>> checkMoveValidity(
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& penalizedPathway,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms,  size_t compZoneRowStart, size_t compZoneColStart, 
    size_t borderRows, size_t borderCols, std::set<std::tuple<size_t,size_t>> cols, std::set<std::tuple<size_t,size_t>> rows,
    double bestFilledPerCost, std::shared_ptr<spdlog::logger> logger)
{
    // Stores possible states after each movement step
    // A state contains the vector of row and col tones
    // Also it contains the list of moves that have been performed
    std::vector<std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,
        std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>> possibleTonesAfterStep;
    std::vector<size_t> initialRows;
    std::vector<size_t> initialCols;
    std::vector<size_t> targetRows;
    std::vector<size_t> targetCols;
    for(const auto& [startRow, endRow] : rows)
    {
        initialRows.push_back(2 * startRow);
        targetRows.push_back(2 * endRow);
    }
    for(const auto& [startCol, endCol] : cols)
    {
        initialCols.push_back(2 * startCol);
        targetCols.push_back(2 * endCol);
    }
    if(isDirectMove(rows, cols, logger))
    {
        logger->debug("Direct move possible");
        std::vector<std::tuple<bool,size_t,int>> path;
        if(targetRows != initialRows)
        {
            for(size_t i = 0; i < targetRows.size(); i++)
            {
                int diff = (int)targetRows[i] - (int)initialRows[i];
                if(diff != 0)
                {
                    path.push_back(std::tuple(true, i, diff));
                }
            }
        }
        if(targetCols != initialCols)
        {
            for(size_t i = 0; i < targetCols.size(); i++)
            {
                int diff = (int)targetCols[i] - (int)initialCols[i];
                if(diff != 0)
                {
                    path.push_back(std::tuple(false, i, diff));
                }
            }
        }
        return std::tuple(initialRows, initialCols, path, (double)(initialRows.size() * initialCols.size()) / moveCost(path));
    }
    std::optional<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>,double>> bestSubsetMove = std::nullopt;

    std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>> 
        targetState(targetRows, targetCols, std::vector<std::tuple<bool,size_t,int>>());
    possibleTonesAfterStep.push_back(std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>());
    possibleTonesAfterStep[0].insert(std::tuple(initialRows, initialCols, std::vector<std::tuple<bool,size_t,int>>()));
    for(size_t steps = 0; steps < possibleTonesAfterStep.size(); steps++)
    {
        logger->debug("{} possible locations after {} steps", possibleTonesAfterStep[steps].size(), steps);
        if(possibleTonesAfterStep[steps].empty())
        {
            return bestSubsetMove;
        }
        possibleTonesAfterStep.push_back(std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>());
        for(const auto& [rowState, colState, path] : possibleTonesAfterStep[steps])
        {
            for(bool vertical : {true, false})
            {
                const auto& changingState = vertical ? rowState : colState;
                const auto& otherState = vertical ? colState : rowState;
                for(size_t toneIndex = 0; toneIndex < changingState.size(); toneIndex++)
                {
                    if(path.size() == 0 || std::get<0>(path.back()) != vertical || std::get<1>(path.back()) != toneIndex)
                    {
                        size_t tone = changingState[toneIndex];
                        for(int dir : {-1, 1})
                        {
                            bool movedIntoPenalizedPathway = false;
                            int maxDistIntoPenalizedPathway = vertical ? MAX_SUBMOVE_DIST_IN_PENALIZED_AREA / HALF_ROW_SPACING : 
                                (MAX_SUBMOVE_DIST_IN_PENALIZED_AREA / HALF_COL_SPACING);
                            for(int dist = 1;;dist++)
                            {
                                bool propagateState = true;
                                bool stopAfterChecking = false;
                                bool stopImmediately = false;
                                for(size_t otherToneIndex = 0; otherToneIndex < otherState.size(); otherToneIndex++)
                                {
                                    size_t otherTone = otherState[otherToneIndex];
                                    int r = vertical ? ((int)tone + dir * dist) : otherTone;
                                    int c = vertical ? otherTone : ((int)tone + dir * dist);
                                    if(r < 0 || r > 2 * (stateArray.rows() - 1) || c < 0 || c > 2 * (stateArray.cols() - 1))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if(dist % 2 == 0 && stateArray(r / 2, c / 2))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if((toneIndex > 0 && ((int)tone + dir * dist) <= changingState[toneIndex - 1]) ||
                                        (toneIndex < (changingState.size() - 1) && ((int)tone + dir * dist) >= changingState[toneIndex + 1]))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if(labelledPathway(borderRows + r, borderCols + c) == 0)
                                    {
                                        if(penalizedPathway(borderRows + r, borderCols + c) <= 
                                            (pythagorasDist((r - (int)initialRows[vertical ? toneIndex : otherToneIndex]) * ROW_SPACING, 
                                            (c - (int)initialCols[vertical ? otherToneIndex : toneIndex]) * COL_SPACING) < 2 * MIN_DIST_FROM_OCC_SITES ? 1 : 0))
                                        {
                                            movedIntoPenalizedPathway = true;
                                        }
                                        else
                                        {
                                            stopImmediately = true;
                                        }
                                    }
                                }
                                std::vector<size_t> newRowState = rowState;
                                std::vector<size_t> newColState = colState;
                                if(vertical)
                                {
                                    newRowState[toneIndex] += dir * dist;
                                }
                                else
                                {
                                    newColState[toneIndex] += dir * dist;
                                }
                                if(movedIntoPenalizedPathway)
                                {
                                    bool toneWasLastMoved = false;
                                    for(auto step = path.rbegin(); step != path.rend(); ++step)
                                    {
                                        if(std::get<0>(*step) != vertical)
                                        {
                                            break;
                                        }
                                        else if(std::get<1>(*step) == toneIndex)
                                        {
                                            toneWasLastMoved = true;
                                            break;
                                        }
                                    }
                                    if(toneWasLastMoved)
                                    {
                                        stopImmediately = true;
                                    }
                                    else
                                    {
                                        maxDistIntoPenalizedPathway--;
                                        if(maxDistIntoPenalizedPathway < 0)
                                        {
                                            stopImmediately = true;
                                            break;
                                        }
                                    }
                                }
                                if(stopImmediately)
                                {
                                    break;
                                }
                                std::vector<std::tuple<bool,size_t,int>> newPath = path;
                                newPath.push_back(std::tuple(vertical, toneIndex, dir * dist));
                                std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>> 
                                    newState(newRowState, newColState, newPath);
                                double newCost = moveCost(newPath);
                                std::vector<size_t> rowIntersectionIndices;
                                for(size_t i = 0; i < targetRows.size(); i++)
                                {
                                    if(newRowState[i] == targetRows[i])
                                    {
                                        rowIntersectionIndices.push_back(i);
                                    }
                                }
                                std::vector<size_t> colIntersectionIndices;
                                for(size_t i = 0; i < targetCols.size(); i++)
                                {
                                    if(newColState[i] == targetCols[i])
                                    {
                                        colIntersectionIndices.push_back(i);
                                    }
                                }
                                double filledPerCost = (rowIntersectionIndices.size() * colIntersectionIndices.size()) / newCost;
                                if(filledPerCost > bestFilledPerCost && filledPerCost > DOUBLE_EQUIVALENCE_THRESHOLD && (!bestSubsetMove.has_value() || filledPerCost > std::get<3>(bestSubsetMove.value())))
                                {
                                    logger->debug("Writing new best subset move with filledPerCost {}", filledPerCost);
                                    std::vector<std::tuple<bool,size_t,int>> subsetPath;
                                    for(const auto& [vertical, index, dist] : newPath)
                                    {
                                        std::vector<size_t> *intersectionIndices;
                                        if(vertical)
                                        {
                                            intersectionIndices = &rowIntersectionIndices;
                                        }
                                        else
                                        {
                                            intersectionIndices = &colIntersectionIndices;
                                        }
                                        const auto iter = std::find(intersectionIndices->begin(), intersectionIndices->end(), index);
                                        if(iter != intersectionIndices->end())
                                        {
                                            subsetPath.push_back(std::tuple(vertical, std::distance(intersectionIndices->begin(), iter), dist));
                                        }
                                    }
                                    std::vector<size_t> initialRowsSubset;
                                    for(const auto& i : rowIntersectionIndices)
                                    {
                                        initialRowsSubset.push_back(initialRows[i]);
                                    }
                                    std::vector<size_t> initialColsSubset;
                                    for(const auto& i : colIntersectionIndices)
                                    {
                                        initialColsSubset.push_back(initialCols[i]);
                                    }
                                    bestSubsetMove = std::tuple(initialRowsSubset, initialColsSubset, subsetPath, filledPerCost);
                                }
                                if(bestSubsetMove.has_value() && targetRows == newRowState && targetCols == newColState)
                                {
                                    logger->debug("Full move possible");
                                    for(const auto& [vertical, index, dist] : std::get<2>(bestSubsetMove.value()))
                                    {
                                        logger->debug("Move {} at index {} by {}", vertical ? "vertically" : "horizontally", index, dist);
                                    }
                                    std::stringstream startIndices;
                                    startIndices << "StartRows: (";
                                    for(const auto& row : std::get<0>(bestSubsetMove.value()))
                                    {
                                        startIndices << row << " ";
                                    }
                                    startIndices << "), startCols: (";
                                    for(const auto& col : std::get<1>(bestSubsetMove.value()))
                                    {
                                        startIndices << col << " ";
                                    }
                                    startIndices << ")";
                                    logger->debug(startIndices.str());
                                    return bestSubsetMove;
                                }
                                if(propagateState)
                                {
                                    double bestPossibleFilledPerCost = (initialRows.size() * initialCols.size()) / newCost;
                                    if(bestPossibleFilledPerCost > bestFilledPerCost && (!bestSubsetMove.has_value() || bestPossibleFilledPerCost > std::get<3>(bestSubsetMove.value())))
                                    {
                                        bool alreadyLookedAt = false;
                                        for(size_t steps = 0; steps < possibleTonesAfterStep.size() - 1; steps++)
                                        {
                                            if(possibleTonesAfterStep[steps].contains(newState))
                                            {
                                                alreadyLookedAt = true;
                                                break;
                                            }
                                        }
                                        if(!alreadyLookedAt)
                                        {
                                            possibleTonesAfterStep[steps + 1].insert(newState);
                                        }
                                    }
                                }
                                if(stopAfterChecking)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return bestSubsetMove;
}

std::optional<std::tuple<ParallelMove,double>> findDistOneMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    const std::vector<std::tuple<size_t,size_t>>& possibleMoves, std::set<std::tuple<size_t,size_t>>& unusableAtoms,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<RowBitMask> rowsToMoveIndexDown;
    std::vector<RowBitMask> rowsToMoveIndexUp;
    std::vector<RowBitMask> colsToMoveIndexDown;
    std::vector<RowBitMask> colsToMoveIndexUp;

    std::vector<RowBitMask> allowedRowsToMoveIndexDown;
    std::vector<RowBitMask> allowedRowsToMoveIndexUp;
    std::vector<RowBitMask> allowedColsToMoveIndexDown;
    std::vector<RowBitMask> allowedColsToMoveIndexUp;

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        colsToMoveIndexDown.push_back(RowBitMask(targetGeometry.cols(), row));
        colsToMoveIndexUp.push_back(RowBitMask(targetGeometry.cols(), row));
        allowedColsToMoveIndexDown.push_back(RowBitMask(targetGeometry.cols(), row));
        allowedColsToMoveIndexUp.push_back(RowBitMask(targetGeometry.cols(), row));
    }
    for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
    {
        rowsToMoveIndexDown.push_back(RowBitMask(targetGeometry.rows(), col));
        rowsToMoveIndexUp.push_back(RowBitMask(targetGeometry.rows(), col));
        allowedRowsToMoveIndexDown.push_back(RowBitMask(targetGeometry.rows(), col));
        allowedRowsToMoveIndexUp.push_back(RowBitMask(targetGeometry.rows(), col));
    }

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(stateArray(row + compZoneRowStart, col + compZoneColStart) && !targetGeometry(row, col) && 
                penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                (MIN_DIST_FROM_OCC_SITES > 0 ? 1 : 0) && !unusableAtoms.contains(std::tuple(row + compZoneRowStart, col + compZoneColStart)))
            {
                if(row > 0 && !stateArray(row - 1 + compZoneRowStart, col + compZoneColStart) && targetGeometry(row - 1, col) && 
                    penalizedPathway(borderRows + 2 * (row - 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                    (MIN_DIST_FROM_OCC_SITES > ROW_SPACING ? 1 : 0))
                {
                    rowsToMoveIndexDown[col].set(row, true);
                }
                if(row < targetGeometry.rows() - 1 && !stateArray(row + 1 + compZoneRowStart, col + compZoneColStart) && targetGeometry(row + 1, col) && 
                    penalizedPathway(borderRows + 2 * (row + 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                    (MIN_DIST_FROM_OCC_SITES > ROW_SPACING ? 1 : 0))
                {
                    rowsToMoveIndexUp[col].set(row, true);
                }
                if(col > 0 && !stateArray(row + compZoneRowStart, col - 1 + compZoneColStart) && targetGeometry(row, col - 1) && 
                    penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col - 1 + compZoneColStart)) <= 
                    (MIN_DIST_FROM_OCC_SITES > COL_SPACING ? 1 : 0))
                {
                    colsToMoveIndexDown[row].set(col, true);
                }
                if(col < targetGeometry.rows() - 1 && !stateArray(row + compZoneRowStart, col + 1 + compZoneColStart) && targetGeometry(row, col + 1) && 
                    penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + 1 + compZoneColStart)) <= 
                    (MIN_DIST_FROM_OCC_SITES > COL_SPACING ? 1 : 0))
                {
                    colsToMoveIndexUp[row].set(col, true);
                }
            }
            if(penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
            {
                if(row > 0 && penalizedPathway(borderRows + 2 * (row - 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
                {
                    allowedRowsToMoveIndexDown[col].set(row, true);
                }
                if(row < targetGeometry.rows() - 1 && penalizedPathway(borderRows + 2 * (row + 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
                {
                    allowedRowsToMoveIndexUp[col].set(row, true);
                }
                if(col > 0 && penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col - 1 + compZoneColStart)) == 0)
                {
                    allowedColsToMoveIndexDown[row].set(col, true);
                }
                if(col < targetGeometry.rows() - 1 && penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + 1 + compZoneColStart)) == 0)
                {
                    allowedColsToMoveIndexUp[row].set(col, true);
                }
            }
        }
    }

    std::vector<std::tuple<RowBitMask,RowBitMask,bool,unsigned int>> bestOverlaps;
    std::vector<RowBitMask> perRowBitMaskIndexDown;
    std::vector<RowBitMask> perRowBitMaskIndexUp;
    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        RowBitMask downBitMask = RowBitMask::fromOr(colsToMoveIndexDown[row], allowedColsToMoveIndexDown[row]);
        downBitMask.indices.pop_back();
        RowBitMask upBitMask = RowBitMask::fromOr(colsToMoveIndexUp[row], allowedColsToMoveIndexUp[row]);
        upBitMask.indices.pop_back();

        perRowBitMaskIndexDown.push_back(downBitMask);
        perRowBitMaskIndexUp.push_back(upBitMask);

        //logger->debug("Per row BitMask bits set at {}: {}/{}; {}/{}", row, colsToMoveIndexDown[row].bitsSet(), downBitMask.bitsSet(), colsToMoveIndexUp[row].bitsSet(), upBitMask.bitsSet());

        unsigned int sum = colsToMoveIndexDown[row].bitsSet() + colsToMoveIndexUp[row].bitsSet();
        if(sum > 0)
        {
            std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(downBitMask), std::move(upBitMask), false, sum);
            auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                    const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                { return std::get<3>(lhs) < std::get<3>(rhs); });
            bestOverlaps.insert(insertLoc, std::move(elem));
        }
    }
    std::vector<RowBitMask> perColBitMaskIndexDown;
    std::vector<RowBitMask> perColBitMaskIndexUp;
    for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
    {
        RowBitMask downBitMask = RowBitMask::fromOr(rowsToMoveIndexDown[col], allowedRowsToMoveIndexDown[col]);
        downBitMask.indices.pop_back();
        RowBitMask upBitMask = RowBitMask::fromOr(rowsToMoveIndexUp[col], allowedRowsToMoveIndexUp[col]);
        upBitMask.indices.pop_back();

        perColBitMaskIndexDown.push_back(downBitMask);
        perColBitMaskIndexUp.push_back(upBitMask);

        //logger->debug("Per col BitMask bits set at {}: {}/{}; {}/{}", col, rowsToMoveIndexDown[col].bitsSet(), downBitMask.bitsSet(), rowsToMoveIndexUp[col].bitsSet(), upBitMask.bitsSet());

        unsigned int sum = rowsToMoveIndexDown[col].bitsSet() + rowsToMoveIndexUp[col].bitsSet();
        if(sum > 0)
        {
            std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(downBitMask), std::move(upBitMask), true, sum);
            auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                    const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                { return std::get<3>(lhs) < std::get<3>(rhs); });
            bestOverlaps.insert(insertLoc, std::move(elem));
        }
    }

    std::optional<std::tuple<RowBitMask,RowBitMask,bool,unsigned int>> bestOverlap = std::nullopt;

    size_t iter = 0;
    while(!bestOverlaps.empty() && iter++ < 1000)
    {
        auto currentElem = bestOverlaps.back();
        auto [downBitMask,upBitMask,vertical,overlap] = currentElem;
        bestOverlaps.pop_back();

        if(!bestOverlap.has_value() || overlap > std::get<3>(bestOverlap.value()))
        {
            logger->debug("Writing new best overlap with {} rows and {} cols", vertical ? downBitMask.bitsSet() + upBitMask.bitsSet() : downBitMask.indices.size(), 
                vertical ? downBitMask.indices.size() : (downBitMask.bitsSet() + upBitMask.bitsSet()));
            bestOverlap.emplace(currentElem);
        }
        std::vector<RowBitMask> *indexDownBitMasks;
        std::vector<RowBitMask> *indexUpBitMasks;

        std::vector<RowBitMask> *usefulIndicesDown;
        std::vector<RowBitMask> *usefulIndicesUp;

        if(vertical)
        {
            indexDownBitMasks = &perColBitMaskIndexDown;
            indexUpBitMasks = &perColBitMaskIndexUp;

            usefulIndicesDown = &rowsToMoveIndexDown;
            usefulIndicesUp = &rowsToMoveIndexUp;
        }
        else
        {
            indexDownBitMasks = &perRowBitMaskIndexDown;
            indexUpBitMasks = &perRowBitMaskIndexUp;

            usefulIndicesDown = &colsToMoveIndexDown;
            usefulIndicesUp = &colsToMoveIndexUp;
        }

        for(size_t newIndex = downBitMask.indices.back() + 1; newIndex < indexDownBitMasks->size(); newIndex++)
        {
            if((*usefulIndicesDown)[newIndex].bitsSet() > 0 || (*usefulIndicesUp)[newIndex].bitsSet() > 0)
            {
                auto newDownBitMask = RowBitMask::fromAnd(downBitMask, (*indexDownBitMasks)[newIndex]);
                auto newUpBitMask = RowBitMask::fromAnd(upBitMask, (*indexUpBitMasks)[newIndex]);
                unsigned int filledSites = 0;
                for(auto index : newDownBitMask.indices)
                {
                    filledSites += RowBitMask::fromAnd(newDownBitMask, (*usefulIndicesDown)[index]).bitsSet();
                    filledSites += RowBitMask::fromAnd(newUpBitMask, (*usefulIndicesUp)[index]).bitsSet();
                }
                if(filledSites > 0 && (newDownBitMask.bitsSet() + newUpBitMask.bitsSet()) > 1)
                {
                    std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(newDownBitMask), 
                        std::move(newUpBitMask), vertical, filledSites);
                    auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                        [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                            const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                        { return std::get<3>(lhs) < std::get<3>(rhs); });
                    bestOverlaps.insert(insertLoc, std::move(elem));
                }
            }
        }
    }
    if(bestOverlap.has_value())
    {
        logger->debug("Best overlap for dist one move: DownCount: {}, UpCount: {}, Verti: {}, Overlap: {}", 
            std::get<0>(bestOverlap.value()).bitsSet(), std::get<1>(bestOverlap.value()).bitsSet(),
            std::get<2>(bestOverlap.value()), std::get<3>(bestOverlap.value()));
        
        ParallelMove move;
        ParallelMove::Step start;
        ParallelMove::Step end;
        int lastTarget = -1;
        unsigned int actualOverlap = 0;
        if(std::get<2>(bestOverlap.value()))
        {
            RowBitMask usefulRowsDown(targetGeometry.rows());
            RowBitMask usefulRowsUp(targetGeometry.rows());

            start.colSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            end.colSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            for(auto& col : std::get<0>(bestOverlap.value()).indices)
            {
                if(RowBitMask::fromAnd(std::get<0>(bestOverlap.value()), rowsToMoveIndexDown[col]).bitsSet() > 0 ||
                    RowBitMask::fromAnd(std::get<1>(bestOverlap.value()), rowsToMoveIndexUp[col]).bitsSet() > 0)
                {
                    usefulRowsDown |= rowsToMoveIndexDown[col];
                    usefulRowsUp |= rowsToMoveIndexUp[col];
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart);
                }
            }
            for(size_t row = 0; row < std::get<0>(bestOverlap.value()).count; row++)
            {
                if(std::get<0>(bestOverlap.value())[row] && usefulRowsDown[row] && row > lastTarget + 1)
                {
                    for(auto& col : start.colSelection)
                    {
                        if(rowsToMoveIndexDown[col - compZoneColStart][row])
                        {
                            actualOverlap++;
                        }
                    }
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart - 1);
                }
                else if(std::get<1>(bestOverlap.value())[row] && usefulRowsUp[row])
                {
                    for(auto& col : start.colSelection)
                    {
                        if(rowsToMoveIndexUp[col - compZoneColStart][row])
                        {
                            actualOverlap++;
                        }
                    }
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart + 1);
                    lastTarget = row + 1;
                }
            }
        }
        else
        {
            RowBitMask usefulColsDown(targetGeometry.cols());
            RowBitMask usefulColsUp(targetGeometry.cols());

            start.rowSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            end.rowSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            for(auto& row : std::get<0>(bestOverlap.value()).indices)
            {
                if(RowBitMask::fromAnd(std::get<0>(bestOverlap.value()), colsToMoveIndexDown[row]).bitsSet() > 0 ||
                    RowBitMask::fromAnd(std::get<1>(bestOverlap.value()), colsToMoveIndexUp[row]).bitsSet() > 0)
                {
                    usefulColsDown |= colsToMoveIndexDown[row];
                    usefulColsUp |= colsToMoveIndexUp[row];
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart);
                }
            }
            for(size_t col = 0; col < std::get<0>(bestOverlap.value()).count; col++)
            {
                if(std::get<0>(bestOverlap.value())[col] && usefulColsDown[col] && col > lastTarget + 1)
                {
                    for(auto& row : start.rowSelection)
                    {
                        if(colsToMoveIndexDown[row - compZoneRowStart][col])
                        {
                            actualOverlap++;
                        }
                    }
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart - 1);
                }
                else if(std::get<1>(bestOverlap.value())[col] && usefulColsUp[col])
                {
                    for(auto& row : start.rowSelection)
                    {
                        if(colsToMoveIndexUp[row - compZoneRowStart][col])
                        {
                            actualOverlap++;
                        }
                    }
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart + 1);
                    lastTarget = col + 1;
                }
            }
        }
        move.steps.push_back(std::move(start));
        move.steps.push_back(std::move(end));

        double filledPerCost = 2 * (double)actualOverlap / move.cost();
        logger->debug("Returning distance one move that sorts {}, filledPerCost: {}", actualOverlap, filledPerCost);
        return std::tuple(move, filledPerCost);
    }
    else
    {
        logger->info("No better distance one move could be found");
        return std::nullopt;
    }
}

std::optional<std::tuple<ParallelMove,double>> findComplexMoveLegacy(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, 
    const std::vector<std::tuple<size_t,size_t>>& possibleMoves, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, double bestIntPerCost, std::shared_ptr<spdlog::logger> logger)
{
    std::optional<std::vector<size_t>> bestMoveStartRows = std::nullopt;
    std::optional<std::vector<size_t>> bestMoveStartCols = std::nullopt;
    std::optional<std::vector<std::tuple<bool,size_t,int>>> bestMovePath = std::nullopt;

    std::vector<std::vector<std::set<std::tuple<size_t,size_t>>>> movableRowsPerTargetColPerSourceCol;
    for(size_t col = 0; col < (size_t)targetGeometry.cols(); col++)
    {
        std::vector<std::set<std::tuple<size_t,size_t>>> movableRowsPerTargetCol;
        for(size_t tCol = 0; tCol < (size_t)targetGeometry.cols(); tCol++)
        {
            movableRowsPerTargetCol.push_back(std::set<std::tuple<size_t,size_t>>());
        }
        movableRowsPerTargetColPerSourceCol.push_back(std::move(movableRowsPerTargetCol));
    }

    for(const auto& [pMStart, pMEnd] : possibleMoves)
    {
        movableRowsPerTargetColPerSourceCol[pMStart % targetGeometry.cols()][pMEnd % targetGeometry.cols()].insert(
            std::tuple(pMStart / targetGeometry.cols() + compZoneRowStart, pMEnd / targetGeometry.cols() + compZoneRowStart));
    }

    for(size_t col = 0; col < (size_t)targetGeometry.cols(); col++)
    {
        for(size_t tCol = 0; tCol < (size_t)targetGeometry.cols(); tCol++)
        {
            double minCost = MOVE_COST_OFFSET + costPerSubMove(abs((int)tCol - (int)col) * COL_SPACING);
            if(movableRowsPerTargetColPerSourceCol[col][tCol].size() > 0 && (movableRowsPerTargetColPerSourceCol[col][tCol].size() / minCost) > bestIntPerCost)
            {
                std::vector<std::tuple<double,std::tuple<size_t,size_t>>> elemDist;
                for(size_t i = 0; i < movableRowsPerTargetColPerSourceCol[col][tCol].size(); i++)
                {
                    const auto& [startRow, endRow] = *std::next(movableRowsPerTargetColPerSourceCol[col][tCol].begin(), i);
                    std::tuple<double,std::tuple<size_t,size_t>> elem = std::tuple(abs((int)endRow - (int)startRow), std::tuple(startRow,endRow));
                    if(std::find_if(elemDist.begin(), elemDist.end(), [&startRow, &endRow](auto const& v) {
                        return (std::get<0>(std::get<1>(v)) == startRow) || (std::get<1>(std::get<1>(v)) == endRow);}) == elemDist.end())
                    {
                        elemDist.insert(std::upper_bound(elemDist.begin(), elemDist.end(), elem, 
                            [](const std::tuple<double,std::tuple<size_t,size_t>>& lhs, const std::tuple<double,std::tuple<size_t,size_t>>& rhs)
                            { return std::get<0>(lhs) < std::get<0>(rhs); }), elem);
                    }
                }
                double intPerCost = -1;
                std::optional<std::set<std::tuple<size_t,size_t>>> bestSubset = std::nullopt;
                std::set<std::tuple<size_t,size_t>> currentSubset;
                for(size_t usedRowCount = 0; usedRowCount < elemDist.size(); usedRowCount++)
                {
                    currentSubset.insert(std::get<1>(elemDist[usedRowCount]));
                    double localCountPerCost = (usedRowCount + 1) / (minCost + costPerSubMove(std::get<0>(elemDist[usedRowCount]) * ROW_SPACING));
                    if(localCountPerCost > intPerCost)
                    {
                        intPerCost = localCountPerCost;
                        bestSubset = currentSubset;
                    }
                }
                if(bestSubset.has_value() && intPerCost > bestIntPerCost)
                {
                    std::stringstream checkingMoveStream;
                    checkingMoveStream << "Checking move col " << col << " -> " << tCol << " with rows (";
                    for(const auto& [startRow,endRow] : bestSubset.value())
                    {
                        checkingMoveStream << startRow << " ";
                    }
                    checkingMoveStream << ") -> (";
                    for(const auto& [startRow,endRow] : bestSubset.value())
                    {
                        checkingMoveStream << endRow << " ";
                    }
                    checkingMoveStream << ")";
                    logger->debug(checkingMoveStream.str());
                    std::set<std::tuple<size_t,size_t>> cols;
                    cols.insert(std::tuple(col + compZoneColStart, tCol + compZoneColStart));
                    auto move = checkMoveValidity(stateArray, labelledPathway, penalizedPathway, unusableAtoms, compZoneRowStart, compZoneColStart, 
                        borderRows, borderCols, cols, bestSubset.value(), bestIntPerCost, logger);
                    if(move.has_value())
                    {
                        logger->debug("Returned move with {} rows and {} cols", std::get<0>(move.value()).size(), std::get<1>(move.value()).size());
                        if(std::get<3>(move.value()) > bestIntPerCost)
                        {
                            std::tie(bestMoveStartRows, bestMoveStartCols, bestMovePath, bestIntPerCost) = move.value();
                        }
                    }
                }
            }
        }
    }
    if(bestMoveStartRows.has_value() && bestMoveStartCols.has_value() && bestMovePath.has_value())
    {
        ParallelMove move;
        ParallelMove::Step step;
        for(const auto& row : bestMoveStartRows.value())
        {
            step.rowSelection.push_back((double)row / 2);
        }
        for(const auto& col : bestMoveStartCols.value())
        {
            step.colSelection.push_back((double)col / 2);
        }
        move.steps.push_back(step);
        bool verticalSegment = false;
        std::map<size_t,double> segmentDist;
        bool first = true;
        for(const auto& [vertical, index, dist] : bestMovePath.value())
        {
            if(first)
            {
                first = false;
                verticalSegment = vertical;
            }
            else if(vertical != verticalSegment)
            {
                std::vector<double> *changingSelection = &(verticalSegment ? step.rowSelection : step.colSelection);
                for(const auto& [segIndex, segDist] : segmentDist)
                {
                    (*changingSelection)[segIndex] += segDist;
                }
                move.steps.push_back(step);

                verticalSegment = vertical;
                segmentDist.clear();
            }
            segmentDist[index] += (double)dist / 2; // / 2 Because the path is expressed in units of half steps
        }
        std::vector<double> *changingSelection = &(verticalSegment ? step.rowSelection : step.colSelection);
        for(const auto& [segIndex, segDist] : segmentDist)
        {
            (*changingSelection)[segIndex] += segDist;
        }
        move.steps.push_back(step);

        logger->info("Returning complex move sorts {} rows and {} cols within computational zone, filledPerCost: {}", 
            move.steps[0].rowSelection.size(), move.steps[0].colSelection.size(), 2 * bestIntPerCost);
        return std::tuple(move, 2 * bestIntPerCost);
    }
    else
    {
        logger->info("No better complex move could be found");
        return std::nullopt;
    }
}

std::optional<std::tuple<ParallelMove,double>> findComplexMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& pathway,
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    const std::vector<std::tuple<size_t,size_t>>& possibleMoves, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, double bestIntPerCost, std::shared_ptr<spdlog::logger> logger)
{
    double bestLocalIntPerCost = 0;
    std::optional<ParallelMove> bestMove = std::nullopt;

    for(bool vertical : {true,false})
    {
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> byIPathways = 
            Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathway.rows(), pathway.cols());
        unsigned int pathwayCount = 0;
        Eigen::Index outerDimSize = vertical ? stateArray.cols() : stateArray.rows();
        size_t outerMaxIndex = 2 * outerDimSize - 1;
        size_t innerMaxIndex = vertical ? (2 * stateArray.rows() - 1) : (2 * stateArray.cols() - 1);
        std::vector<std::vector<unsigned int>> pathwaysPerIndex;
        pathwaysPerIndex.resize(outerMaxIndex);
        for(size_t i = 0; i < outerMaxIndex; i++)
        {
            std::vector<std::tuple<size_t,size_t>> startsAndEnds;
            bool inPathway = false;
            unsigned int pathwayLength = 0;
            for(size_t j = 0; j < innerMaxIndex; j++)
            {
                Eigen::Index pathwayRow = borderRows + (vertical ? j : i);
                Eigen::Index pathwayCol = borderCols + (vertical ? i : j);
                if(pathway(pathwayRow, pathwayCol) == 0)
                {
                    if(!inPathway)
                    {
                        pathwayLength = 1;
                        inPathway = true;
                    }
                    else
                    {
                        pathwayLength++;
                    }
                }
                else
                {
                    if(inPathway)
                    {
                        if(pathwayLength > 1)
                        {
                            pathwayCount++;
                            for(size_t jBackwards = 1; jBackwards <= pathwayLength; jBackwards++)
                            {
                                Eigen::Index pathwayRowBackwards = borderRows + (vertical ? j - jBackwards : i);
                                Eigen::Index pathwayColBackwards = borderCols + (vertical ? i : j - jBackwards);
                                byIPathways(pathwayRowBackwards, pathwayColBackwards) = pathwayCount;
                            }
                            pathwaysPerIndex[i].push_back(pathwayCount);
                        }
                        inPathway = false;
                    }
                }
            }
        }

        std::vector<std::vector<std::tuple<size_t,size_t>>> reachableTargetSitesPerPathway;
        reachableTargetSitesPerPathway.resize(pathwayCount);

        for(size_t row = 0; row < 2 * stateArray.rows() - 1; row++)
        {
            if(!vertical || (row >= 2 * compZoneRowStart && row <= 2 * (compZoneRowEnd - 1)))
            {
                for(size_t col = 0; col < 2 * stateArray.cols() - 1; col++)
                {
                    if(vertical || (col >= 2 * compZoneColStart && col <= 2 * (compZoneColEnd - 1)))
                    {
                        unsigned int iPathway = byIPathways(borderRows + row, borderCols + col);
                        if(iPathway != 0)
                        {
                            for(bool positive : {true, false})
                            {
                                for(unsigned int dist = (positive ? 0 : 1);; dist++)
                                {
                                    size_t newRow = row;
                                    size_t newCol = col;
                                    if(vertical)
                                    { 
                                        if(positive)
                                        {
                                            if(col + dist >= 2 * stateArray.cols() - 1)
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                newCol += dist;
                                            }
                                        }
                                        else
                                        {
                                            if(dist > col)
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                newCol -= dist;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(positive)
                                        {
                                            if(row + dist >= 2 * stateArray.rows() - 1)
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                newRow += dist;
                                            }
                                        }
                                        else
                                        {
                                            if(dist > row)
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                newRow -= dist;
                                            }
                                        }
                                    }
                                    size_t pathwayRow = borderRows + newRow;
                                    size_t pathwayCol = borderCols + newCol;
                                    if(penalizedPathway(pathwayRow, pathwayCol) > 0)
                                    {
                                        break;
                                    }
                                    else if(newRow % 2 == 0 && newCol % 2 == 0 && 
                                        newRow / 2 >= compZoneRowStart && newRow / 2 < compZoneRowEnd && 
                                        newCol / 2 >= compZoneColStart && newCol / 2 < compZoneColEnd && 
                                        usableTargetSites.contains(std::tuple(newRow / 2, newCol / 2)))
                                    {
                                        reachableTargetSitesPerPathway[iPathway - 1].push_back(std::tuple(newRow / 2, newCol / 2));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::vector<std::vector<std::tuple<size_t,size_t>>> validSourceSitesPerPathway;
        validSourceSitesPerPathway.resize(pathwayCount);
        std::vector<std::vector<std::tuple<size_t,size_t,unsigned int,unsigned int>>> 
            validSourceSitesThatCanServeTwoPathwayPerI;
        validSourceSitesThatCanServeTwoPathwayPerI.resize(outerMaxIndex);

        for(int row = 0; row < stateArray.rows(); row++)
        {
            for(int col = 0; col < stateArray.cols(); col++)
            {
                if(stateArray(row, col) && (row < compZoneRowStart || row >= compZoneRowEnd || 
                    col < compZoneColStart || col >= compZoneColEnd || !targetGeometry(row - compZoneRowStart, col - compZoneColStart)))
                {
                    for(bool perpendicularPositive : {true,false})
                    {
                        for(int pDist = (perpendicularPositive ? 0 : 1);; pDist++)
                        {
                            int shiftedRow = 2 * row;
                            int shiftedCol = 2 * col;
                            if(vertical)
                            {
                                shiftedCol += pDist * (perpendicularPositive ? 1 : -1);
                                if(shiftedCol < 0 || shiftedCol >= 2 * stateArray.cols() - 1)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                shiftedRow += pDist * (perpendicularPositive ? 1 : -1);
                                if(shiftedRow < 0 || shiftedRow >= 2 * stateArray.rows() - 1)
                                {
                                    break;
                                }
                            }
                            if((vertical ? HALF_COL_SPACING : HALF_ROW_SPACING) * pDist < MIN_DIST_FROM_OCC_SITES)
                            {
                                if(penalizedPathway(borderRows + shiftedRow, borderCols + shiftedCol) > 1)
                                {
                                    break;
                                }
                                else
                                {
                                    unsigned int pathwayNegative = 0;
                                    unsigned int pathwayPositive = 0;
                                    for(bool alongPathwayDirPositive : {true,false})
                                    {
                                        for(int alongPDirDist = 1;; alongPDirDist++)
                                        {
                                            int twiceShiftedRow = shiftedRow;
                                            int twiceShiftedCol = shiftedCol;
                                            if(vertical)
                                            {
                                                twiceShiftedRow += alongPDirDist * (alongPathwayDirPositive ? 1 : -1);
                                                if(twiceShiftedRow < 0 || twiceShiftedRow >= 2 * stateArray.rows() - 1)
                                                {
                                                    break;
                                                }
                                            }
                                            else
                                            {
                                                twiceShiftedCol += alongPDirDist * (alongPathwayDirPositive ? 1 : -1);
                                                if(twiceShiftedCol < 0 || twiceShiftedCol >= 2 * stateArray.cols() - 1)
                                                {
                                                    break;
                                                }
                                            }
                                            if(pythagorasDist((double)(twiceShiftedRow - 2 * row) * HALF_ROW_SPACING, 
                                                (double)(twiceShiftedCol - 2 * col) * HALF_COL_SPACING) < MIN_DIST_FROM_OCC_SITES)
                                            {
                                                if(penalizedPathway(borderRows + twiceShiftedRow, borderCols + twiceShiftedCol) > 1)
                                                {
                                                    break;
                                                }
                                            }
                                            else
                                            {
                                                if(alongPathwayDirPositive)
                                                {
                                                    pathwayPositive = byIPathways(borderRows + twiceShiftedRow, borderCols + twiceShiftedCol);
                                                }
                                                else
                                                {
                                                    pathwayNegative = byIPathways(borderRows + twiceShiftedRow, borderCols + twiceShiftedCol);
                                                }
                                                //validSourceSitesPerPathway[iPathwayLabel - 1].push_back(std::tuple(row, col));
                                                break;
                                            }
                                        }
                                    }
                                    if(pathwayNegative == 0 || pathwayPositive == 0)
                                    {
                                        if(pathwayNegative != 0)
                                        {
                                            validSourceSitesPerPathway[pathwayNegative - 1].push_back(std::tuple(row, col));
                                        }
                                        if(pathwayPositive != 0)
                                        {
                                            validSourceSitesPerPathway[pathwayPositive - 1].push_back(std::tuple(row, col));
                                        }
                                    }
                                    else if(vertical)
                                    {
                                        validSourceSitesThatCanServeTwoPathwayPerI[shiftedCol].push_back(
                                            std::tuple(row, col, pathwayNegative, pathwayPositive));
                                    }
                                    else
                                    {
                                        validSourceSitesThatCanServeTwoPathwayPerI[shiftedRow].push_back(
                                            std::tuple(row, col, pathwayNegative, pathwayPositive));
                                    }
                                }
                            }
                            else
                            {
                                unsigned int iPathwayLabel = byIPathways(borderRows + shiftedRow, borderCols + shiftedCol);
                                if(iPathwayLabel == 0)
                                {
                                    break;
                                }
                                else
                                {
                                    validSourceSitesPerPathway[iPathwayLabel - 1].push_back(std::tuple(row, col));
                                }
                            }
                        }
                    }
                }
            }
        }

        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> movablePerStartAndEndI = 
            Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(outerDimSize, outerDimSize);
        for(size_t i = 0; i < outerMaxIndex; i++)
        {
            movablePerStartAndEndI.fill(0);
            std::map<unsigned int,std::tuple<std::vector<std::vector<size_t>>,std::vector<std::vector<size_t>>>> 
                startAndEndIndicesPerPathwayPerI;
            for(unsigned int pathway : pathwaysPerIndex[i])
            {
                std::vector<std::vector<size_t>> startJsPerI;
                startJsPerI.resize(outerDimSize);
                std::vector<std::vector<size_t>> endJsPerI;
                endJsPerI.resize(outerDimSize);
                for(const auto& [row,col] : validSourceSitesPerPathway[pathway - 1])
                {
                    if(vertical)
                    {
                        startJsPerI[col].push_back(row);
                    }
                    else
                    {
                        startJsPerI[row].push_back(col);
                    }
                }
                for(const auto& [row,col] : reachableTargetSitesPerPathway[pathway - 1])
                {
                    if(vertical)
                    {
                        endJsPerI[col].push_back(row);
                    }
                    else
                    {
                        endJsPerI[row].push_back(col);
                    }
                }
                startAndEndIndicesPerPathwayPerI[pathway] = std::tuple(std::move(startJsPerI), std::move(endJsPerI));
            }
            for(const auto& [row, col, pathwayNegative, pathwayPositive] : validSourceSitesThatCanServeTwoPathwayPerI[i])
            {
                auto& [negativeSideStartJsPerI, negativeSideEndJsPerI] = startAndEndIndicesPerPathwayPerI[pathwayNegative];
                auto& [positiveSideStartJsPerI, positiveSideEndJsPerI] = startAndEndIndicesPerPathwayPerI[pathwayPositive];
                size_t sourceI;
                size_t sourceJ;
                if(vertical)
                {
                    sourceI = col;
                    sourceJ = row;
                }
                else
                {
                    sourceI = row;
                    sourceJ = col;
                }
                int negativeSideDiff = (int)negativeSideStartJsPerI[sourceI].size() - (int)negativeSideEndJsPerI[sourceI].size();
                int positiveSideDiff = (int)positiveSideStartJsPerI[sourceI].size() - (int)positiveSideEndJsPerI[sourceI].size();
                if(negativeSideDiff < positiveSideDiff)
                {
                    negativeSideStartJsPerI[sourceI].push_back(sourceJ);
                }
                else
                {
                    positiveSideStartJsPerI[sourceI].push_back(sourceJ);
                }
            }
            for(unsigned int pathway : pathwaysPerIndex[i])
            {
                const auto& [startJsPerI, endJsPerI] = startAndEndIndicesPerPathwayPerI[pathway];
                for(size_t startI = 0; startI < outerDimSize; startI++)
                {
                    if(startJsPerI[startI].size() > 0)
                    {
                        for(size_t endI = 0; endI < outerDimSize; endI++)
                        {
                            if(endJsPerI[endI].size() > 0)
                            {
                                movablePerStartAndEndI(startI, endI) += startJsPerI[startI].size() < endJsPerI[endI].size() ? 
                                    startJsPerI[startI].size() : endJsPerI[endI].size();
                            }
                        }
                    }
                }
            }
            Eigen::Index startI, endI;
            unsigned int maxShifted = movablePerStartAndEndI.maxCoeff(&startI, &endI);
            double maxIntPerCost = maxShifted / (MOVE_COST_OFFSET + costPerSubMove(abs((int)i - startI)) + costPerSubMove(abs((int)i - endI)));
            if(maxIntPerCost > bestLocalIntPerCost && maxIntPerCost > bestIntPerCost && maxShifted > 0)
            {
                ParallelMove move;
                ParallelMove::Step start;
                ParallelMove::Step step1;
                ParallelMove::Step step2;
                ParallelMove::Step end;
                std::vector<double> *startIndices;
                std::vector<double> *endIndices;
                if(vertical)
                {
                    start.colSelection.push_back(startI);
                    step1.colSelection.push_back((double)i / 2);
                    step2.colSelection.push_back((double)i / 2);
                    end.colSelection.push_back(endI);
                    startIndices = &start.rowSelection;
                    endIndices = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(startI);
                    step1.rowSelection.push_back((double)i / 2);
                    step2.rowSelection.push_back((double)i / 2);
                    end.rowSelection.push_back(endI);
                    startIndices = &start.colSelection;
                    endIndices = &end.colSelection;
                }
                for(unsigned int pathway : pathwaysPerIndex[i])
                {
                    const auto& [startJsPerI, endJsPerI] = startAndEndIndicesPerPathwayPerI[pathway];
                    const auto& startJs = startJsPerI[startI];
                    const auto& endJs = endJsPerI[endI];
                    size_t sitesRequired = startJs.size() < endJs.size() ? startJs.size() : endJs.size();
                    if(sitesRequired > 0)
                    {
                        for(size_t siteIndex = 0; siteIndex < sitesRequired; siteIndex++)
                        {
                            startIndices->insert(std::upper_bound(startIndices->begin(), startIndices->end(), startJs[siteIndex]), startJs[siteIndex]);
                            endIndices->insert(std::upper_bound(endIndices->begin(), endIndices->end(), endJs[siteIndex]), endJs[siteIndex]);
                        }
                    }
                }
                move.steps.push_back(start);
                if(i != startI * 2)
                {
                    if(vertical)
                    {
                        step1.rowSelection = start.rowSelection;
                    }
                    else
                    {
                        step1.colSelection = start.colSelection;
                    }
                    move.steps.push_back(std::move(step1));
                }
                if(i != endI * 2)
                {
                    if(vertical)
                    {
                        step2.rowSelection = end.rowSelection;
                    }
                    else
                    {
                        step2.colSelection = end.colSelection;
                    }
                    move.steps.push_back(std::move(step2));
                }
                move.steps.push_back(std::move(end));
                bestLocalIntPerCost = maxShifted / move.cost();
                bestMove = std::move(move);
            }
        }
    }

    if(bestMove.has_value())
    {
        logger->info("Returning pathway move that fills {}, filledPerCost: {}", 
            bestMove.value().steps[0].rowSelection.size() * bestMove.value().steps[0].colSelection.size(), 
            bestLocalIntPerCost);
        return std::tuple(bestMove.value(), bestLocalIntPerCost);
    }
    else
    {
        logger->info("No better pathway move could be found");
        return std::nullopt;
    }
}

std::optional<std::tuple<ParallelMove,double>> removeUnusableAtomsInBorderPathway(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, size_t compZoneRowStart, size_t compZoneRowEnd, 
    size_t compZoneColStart, size_t compZoneColEnd, size_t borderRows, 
    size_t borderCols, std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrayCopy = stateArray;
    for(const auto& [r, c] : unusableAtoms)
    {
        stateArrayCopy(r,c) = false;
    }

    auto pathway = generatePathway(borderRows, borderCols, stateArrayCopy, MIN_DIST_FROM_OCC_SITES, 0);
    auto [labelledPathway, labelCount] = labelPathway(pathway);

    std::vector<std::tuple<size_t,size_t>> removableUnusableAtoms;

    for(const auto& [r, c] : unusableAtoms)
    {
        if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd &&
            labelledPathway(2 * r + borderRows, 2 * c + borderCols) == 1)
        {
            removableUnusableAtoms.push_back(std::tuple(r,c));
        }
    }
    logger->debug("{} unusable atoms could be removed", removableUnusableAtoms.size());
    
    std::vector<std::tuple<size_t, bool, bool>> outsidePathwayVerticalAtLowIndex;
    for(size_t col = 2 * compZoneColStart; col <= 2 * (compZoneColEnd - 1); col++)
    {
        bool isPathway = true;
        for(size_t row = 0; row < 2 * compZoneRowStart; row++)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(col, true, true));
        }

        isPathway = true;
        for(size_t row = 2 * (stateArray.rows() - 1); row > 2 * (compZoneRowEnd - 1); row--)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(col, true, false));
        }
    }
    for(size_t row = 2 * compZoneRowStart; row <= 2 * (compZoneRowEnd - 1); row++)
    {
        bool isPathway = true;
        for(size_t col = 0; col < 2 * compZoneColStart; col++)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(row, false, true));
        }

        isPathway = true;
        for(size_t col = 2 * (stateArray.cols() - 1); col > 2 * (compZoneColEnd - 1); col--)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(row, false, false));
        }
    }

    std::vector<std::vector<std::vector<std::tuple<int,int>>>> reachableSitesAfterMove;
    reachableSitesAfterMove.push_back(std::vector<std::vector<std::tuple<int,int>>>());
    for(const auto& [index, vertical, atLowIndex] : outsidePathwayVerticalAtLowIndex)
    {
        if(vertical)
        {
            if(atLowIndex)
            {
                reachableSitesAfterMove[0].push_back(std::vector<std::tuple<int,int>>({std::tuple(2 * compZoneRowStart - 1, index)}));
            }
            else
            {
                reachableSitesAfterMove[0].push_back(std::vector<std::tuple<int,int>>({std::tuple(2 * compZoneRowEnd - 1, index)}));
            }
        }
        else
        {
            if(atLowIndex)
            {
                reachableSitesAfterMove[0].push_back(std::vector<std::tuple<int,int>>({std::tuple(index, 2 * compZoneColStart - 1)}));
            }
            else
            {
                reachableSitesAfterMove[0].push_back(std::vector<std::tuple<int,int>>({std::tuple(index, 2 * compZoneColEnd - 1)}));
            }
        }
    }

    std::set<std::tuple<size_t,size_t>> lookedAtSites;
    for(size_t moveDist = 0; !reachableSitesAfterMove[moveDist].empty(); moveDist++)
    {
        reachableSitesAfterMove.push_back(std::vector<std::vector<std::tuple<int,int>>>());
        std::vector<std::vector<std::tuple<int,int>>> currentlyMovableAtoms;
        logger->debug("{} sites reachable at dist {}", reachableSitesAfterMove[moveDist].size(), moveDist);
        for(const auto& path : reachableSitesAfterMove[moveDist])
        {
            const auto& [row, col] = path.back();
            for(int dir = 0; dir < 4; dir++)
            {
                int rowDir = dir % 2 == 0 ? dir - 1 : 0;
                int colDir = dir % 2 == 1 ? dir - 2 : 0;
                bool hasCrossedAtom = false;
                for(int dist = 1; true; dist++)
                {
                    int currRow = row + dist * rowDir;
                    int currCol = col + dist * colDir;
                    if(currRow < 2 * compZoneRowStart || currRow > 2 * (compZoneRowEnd - 1) || 
                        currCol < 2 * compZoneColStart || currCol > 2 * (compZoneColEnd - 1) || 
                        pathway(borderRows + currRow, borderCols + currCol) > 0)
                    {
                        break;
                    }
                    else if(currRow % 2 == 0 && currCol % 2 == 0 && stateArray(currRow / 2, currCol / 2))
                    {
                        std::vector<std::tuple<int,int>> newPath = path;
                        newPath.push_back(std::tuple(currRow, currCol));
                        currentlyMovableAtoms.push_back(newPath);
                        hasCrossedAtom = true;
                    }
                    else if(!hasCrossedAtom)
                    {
                        if(!lookedAtSites.contains(std::tuple(currRow,currCol)))
                        {
                            std::vector<std::tuple<int,int>> newPath = path;
                            newPath.push_back(std::tuple(currRow,currCol));
                            reachableSitesAfterMove[moveDist + 1].push_back(newPath);
                            lookedAtSites.insert(std::tuple(currRow,currCol));
                        }
                    }
                }
            }
        }

        std::set<std::tuple<size_t,size_t>> atomsToBeMoved;
        for(const auto& path : currentlyMovableAtoms)
        {
            atomsToBeMoved.insert(path.back());
        }
        if(atomsToBeMoved.size() > 0)
        {
            std::vector<std::vector<std::tuple<int,int>>> bestRemovalSet;
            bool lastMoveVertical = (std::get<0>(currentlyMovableAtoms[0][0]) == 2 * compZoneRowStart - 1) || 
                (std::get<0>(currentlyMovableAtoms[0][0]) == 2 * compZoneRowStart - 1);
            bool atLowEnd = (std::get<0>(currentlyMovableAtoms[0][0]) == 2 * compZoneRowStart - 1) || 
                (std::get<1>(currentlyMovableAtoms[0][0]) == 2 * compZoneColStart - 1);

            for(const auto& path : currentlyMovableAtoms)
            {
                bool allowed = true;
                bool currentLastMoveVertical = (std::get<0>(path[0]) == 2 * compZoneRowStart - 1) || 
                    (std::get<0>(path[0]) == 2 * compZoneRowStart - 1);
                if(currentLastMoveVertical != lastMoveVertical)
                {
                    allowed = false;
                    continue;
                }
                for(const auto& otherPath : bestRemovalSet)
                {
                    if(path.size() != otherPath.size())
                    {
                        allowed = false;
                        break;
                    }
                    for(size_t pathIndex = 1; pathIndex < path.size(); pathIndex++)
                    {
                        const auto& [row,col] = path[pathIndex];
                        const auto& [lastRow,lastCol] = path[pathIndex - 1];
                        const auto& [otherRow,otherCol] = otherPath[pathIndex];
                        const auto& [lastOtherRow,lastOtherCol] = otherPath[pathIndex - 1];

                        if((row == otherRow && col == otherCol) || (lastRow == lastOtherRow && lastCol == lastOtherCol))
                        {
                            allowed = false;
                            break;
                        }

                        if(lastMoveVertical)
                        {
                            if(col != otherCol || lastCol != lastOtherCol || ((lastRow < lastOtherRow) != (row < otherRow)))
                            {
                                allowed = false;
                                break;
                            }
                        }
                        else
                        {
                            if(row != otherRow || lastRow != lastOtherRow || ((lastCol < lastOtherCol) != (col < otherCol)))
                            {
                                allowed = false;
                                break;
                            }
                        }
                    }
                }
                if(allowed)
                {
                    bestRemovalSet.insert(std::upper_bound(bestRemovalSet.begin(), bestRemovalSet.end(), path, 
                        [](const auto& lhs, const auto& rhs){ return std::get<0>(lhs.back()) < std::get<0>(rhs.back()) || 
                            std::get<1>(lhs.back()) < std::get<1>(rhs.back()); }), path);
                }
            }
            ParallelMove move;
            for(size_t pathIndex = bestRemovalSet.begin()->size() - 1; pathIndex >= 1; pathIndex--)
            {
                ParallelMove::Step step;
                if(lastMoveVertical)
                {
                    step.colSelection.push_back((double)(std::get<1>(bestRemovalSet[0][pathIndex])) / 2.);
                }
                else
                {
                    step.rowSelection.push_back((double)(std::get<0>(bestRemovalSet[0][pathIndex])) / 2.);
                }
                for(const auto& path : bestRemovalSet)
                {
                    if(lastMoveVertical)
                    {
                        step.rowSelection.push_back((double)(std::get<0>(path[pathIndex])) / 2.);
                    }
                    else
                    {
                        step.colSelection.push_back((double)(std::get<1>(path[pathIndex])) / 2.);
                    }
                }
                move.steps.push_back(std::move(step));
            }
            ParallelMove::Step end;
            if(lastMoveVertical)
            {
                end.colSelection.push_back((double)(std::get<1>(bestRemovalSet[0].back())) / 2.);
                int baseValue = atLowEnd ? (-move.steps.back().rowSelection.size()) : stateArray.rows();
                for(int i = 0; i < move.steps.back().rowSelection.size(); i++)
                {
                    end.rowSelection.push_back(baseValue + i);
                }
            }
            else
            {
                end.rowSelection.push_back((double)(std::get<0>(bestRemovalSet[0].back())) / 2.);
                int baseValue = atLowEnd ? (-move.steps.back().colSelection.size()) : stateArray.cols();
                for(int i = 0; i < move.steps.back().colSelection.size(); i++)
                {
                    end.colSelection.push_back(baseValue + i);
                }
            }
            move.steps.push_back(std::move(end));
            double moveBenefit = (double)(move.steps[0].rowSelection.size() * move.steps[0].colSelection.size()) / move.cost();
            logger->info("Returning removal move that eliminates {} unusable atoms", 
                move.steps[0].rowSelection.size() * move.steps[0].colSelection.size());
            return std::tuple(move, moveBenefit);
        }
    }

    logger->info("No better removal move could be found");
    return std::nullopt;
}

bool updateDataStructuresAfterFindingMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> pathway,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> labelledPathway, 
    unsigned int& labelCount, size_t borderRows, size_t borderCols,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    const ParallelMove &move, std::set<std::tuple<size_t,size_t>>& unusableAtoms, 
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto occMask = generateMask(RECOMMENDED_DIST_FROM_OCC_SITES, 0.5);
    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;
    auto emptyMask = generateMask(RECOMMENDED_DIST_FROM_EMPTY_SITES, 0.5);
    Eigen::Index halfEmptyRows = emptyMask.rows() / 2;
    Eigen::Index halfEmptyCols = emptyMask.cols() / 2;
    auto penalizedOccMask = generateMask(MIN_DIST_FROM_OCC_SITES, 0.5);
    Eigen::Index halfPOccRows = penalizedOccMask.rows() / 2;
    Eigen::Index halfPOccCols = penalizedOccMask.cols() / 2;

    for(size_t rIndex = 0; rIndex < move.steps[0].rowSelection.size(); rIndex++)
    {
        for(size_t cIndex = 0; cIndex < move.steps[0].colSelection.size(); cIndex++)
        {
            int startRow = move.steps[0].rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD; 
            int startCol = move.steps[0].colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD;

            if(move.steps[0].rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD - startRow < 0.25 && 
                move.steps[0].colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD - startCol < 0.25 && 
                stateArray(startRow, startCol))
            {
                int newRow = move.steps.back().rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD;
                int newCol = move.steps.back().colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD;
                int pathwayStartRow = 2 * startRow + borderRows;
                int pathwayStartCol = 2 * startCol + borderCols;

                if(unusableAtoms.erase(std::tuple(startRow, startCol)))
                {
                    if(newRow >= 0 && newRow < stateArray.rows() && newCol >= 0 && newCol < stateArray.cols())
                    {
                        unusableAtoms.insert(std::tuple(newRow, newCol));
                    }
                }

                if(newRow >= compZoneRowStart && newRow < compZoneRowEnd && newCol >= compZoneColStart && newCol < compZoneColEnd)
                {
                    usableTargetSites.erase(std::tuple(newRow, newCol));
                    pathway(Eigen::seqN(2 * newRow + borderRows - halfOccRows, occMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfOccCols, occMask.cols())) += occMask.cast<unsigned int>();
                    pathway(Eigen::seqN(2 * newRow + borderRows - halfEmptyRows, emptyMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfEmptyCols, emptyMask.cols())) -= emptyMask.cast<unsigned int>();
                    penalizedPathway(Eigen::seqN(2 * newRow + borderRows - halfPOccRows, penalizedOccMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfPOccCols, penalizedOccMask.cols())) += penalizedOccMask.cast<unsigned int>();
                }

                pathway(Eigen::seqN(pathwayStartRow - halfOccRows, occMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfOccCols, occMask.cols())) -= occMask.cast<unsigned int>();
                pathway(Eigen::seqN(pathwayStartRow - halfEmptyRows, emptyMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfEmptyCols, emptyMask.cols())) += emptyMask.cast<unsigned int>();
                penalizedPathway(Eigen::seqN(pathwayStartRow - halfPOccRows, penalizedOccMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfPOccCols, penalizedOccMask.cols())) -= penalizedOccMask.cast<unsigned int>();
                for(int rowShift = -halfPOccRows / 2; rowShift <= halfPOccRows / 2; rowShift++)
                {
                    int shiftedRow = startRow + rowShift;
                    if(shiftedRow >= compZoneRowStart && shiftedRow < compZoneRowEnd)
                    {
                        for(int colShift = -halfPOccCols / 2; colShift <= halfPOccCols / 2; colShift++)
                        {
                            int shiftedCol = startCol + colShift;
                            if(shiftedCol >= compZoneColStart && shiftedCol < compZoneColEnd && 
                                targetGeometry(shiftedRow - compZoneRowStart, shiftedCol - compZoneColStart) && 
                                !stateArray(shiftedRow, shiftedCol) && 
                                penalizedPathway(2 * shiftedRow + borderRows, 2 * shiftedCol + borderCols) == 0)
                            {
                                usableTargetSites.insert(std::tuple(shiftedRow, shiftedCol));
                            }
                        }
                    }
                }
            }
        }
    }

    std::tie(labelledPathway, labelCount) = labelPathway(pathway);
    return true;
}

std::tuple<std::set<std::tuple<size_t,size_t>>,std::set<std::tuple<size_t,size_t>>> findUnusableAtomsAndUsableTargetSites(
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto usabilityPreventingNeighborhoodMask = generateMask(MIN_DIST_FROM_OCC_SITES);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    std::set<std::tuple<size_t,size_t>> unusableAtoms;
    std::set<std::tuple<size_t,size_t>> usableTargetSites;
    bool changesWereMade = true;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> unusabilityArray(stateArray.rows(), stateArray.cols());
    unusabilityArray.fill(false);

    while(changesWereMade)
    {
        changesWereMade = false;
        for(Eigen::Index row = 0; row < stateArray.rows(); row++)
        {
            for(Eigen::Index col = 0; col < stateArray.cols(); col++)
            {
                if(stateArray(row,col) && (row < (int)compZoneRowStart || row >= (int)compZoneRowEnd || 
                    col < (int)compZoneColStart || col >= (int)compZoneColEnd || 
                    !targetGeometry(row - compZoneRowStart, col - compZoneColStart) ||
                    unusabilityArray(row,col)))
                {
                    for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                        rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                    {
                        int shiftedRow = row + rowShift;
                        if(shiftedRow >= 0 && shiftedRow < stateArray.rows())
                        {
                            for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                                colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                            {
                                int shiftedCol = col + colShift;
                                if(shiftedCol >= 0 && shiftedCol < stateArray.cols() && usabilityPreventingNeighborhoodMask(
                                    rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                    colShift + usabilityPreventingNeighborhoodMaskColDist))
                                {
                                    changesWereMade |= !unusabilityArray(shiftedRow, shiftedCol);
                                    unusabilityArray(shiftedRow, shiftedCol) = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::stringstream strstream;
    strstream << "Unusable sites: \n";
    for(int row = 0; row < stateArray.rows(); row++)
    {
        for(int col = 0; col < stateArray.cols(); col++)
        {
            if(stateArray(row,col) && unusabilityArray(row, col))
            {
                unusableAtoms.insert(std::tuple(row,col));
                strstream << "X";
            }
            else if(row >= compZoneRowStart && row < compZoneRowEnd && col >= compZoneColStart && col < compZoneColEnd && 
                !stateArray(row,col) && targetGeometry(row - compZoneRowStart, col - compZoneColStart) && 
                !unusabilityArray(row, col))
            {
                usableTargetSites.insert(std::tuple(row,col));
                strstream << "A";
            }
            else if(stateArray(row,col))
            {
                strstream << "█";
            }
            else
            {
                strstream << " ";
            }
        }
        strstream << "\n";
    }
    logger->debug(strstream.str());
    return std::tuple(unusableAtoms, usableTargetSites);
}

bool findAndExecuteMoves(Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> pathway, 
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> labelledPathway, unsigned int labelCount,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableTargetSites, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, std::vector<ParallelMove>& moveList, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, std::shared_ptr<spdlog::logger> logger)
{
    while(true)
    {
        auto targetSitesNeighboringPathways = findTargetSitesPerPathway(penalizedPathway,
            labelledPathway, usableTargetSites, borderRows, borderCols, labelCount, 
            compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd,
            stateArray(Eigen::seq(compZoneRowStart, compZoneRowEnd - 1),Eigen::seq(compZoneColStart, compZoneColEnd - 1)), 
            targetGeometry, logger);

        std::vector<std::tuple<size_t,size_t>> possibleMoves;

        auto occMask = generateMask(MIN_DIST_FROM_OCC_SITES, 0.5);
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> localAccessibility(occMask.rows() + 2, occMask.cols() + 2);

        for(size_t r = 0; r < (size_t)targetGeometry.rows(); r++)
        {
            Eigen::Index pathwayRowIndex = 2 * (r + compZoneRowStart) + borderRows;
            for(size_t c = 0; c < (size_t)targetGeometry.cols(); c++)
            {
                if(!targetGeometry(r,c) && stateArray(r + compZoneRowStart, c + compZoneColStart) && 
                    !unusableAtoms.contains(std::tuple(r + compZoneRowStart, c + compZoneColStart)))
                {
                    Eigen::Index pathwayColIndex = 2 * (c + compZoneRowStart) + borderCols;
                    std::set<size_t> accessibleTargetSites;
                    bool changesWereMade = true;
                    localAccessibility.fill(false);
                    localAccessibility(localAccessibility.rows() / 2, localAccessibility.cols() / 2) = true;
                    while(changesWereMade)
                    {
                        changesWereMade = false;
                        for(Eigen::Index lRow = 0; lRow < localAccessibility.rows(); lRow++)
                        {
                            for(Eigen::Index lCol = 0; lCol < localAccessibility.cols(); lCol++)
                            {
                                if(localAccessibility(lRow, lCol))
                                {
                                    if((localAccessibility.rows() / 2 - lRow) % 2 == 0 && (localAccessibility.cols() / 2 - lCol) % 2 == 0 && 
                                        usableTargetSites.contains(std::tuple(r + compZoneRowStart - (localAccessibility.rows() / 2 - lRow) / 2,
                                            c + compZoneColStart - (localAccessibility.cols() / 2 - lCol) / 2)))
                                    {
                                        accessibleTargetSites.insert((r - (localAccessibility.rows() / 2 - lRow) / 2) * 
                                            targetGeometry.cols() + c - (localAccessibility.cols() / 2 - lCol) / 2);
                                    }
                                    if(labelledPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                        pathwayColIndex - localAccessibility.cols() / 2 + lCol) != 0)
                                    {
                                        for(const auto& targetSite : targetSitesNeighboringPathways[
                                            labelledPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                                pathwayColIndex - localAccessibility.cols() / 2 + lCol) - 1])
                                        {
                                            accessibleTargetSites.insert(targetSite);
                                        }
                                    }
                                }
                                else
                                {
                                    bool withinOwnAtomsProhibitedArea = lRow > 0 && lRow < localAccessibility.rows() - 1 && 
                                        lCol > 0 && lCol < localAccessibility.cols() - 1 && occMask(lRow - 1, lCol - 1);
                                    bool notInOtherPathway = penalizedPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                        pathwayColIndex - localAccessibility.cols() / 2 + lCol) <= (withinOwnAtomsProhibitedArea ? 1 : 0);
                                    bool adjacentToAccessibleLoc = 
                                        (lRow > 0 && localAccessibility(lRow - 1, lCol)) || 
                                        (lRow < localAccessibility.rows() - 1 && localAccessibility(lRow + 1, lCol)) || 
                                        (lCol > 0 && localAccessibility(lRow, lCol - 1)) || 
                                        (lCol < localAccessibility.cols() - 1 && localAccessibility(lRow, lCol + 1));
                                    if(notInOtherPathway && adjacentToAccessibleLoc)
                                    {
                                        localAccessibility(lRow, lCol) = true;
                                        changesWereMade = true;
                                    }
                                }
                            }
                        }
                    }
                    for(const auto& targetSite : accessibleTargetSites)
                    {
                        possibleMoves.push_back(std::tuple(r * targetGeometry.cols() + c, targetSite));
                    }
                }
            }
        }

        std::optional<ParallelMove> bestMove = std::nullopt;
        double bestIntPerCost = -1;

        auto distOneMove = findDistOneMove(penalizedPathway, stateArray, targetGeometry, possibleMoves, unusableAtoms, compZoneRowStart, 
            compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, logger);
        if(distOneMove.has_value())
        {
            std::tie(bestMove, bestIntPerCost) = distOneMove.value();
        }

        auto removalMove = removeUnusableAtomsInBorderPathway(stateArray, unusableAtoms, compZoneRowStart, compZoneRowEnd, 
            compZoneColStart, compZoneColEnd, borderRows, borderCols, moveList, logger);
        if(removalMove.has_value() && (!bestMove.has_value() || std::get<1>(removalMove.value()) > bestIntPerCost))
        {
            std::tie(bestMove, bestIntPerCost) = removalMove.value();
        }

        auto complexMove = findComplexMove(penalizedPathway, pathway, labelledPathway, stateArray, targetGeometry, unusableAtoms, usableTargetSites,
            possibleMoves, compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, bestIntPerCost, logger);
        if(complexMove.has_value() && (!bestMove.has_value() || std::get<1>(complexMove.value()) > bestIntPerCost))
        {
            std::tie(bestMove, bestIntPerCost) = complexMove.value();
        }

        auto complexMoveLegacy = findComplexMoveLegacy(penalizedPathway, labelledPathway, stateArray, targetGeometry, unusableAtoms, 
            possibleMoves, compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, bestIntPerCost, logger);
        if(complexMoveLegacy.has_value() && (!bestMove.has_value() || std::get<1>(complexMoveLegacy.value()) > bestIntPerCost))
        {
            std::tie(bestMove, bestIntPerCost) = complexMoveLegacy.value();
        }

        if(bestMove.has_value())
        {
            if(!updateDataStructuresAfterFindingMove(penalizedPathway, pathway, labelledPathway, labelCount, borderRows, borderCols,
                stateArray, targetGeometry, bestMove.value(), unusableAtoms, usableTargetSites, 
                compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, logger))
            {
                return false;
            }
            if(!bestMove.value().execute(stateArray, logger))
            {
                return false;
            }
            moveList.push_back(std::move(bestMove.value()));
        }
        else
        {
            break;
        }
    }

    return true;
}

bool checkTargetGeometryFeasibility(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto usabilityPreventingNeighborhoodMask = generateMask(MIN_DIST_FROM_OCC_SITES);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(targetGeometry(row,col))
            {
                for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                    rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                {
                    int shiftedRow = row + rowShift;
                    if(shiftedRow >= 0 && shiftedRow < targetGeometry.rows())
                    {
                        for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                            colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                        {
                            int shiftedCol = col + colShift;
                            if(shiftedCol >= 0 && shiftedCol < targetGeometry.cols() && usabilityPreventingNeighborhoodMask(
                                rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                colShift + usabilityPreventingNeighborhoodMaskColDist))
                            {
                                if(targetGeometry(shiftedRow,shiftedCol))
                                {
                                    logger->error("Target atoms too close together to be sorted");
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    size_t borderRows = MIN_DIST_FROM_OCC_SITES / HALF_ROW_SPACING + 2;
    size_t borderCols = MIN_DIST_FROM_OCC_SITES / HALF_COL_SPACING + 2;
    auto pathway = generatePathway(borderRows, borderCols, targetGeometry, MIN_DIST_FROM_OCC_SITES, 0);
    auto [labelledPathway, labelCount] = labelPathway(pathway);

    auto occMask = generateMask(MIN_DIST_FROM_OCC_SITES, 0.5);
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> localAccessibility(occMask.rows() + 2, occMask.cols() + 2);
    for(Eigen::Index r = 0; r < targetGeometry.rows(); r++)
    {
        for(Eigen::Index c = 0; c < targetGeometry.cols(); c++)
        {
            if(targetGeometry(r,c))
            {
                bool directConnectionToPathwayExists = false;
                bool changesWereMade = true;
                localAccessibility.fill(false);
                localAccessibility(localAccessibility.rows() / 2, localAccessibility.cols() / 2) = true;
                while(changesWereMade && !directConnectionToPathwayExists)
                {
                    changesWereMade = false;
                    for(Eigen::Index lRow = 0; lRow < localAccessibility.rows() && !directConnectionToPathwayExists; lRow++)
                    {
                        for(Eigen::Index lCol = 0; lCol < localAccessibility.cols() && !directConnectionToPathwayExists; lCol++)
                        {
                            if(localAccessibility(lRow, lCol))
                            {
                                if(labelledPathway(2 * r + borderRows - localAccessibility.rows() / 2 + lRow,
                                    2 * c + borderCols - localAccessibility.cols() / 2 + lCol) == 1)
                                {
                                    directConnectionToPathwayExists = true;
                                }
                            }
                            else
                            {
                                bool withinOwnAtomsProhibitedArea = lRow > 0 && lRow < localAccessibility.rows() - 1 && 
                                    lCol > 0 && lCol < localAccessibility.cols() - 1 && occMask(lRow - 1, lCol - 1);
                                bool notInOtherPathway = pathway(2 * r + borderRows - localAccessibility.rows() / 2 + lRow,
                                    2 * c + borderCols - localAccessibility.cols() / 2 + lCol) <= (withinOwnAtomsProhibitedArea ? 1 : 0);
                                bool adjacentToAccessibleLoc = 
                                    (lRow > 0 && localAccessibility(lRow - 1, lCol)) || 
                                    (lRow < localAccessibility.rows() - 1 && localAccessibility(lRow + 1, lCol)) || 
                                    (lCol > 0 && localAccessibility(lRow, lCol - 1)) || 
                                    (lCol < localAccessibility.cols() - 1 && localAccessibility(lRow, lCol + 1));
                                if(notInOtherPathway && adjacentToAccessibleLoc)
                                {
                                    localAccessibility(lRow, lCol) = true;
                                    changesWereMade = true;
                                }
                            }
                        }
                    }
                }
                
                if(!directConnectionToPathwayExists)
                {
                    logger->error("No connection to pathway for ({}/{})", r, c);
                    return false;
                }
            }
        }
    }
    std::stringstream strstream;
    strstream << "Target: \n";
    for(size_t r = 0; r < (size_t)targetGeometry.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)targetGeometry.cols(); c++)
        {
            strstream << (targetGeometry(r,c) ? "X" : " ");
        }
        strstream << "\n";
    }
    strstream << "Pathway: \n";
    for(Eigen::Index r = 0; r < pathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < pathway.cols(); c++)
        {
            strstream << (pathway(r, c) == 0 ? " " : "█");
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    return true;
}

bool removeAllDirectlyRemovableUnusableAtoms(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, size_t borderRows, size_t borderCols, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrayCopy = stateArray;
    for(const auto& [r, c] : unusableAtoms)
    {
        stateArrayCopy(r,c) = false;
    }

    auto pathway = generatePathway(borderRows, borderCols, stateArrayCopy, MIN_DIST_FROM_OCC_SITES, 0);

    for(bool traverseRow : {true,false})
    {
        if((traverseRow && MIN_DIST_FROM_OCC_SITES <= COL_SPACING) || 
            (!traverseRow && MIN_DIST_FROM_OCC_SITES <= ROW_SPACING))
        {
            for(bool lowToHigh : {true,false})
            {
                std::vector<int> minDist;
                std::vector<int> maxDist;
                size_t alongBorderMaxIndex = traverseRow ? stateArray.cols() : stateArray.rows();
                RowBitMask indexDone(alongBorderMaxIndex);
                size_t inwardMax = traverseRow ? stateArray.rows() : stateArray.cols();
                size_t startIndex = lowToHigh ? 0 : (inwardMax - 1);
                int dir = lowToHigh ? 1 : -1;
                for(size_t alongBorder = 0; alongBorder < alongBorderMaxIndex; alongBorder++)
                {
                    int minDistV = -1;
                    bool inBlock = false;
                    for(int inwardDist = 0; inwardDist < (int)inwardMax; inwardDist++)
                    {
                        size_t r = traverseRow ? startIndex + inwardDist * dir : alongBorder;
                        size_t c = traverseRow ? alongBorder : (startIndex + inwardDist * dir);
                        if(unusableAtoms.contains(std::tuple(r,c)))
                        {
                            inBlock = true;
                        }
                        else if(inBlock)
                        {
                            inBlock = false;
                            minDistV = inwardDist - 1;
                        }
                        if(!pathway(2 * r + borderRows, 2 * c + borderCols) == 0 || inwardDist == (int)(inwardMax - 1))
                        {
                            minDist.push_back(minDistV);
                            maxDist.push_back(inwardDist - 1);
                            indexDone.set(alongBorder, minDistV == -1);
                            break;
                        }
                    }
                }
                logger->debug("Indices to move directly: {}/{}", alongBorderMaxIndex - indexDone.bitsSet(), alongBorderMaxIndex);

                while(indexDone.bitsSet() < alongBorderMaxIndex)
                {
                    int maxMinDist = -1;
                    for(size_t i = 0; i < minDist.size(); i++)
                    {
                        if(!indexDone[i] && minDist[i] > maxMinDist)
                        {
                            maxMinDist = minDist[i];
                        }
                    }
                    std::vector<size_t> usedIndices;
                    unsigned int selectedIndices = 0;
                    unsigned int maxIndicesAlongBorder = traverseRow ? AOD_COL_LIMIT : AOD_ROW_LIMIT;
                    if(AOD_TOTAL_LIMIT < maxIndicesAlongBorder)
                    {
                        maxIndicesAlongBorder = AOD_TOTAL_LIMIT;
                    }
                    for(size_t alongBorder = 0; alongBorder < alongBorderMaxIndex && selectedIndices < maxIndicesAlongBorder; alongBorder++)
                    {
                        if(!indexDone[alongBorder] && maxDist[alongBorder] >= maxMinDist)
                        {
                            usedIndices.push_back(alongBorder);
                            indexDone.set(alongBorder, true);
                            selectedIndices++;
                        }
                    }
                    unsigned int maxIndicesInward = traverseRow ? AOD_ROW_LIMIT : AOD_COL_LIMIT;
                    if(maxIndicesInward * selectedIndices > AOD_TOTAL_LIMIT)
                    {
                        maxIndicesInward = AOD_TOTAL_LIMIT / selectedIndices;
                    }
                    std::vector<int> inwardIndicesRequired;
                    for(int inwardDist = 0; inwardDist <= maxMinDist; inwardDist++)
                    {
                        bool stepRequired = false;
                        for(const auto& alongBorder : usedIndices)
                        {
                            if(unusableAtoms.erase(traverseRow ? std::tuple(startIndex + inwardDist * dir,alongBorder) : 
                                std::tuple(alongBorder, startIndex + inwardDist * dir)))
                            {
                                stepRequired = true;
                            }
                        }
                        if(stepRequired)
                        {
                            inwardIndicesRequired.push_back(startIndex + inwardDist * dir);
                        }
                    }

                    auto first = inwardIndicesRequired.begin();
                    unsigned int moveCount = (unsigned int)ceil((double)inwardIndicesRequired.size() / (double)maxIndicesInward);
                    for(unsigned int moveIndex = 0; moveIndex < moveCount; moveIndex++)
                    {
                        ParallelMove move;
                        ParallelMove::Step start;
                        std::vector<double> *alongBorderSelection;
                        std::vector<double> *inwardSelection;
                        if(traverseRow)
                        {
                            alongBorderSelection = &start.colSelection;
                            inwardSelection = &start.rowSelection;
                        }
                        else
                        {
                            alongBorderSelection = &start.rowSelection;
                            inwardSelection = &start.colSelection;
                        }
                        auto last = ((moveIndex + 1) * maxIndicesInward >= inwardIndicesRequired.size()) ? inwardIndicesRequired.end() : std::next(first, maxIndicesInward);
                        inwardSelection->insert(inwardSelection->end(), first, last);
                        if(!lowToHigh)
                        {
                            std::reverse(inwardSelection->begin(), inwardSelection->end());
                        }
                        alongBorderSelection->insert(alongBorderSelection->end(), usedIndices.begin(), usedIndices.end());
                        move.steps.push_back(std::move(start));

                        ParallelMove::Step end;
                        if(traverseRow)
                        {
                            alongBorderSelection = &end.colSelection;
                            inwardSelection = &end.rowSelection;
                        }
                        else
                        {
                            alongBorderSelection = &end.rowSelection;
                            inwardSelection = &end.colSelection;
                        }
                        int count = ((moveIndex + 1) * maxIndicesInward > inwardIndicesRequired.size()) ? inwardIndicesRequired.size() - moveIndex * maxIndicesInward : maxIndicesInward;
                        logger->debug("Moving {}/{} atoms", count, inwardIndicesRequired.size());
                        for(int i = 0; i < count; i++)
                        {
                            inwardSelection->push_back((int)startIndex - dir * (lowToHigh ? (count - i) : (i + 1)));
                        }
                        alongBorderSelection->insert(alongBorderSelection->end(), usedIndices.begin(), usedIndices.end());
                        move.steps.push_back(std::move(end));

                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));

                        if(moveIndex < moveCount - 1)
                        {
                            first = std::next(first, maxIndicesInward);
                        }
                    }
                }
            }
        }
    }
    return true;
}

ParallelMove createRemovalMove(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, bool verticalMove, 
    std::vector<double> unchangingIndices, std::vector<double> movedIndices, bool moveTowardsLowIndices)
{
    ParallelMove move;
    ParallelMove::Step start;
    ParallelMove::Step end;
    if(verticalMove)
    {
        start.colSelection = unchangingIndices;
        end.colSelection = unchangingIndices;
        start.rowSelection = movedIndices;
        for(size_t outsideDist = 0; outsideDist < movedIndices.size(); outsideDist++)
        {
            if(moveTowardsLowIndices)
            {
                end.rowSelection.push_back(-(double)(movedIndices.size() - outsideDist));
            }
            else
            {
                end.rowSelection.push_back(stateArray.rows() + outsideDist);
            }
        }
    }
    else
    {
        start.rowSelection = unchangingIndices;
        end.rowSelection = unchangingIndices;
        start.colSelection = movedIndices;
        for(size_t outsideDist = 0; outsideDist < movedIndices.size(); outsideDist++)
        {
            if(moveTowardsLowIndices)
            {
                end.colSelection.push_back(-(double)(movedIndices.size() - outsideDist));
            }
            else
            {
                end.colSelection.push_back(stateArray.cols() + outsideDist);
            }
        }
    }
    move.steps.push_back(std::move(start));
    move.steps.push_back(std::move(end));

    return move;
}

bool createMinimallyInvasiveAccessPathway(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, size_t borderRows, size_t borderCols, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    std::vector<ParallelMove>& moveList, unsigned int count, std::shared_ptr<spdlog::logger> logger)
{
    bool pathwayVertical = COL_SPACING > ROW_SPACING;
    double spacing = pathwayVertical ? COL_SPACING : ROW_SPACING;
    if(2 * MIN_DIST_FROM_OCC_SITES < spacing)
    {
        return true;
    }
    unsigned int width = ceil((double)(2 * MIN_DIST_FROM_OCC_SITES) / spacing) - 1;
    size_t compZoneWidth = pathwayVertical ? (compZoneColEnd - compZoneColStart) : (compZoneRowEnd - compZoneRowStart);
    if(width > compZoneWidth)
    {
        logger->error("Width of outside pathway exceeds size of computation zone");
        return false;
    }
    std::vector<unsigned int> usableAtomsPerIndexLowIndexSide;
    std::vector<unsigned int> usableAtomsPerIndexHighIndexSide;
    RowBitMask lowSideContainsAtoms(compZoneWidth);
    RowBitMask highSideContainsAtoms(compZoneWidth);
    for(size_t alongBorderIndex = 0; alongBorderIndex < compZoneWidth; alongBorderIndex++)
    {
        unsigned int removedUsableAtoms = 0;
        for(size_t inwardIndex = 0; inwardIndex < (pathwayVertical ? compZoneRowStart : compZoneColStart); inwardIndex++)
        {
            size_t row = pathwayVertical ? inwardIndex : (alongBorderIndex + compZoneRowStart);
            size_t col = pathwayVertical ? (alongBorderIndex + compZoneColStart) : inwardIndex;
            if(stateArray(row, col))
            {
                lowSideContainsAtoms.set(alongBorderIndex, true);
                if(!unusableAtoms.contains(std::tuple(row, col)))
                {
                    removedUsableAtoms++;
                }
            }
        }
        usableAtomsPerIndexLowIndexSide.push_back(removedUsableAtoms);
        removedUsableAtoms = 0;
        for(size_t inwardIndex = pathwayVertical ? (stateArray.rows() - 1) : (stateArray.cols() - 1); 
            inwardIndex >= (pathwayVertical ? compZoneRowEnd : compZoneColEnd); inwardIndex--)
        {
            size_t row = pathwayVertical ? inwardIndex : (alongBorderIndex + compZoneRowStart);
            size_t col = pathwayVertical ? (alongBorderIndex + compZoneColStart) : inwardIndex;
            if(stateArray(row, col))
            {
                highSideContainsAtoms.set(alongBorderIndex, true);
                if(!unusableAtoms.contains(std::tuple(row, col)))
                {
                    removedUsableAtoms++;
                }
            }
        }
        usableAtomsPerIndexHighIndexSide.push_back(removedUsableAtoms);
    }

    std::vector<std::tuple<size_t,unsigned int>> bestIndicesLow;
    std::vector<std::tuple<size_t,unsigned int>> bestIndicesHigh;
    for(size_t alongBorderIndex = 0; alongBorderIndex < compZoneWidth - width; alongBorderIndex++)
    {
        unsigned int totalRemovedUsableAtoms = 0;
        for(size_t i = 0; i < width; i++)
        {
            totalRemovedUsableAtoms += usableAtomsPerIndexLowIndexSide[alongBorderIndex + i];
        }
        std::tuple<size_t,unsigned int> newElemLow(alongBorderIndex, totalRemovedUsableAtoms);
        bestIndicesLow.insert(std::upper_bound(bestIndicesLow.begin(), bestIndicesLow.end(), newElemLow, 
            [](const auto& lhs, const auto& rhs) { return std::get<1>(lhs) < std::get<1>(rhs); }), newElemLow);

        totalRemovedUsableAtoms = 0;
        for(size_t i = 0; i < width; i++)
        {
            totalRemovedUsableAtoms += usableAtomsPerIndexHighIndexSide[alongBorderIndex + i];
        }
        std::tuple<size_t,unsigned int> newElemHigh(alongBorderIndex, totalRemovedUsableAtoms);
        bestIndicesHigh.insert(std::upper_bound(bestIndicesHigh.begin(), bestIndicesHigh.end(), newElemHigh, 
            [](const auto& lhs, const auto& rhs) { return std::get<1>(lhs) < std::get<1>(rhs); }), newElemHigh);
    }

    std::set<size_t> usedIndicesLow;
    std::set<size_t> usedIndicesHigh;
    for(unsigned int index = 0; index < count; index++)
    {
        std::optional<unsigned int> lowestRemovedUnusableAtoms = std::nullopt;
        std::optional<size_t> bestNewIndex = std::nullopt;
        std::optional<int> bestMinDist = std::nullopt;
        unsigned int removedAtoms = 0;

        std::vector<std::tuple<size_t,unsigned int>> *bestIndices;
        std::set<size_t> *usedIndices;

        if((usedIndicesLow.size() <= usedIndicesHigh.size() || bestIndicesHigh.size() <= usedIndicesHigh.size()) && 
            bestIndicesLow.size() > usedIndicesLow.size())
        {
            bestIndices = &bestIndicesLow;
            usedIndices = &usedIndicesLow;
        }
        else if(bestIndicesHigh.size() > usedIndicesHigh.size())
        {
            bestIndices = &bestIndicesHigh;
            usedIndices = &usedIndicesHigh;
        }
        else
        {
            logger->error("Not enough indices to provide required number of pathways to outside");
            return false;
        }
        for(const auto& [index, removedUsableAtoms] : *bestIndices)
        {
            if(!usedIndices->contains(index))
            {
                if(!lowestRemovedUnusableAtoms.has_value())
                {
                    lowestRemovedUnusableAtoms = removedUsableAtoms;
                }
                if(!bestNewIndex.has_value())
                {
                    bestNewIndex = index;
                    removedAtoms = removedUsableAtoms;
                    for(const auto& alreadySelectedIndex : *usedIndices)
                    {
                        int dist = abs((int)alreadySelectedIndex - (int)index);
                        if(!bestMinDist.has_value() || dist < bestMinDist.value())
                        {
                            bestMinDist = dist;
                        }
                    }
                    if(usedIndices->empty())
                    {
                        break;
                    }
                }
                else
                {
                    std::optional<int> localMinDist = std::nullopt;
                    for(const auto& alreadySelectedIndex : *usedIndices)
                    {
                        int dist = abs((int)alreadySelectedIndex - (int)index);
                        if(!localMinDist.has_value() || dist < localMinDist.value())
                        {
                            localMinDist = dist;
                        }
                    }
                    if(bestMinDist.has_value() && localMinDist.has_value() && localMinDist.value() > bestMinDist.value())
                    {
                        bestNewIndex = index;
                        removedAtoms = removedUsableAtoms;
                        bestMinDist = localMinDist;
                    }
                }
            }
            if(lowestRemovedUnusableAtoms.has_value() && bestNewIndex.has_value() && 
                removedUsableAtoms > lowestRemovedUnusableAtoms.value() + 2)
            {
                break;
            }
        }

        if(bestNewIndex.has_value())
        {
            logger->info("Sacrificing {} useful atoms to create pathway of width {} from compZone to outside start at index {}", 
                removedAtoms, width, bestNewIndex.value());
            usedIndices->insert(bestNewIndex.value());
        }
        else
        {
            logger->error("No index for creating pathway to outside could be determined");
            return false;
        }
    }

    while(!usedIndicesLow.empty() || !usedIndicesHigh.empty())
    {
        bool atLowIndex;
        size_t newIndex;
        if(!usedIndicesLow.empty())
        {
            atLowIndex = true;
            newIndex = *usedIndicesLow.begin();
            usedIndicesLow.erase(newIndex);
        }
        else
        {
            atLowIndex = false;
            newIndex = *usedIndicesHigh.begin();
            usedIndicesHigh.erase(newIndex);
        }

        unsigned int requiredIndicesAlongBorder = 0;
        for(size_t withinWidth = 0; withinWidth < width; withinWidth++)
        {
            if((atLowIndex && lowSideContainsAtoms[newIndex + withinWidth]) || 
                (!atLowIndex && highSideContainsAtoms[newIndex + withinWidth]))
            {
                requiredIndicesAlongBorder++;
            }
        }
        unsigned int maxAlongBorderIndices = pathwayVertical ? AOD_COL_LIMIT : AOD_ROW_LIMIT;
        if(AOD_TOTAL_LIMIT < maxAlongBorderIndices)
        {
            maxAlongBorderIndices = AOD_TOTAL_LIMIT;
        }
        if(requiredIndicesAlongBorder == 0)
        {
            continue;
        }

        size_t alongBorderIndex = newIndex;
        size_t alongBorderSegments = ceil((double)requiredIndicesAlongBorder / (double)maxAlongBorderIndices - DOUBLE_EQUIVALENCE_THRESHOLD);
        for(size_t alongBorderSeg = 0; alongBorderSeg < alongBorderSegments; alongBorderSeg++)
        {
            size_t usedIndicesAlongBorder = requiredIndicesAlongBorder / alongBorderSegments + 
                ((alongBorderSeg < requiredIndicesAlongBorder % alongBorderSegments) ? 1 : 0);
            unsigned int maxInwardIndices = pathwayVertical ? AOD_ROW_LIMIT : AOD_COL_LIMIT;
            if(usedIndicesAlongBorder * maxInwardIndices > AOD_TOTAL_LIMIT)
            {
                maxInwardIndices = AOD_TOTAL_LIMIT / usedIndicesAlongBorder;
            }
            std::vector<double> alongBorderIndices;
            for(; alongBorderIndices.size() < usedIndicesAlongBorder && 
                alongBorderIndex < newIndex + width; alongBorderIndex++)
            {
                if((atLowIndex && lowSideContainsAtoms[alongBorderIndex]) || 
                    (!atLowIndex && highSideContainsAtoms[alongBorderIndex]))
                {
                    alongBorderIndices.push_back(alongBorderIndex + (pathwayVertical ? compZoneColStart : compZoneRowStart));
                }
            }
            size_t inwardIndex;
            size_t stopIndex;
            if(atLowIndex)
            {
                inwardIndex = 0;
                if(pathwayVertical)
                {
                    stopIndex = compZoneRowStart;
                }
                else
                {
                    stopIndex = compZoneColStart;
                }
            }
            else
            {
                if(pathwayVertical)
                {
                    inwardIndex = stateArray.rows() - 1;
                    stopIndex = compZoneRowEnd - 1;
                }
                else
                {
                    inwardIndex = stateArray.cols() - 1;
                    stopIndex = compZoneColEnd - 1;
                }
            }
            std::vector<double> inwardIndices;
            for(; inwardIndex != stopIndex; inwardIndex += (atLowIndex ? 1 : -1))
            {
                bool indexRequired = false;
                for(const auto& usedIndex : alongBorderIndices)
                {
                    size_t usedIndexSt = floor(usedIndex + DOUBLE_EQUIVALENCE_THRESHOLD);
                    if(stateArray((pathwayVertical ? inwardIndex : usedIndexSt), (pathwayVertical ? usedIndexSt : inwardIndex)))
                    {
                        indexRequired = true;
                        break;
                    }
                }
                if(indexRequired)
                {
                    if(atLowIndex)
                    {
                        inwardIndices.push_back(inwardIndex);
                    }
                    else
                    {
                        inwardIndices.insert(inwardIndices.begin(), inwardIndex);
                    }
                    if(inwardIndices.size() == maxInwardIndices)
                    {
                        logger->debug("Creating removal move with {} unchanging and {} changing indices", 
                            alongBorderIndices.size(), inwardIndices.size());
                        auto move = createRemovalMove(stateArray, pathwayVertical, alongBorderIndices, inwardIndices, atLowIndex);
                        inwardIndices.clear();
                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));
                    }
                }
            }
            if(!inwardIndices.empty())
            {
                logger->debug("Creating removal move with {} unchanging and {} changing indices", 
                    alongBorderIndices.size(), inwardIndices.size());
                auto move = createRemovalMove(stateArray, pathwayVertical, alongBorderIndices, inwardIndices, atLowIndex);
                move.execute(stateArray, logger);
                moveList.push_back(std::move(move));
            }
        }
    }
    return true;
}

bool sortArray(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    size_t borderRows = RECOMMENDED_DIST_FROM_OCC_SITES / HALF_ROW_SPACING + 1;
    size_t borderCols = RECOMMENDED_DIST_FROM_OCC_SITES / HALF_COL_SPACING + 1;

    auto pathway = generatePathway(borderRows, borderCols, stateArray);
    auto [labelledPathway, labelCount] = labelPathway(pathway);

    std::stringstream pathwayStream;
    pathwayStream << "Labelled pathway: \n";
    for(Eigen::Index r = 0; r < labelledPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < labelledPathway.cols(); c++)
        {
            pathwayStream << (labelledPathway(r, c) != 0 ? " " : "█");
        }
        pathwayStream << "\n";
    }
    logger->info(pathwayStream.str());

    auto [unusableAtoms, usableTargetSites] = findUnusableAtomsAndUsableTargetSites(stateArray, 
        targetGeometry, compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, logger);

    if(!removeAllDirectlyRemovableUnusableAtoms(stateArray, unusableAtoms,
        borderRows, borderCols, moveList, logger))
    {
        return false;
    }
    if(!createMinimallyInvasiveAccessPathway(stateArray, unusableAtoms, borderRows, borderCols, 
        compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, moveList, 10, logger))
    {
        return false;
    }

    auto penalizedPathway = generatePathway(borderRows, borderCols, stateArray, MIN_DIST_FROM_OCC_SITES, 0);

    std::stringstream ppathwayStream;
    ppathwayStream << "Penalized pathway: \n";
    for(Eigen::Index r = 0; r < penalizedPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < penalizedPathway.cols(); c++)
        {
            ppathwayStream << (penalizedPathway(r, c) == 0 ? " " : "█");
        }
        ppathwayStream << "\n";
    }
    logger->info(ppathwayStream.str());

    pathway = generatePathway(borderRows, borderCols, stateArray);
    std::tie(labelledPathway, labelCount) = labelPathway(pathway);

    if(!findAndExecuteMoves(pathway, labelledPathway, labelCount, penalizedPathway, unusableAtoms, usableTargetSites, compZoneRowStart, 
        compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, moveList,
        stateArray, targetGeometry, logger))
    {
        return false;
    }

    std::stringstream strstream;
    strstream << "PathwayState: \n";
    for(Eigen::Index r = 0; r < labelledPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < labelledPathway.cols(); c++)
        {
            strstream << (labelledPathway(r, c) != 0 ? " " : "█");
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    return true;
}

std::optional<std::vector<ParallelMove>> sortLatticeGeometriesParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    std::shared_ptr<spdlog::logger> logger;
    Config& config = Config::getInstance();
    if((logger = spdlog::get(config.parallelLoggerName)) == nullptr)
    {
        logger = spdlog::basic_logger_mt(config.parallelLoggerName, config.logFileName);
    }
    logger->set_level(spdlog::level::debug);

    if(!compZoneRowEnd - compZoneRowStart == (size_t)targetGeometry.rows())
    {
        logger->error("Comp zone does not have same number of rows as target geometry, aborting");
        return std::nullopt;
    }
    if(!compZoneColEnd - compZoneColStart == (size_t)targetGeometry.cols())
    {
        logger->error("Comp zone does not have same number of cols as target geometry, aborting");
        return std::nullopt;
    }

    std::stringstream strstream;
    strstream << "Initial state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    strstream << (stateArray(r,c) ? "█" : "▒");
                }
                else
                {
                    strstream << (stateArray(r,c) ? "X" : " ");
                }
            }
            else
            {
                strstream << (stateArray(r,c) ? "█" : " ");
            }
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    std::vector<ParallelMove> moveList;
    if(!checkTargetGeometryFeasibility(targetGeometry, logger))
    {
        return std::nullopt;
    }
    if(!sortArray(stateArray, compZoneRowStart, compZoneRowEnd, 
        compZoneColStart, compZoneColEnd, targetGeometry, moveList, logger))
    {
        return std::nullopt;
    }

    std::stringstream endstrstream;
    endstrstream << "Final state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    endstrstream << (stateArray(r,c) != 0 ? "█" : "▒");
                }
                else
                {
                    endstrstream << (stateArray(r,c) != 0 ? "X" : " ");
                }
            }
            else
            {
                endstrstream << (stateArray(r,c) != 0 ? "█" : " ");
            }
        }
        endstrstream << "\n";
    }
    logger->info(endstrstream.str());

    return moveList;
}