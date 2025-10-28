/***
 * Sorting approach that imposes a fixed movement scheme instead of finding the best move in a greedy fashion
 * Idea by Francisco Romão and Jonas Winklmann, Implemented by Jonas Winklmann
 */

#include "sortParallel.hpp"
#include "sortLattice.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <chrono>
#include <omp.h>
#include <ranges>

#include "config.hpp"
#include "spdlog/sinks/basic_file_sink.h"

// Provides access to array data while possibly flipping indices, useful due to independence on dimension
bool& accessStateArrayDimIndepedent(ArrayAccessor& stateArray, 
    Eigen::Index indexXC, Eigen::Index indexAC, bool vertical)
{
    if(vertical)
    {
        return stateArray(indexAC, indexXC);
    }
    else
    {
        return stateArray(indexXC, indexAC);
    }
}

// 0 for start, array size for end, anything in between for start of higher-index part
std::optional<int> determineBestStartPosition(ArrayAccessor& stateArray,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeXC,
    size_t compZoneXCStart, size_t compZoneXCEnd, 
    const std::vector<std::vector<int>>& unusableAtoms, 
    const std::vector<std::vector<int>>& usableAtoms, 
    const std::vector<std::vector<int>>& targetSites)
{
    unsigned int indicesToClear = (sortingChannelWidth + 1) * 2;

    std::optional<int> bestLastIndexBeforeChannel = std::nullopt;
    std::optional<double> bestDist = std::nullopt;

    // Only consider starting from the middle if movement parallelization even makes sense
    if(((vertical && Config::getInstance().aodColLimit > 1 && Config::getInstance().aodRowLimit < Config::getInstance().aodTotalLimit) || 
        (!vertical && Config::getInstance().aodRowLimit > 1 && Config::getInstance().aodColLimit < Config::getInstance().aodTotalLimit)) && arraySizeXC > indicesToClear)
    {
        unsigned int usableBeforeChannel = 0;
        unsigned int usableAfterChannel = std::accumulate(
            usableAtoms.begin() + indicesToClear, usableAtoms.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });

        unsigned int requiredBeforeChannel = std::accumulate(
            targetSites.begin(), targetSites.begin() + sortingChannelWidth + 1, 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });
        unsigned int requiredAfterChannel = std::accumulate(
            targetSites.begin() + sortingChannelWidth + 1, targetSites.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });

        // Iterate over possible starting positions and use the one that has enough atoms on both sides while being as centered as possible
        for(size_t lastIndexBeforeChannel = 0; lastIndexBeforeChannel < arraySizeXC - indicesToClear - 1; lastIndexBeforeChannel++)
        {
            // Update number of existing and required atoms
            usableBeforeChannel += usableAtoms[lastIndexBeforeChannel].size();
            usableAfterChannel -= usableAtoms[lastIndexBeforeChannel + indicesToClear].size();
            requiredBeforeChannel += targetSites[lastIndexBeforeChannel + sortingChannelWidth + 1].size();
            requiredAfterChannel -= targetSites[lastIndexBeforeChannel + sortingChannelWidth + 1].size();

            // Check if atoms numbers suffice. If so, use it if it is the closest index to the center yet
            // This could be done more efficiently
            double dist = abs((double)(arraySizeXC - 1) / 2. - ((double)(lastIndexBeforeChannel + sortingChannelWidth) + 1.5));
            if(usableBeforeChannel >= requiredBeforeChannel && usableAfterChannel >= requiredAfterChannel && 
                (!bestLastIndexBeforeChannel.has_value() || !bestDist.has_value() || dist < bestDist.value()))
            {
                bestLastIndexBeforeChannel = lastIndexBeforeChannel;
                bestDist = dist;
            }
        }
    }

    if(bestLastIndexBeforeChannel.has_value())
    {
        return bestLastIndexBeforeChannel.value() + sortingChannelWidth + 2;
    }
    else
    {
        // If no index was found before, take the first one that has sufficiently many atoms on both sides
        unsigned int totalTargetSites = std::accumulate(targetSites.begin(), targetSites.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });
        unsigned int excessStartingLowIndex = std::accumulate(
            usableAtoms.begin() + sortingChannelWidth + 1, usableAtoms.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); }) - totalTargetSites;
        unsigned int excessStartingHighIndex = std::accumulate(
            usableAtoms.begin(), usableAtoms.end() - sortingChannelWidth - 1, 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); }) - totalTargetSites;
        if(excessStartingLowIndex >= 0 && excessStartingHighIndex >= 0)
        {
            if(compZoneXCStart > arraySizeXC - compZoneXCEnd)
            {
                return arraySizeXC;
            }
            else if(compZoneXCStart < arraySizeXC - compZoneXCEnd)
            {
                return 0;
            }
        }
        if(excessStartingLowIndex >= 0)
        {
            return 0;
        }
        else if(excessStartingHighIndex >= 0)
        {
            return arraySizeXC;
        }
        else
        {
            return std::nullopt;
        }
    }
}

// Remove all atoms from initial sorting channel
void clearSortingChannel(ArrayAccessor& stateArray, int startIndex,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeXC, unsigned int arraySizeAC,
    unsigned int maxTones, double spacingXC, int targetGapXC, int targetGapAC, std::vector<std::vector<int>>& usableAtoms, 
    std::vector<std::vector<int>>& unusableAtoms, std::shared_ptr<spdlog::logger> logger)
{
    if(startIndex <= (int)sortingChannelWidth + 1 || startIndex >= (int)arraySizeXC - (int)sortingChannelWidth - 1)
    {
        // Remove atoms at one side, atoms can just be moved away directly
        bool atLowIndex = startIndex <= (int)sortingChannelWidth + 1;
        int nextIndexToDealWith = 0;
        unsigned int count = sortingChannelWidth + 1;

        // Remove atoms from registers
        if(atLowIndex)
        {
            for(int i = 0; i < (int)count; i++)
            {
                usableAtoms[i].clear();
                unusableAtoms[i].clear();
            }
        }
        else
        {
            for(int i = 0; i < (int)count; i++)
            {
                usableAtoms[arraySizeXC - count + i].clear();
                unusableAtoms[arraySizeXC - count + i].clear();
            }
        }

        // Check whether it is possible and makes sense to remove all indices simultaneously
        if(((vertical && Config::getInstance().aodRowLimit >= count) || (!vertical && Config::getInstance().aodColLimit >= count)) && 
            count * maxTones <= Config::getInstance().aodTotalLimit && spacingXC >= Config::getInstance().minDistFromOccSites)
        {
            while(nextIndexToDealWith < (int)arraySizeAC)
            {
                // Create move to remove atoms from sorting channel
                ParallelMove move;
                ParallelMove::Step start, end;
                std::vector<double> *startSelectionXC, *endSelectionXC, *startSelectionAC, *endSelectionAC;
                if(vertical)
                {
                    startSelectionXC = &start.colSelection;
                    endSelectionXC = &end.colSelection;
                    startSelectionAC = &start.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    startSelectionXC = &start.rowSelection;
                    endSelectionXC = &end.rowSelection;
                    startSelectionAC = &start.colSelection;
                    endSelectionAC = &end.colSelection;
                }
                if(atLowIndex)
                {
                    for(int i = 0; i < (int)count; i++)
                    {
                        startSelectionXC->push_back(i);
                        endSelectionXC->push_back(-(int)count - targetGapXC + 1 + i);
                    }
                }
                else
                {
                    for(int i = 0; i < (int)count; i++)
                    {
                        startSelectionXC->push_back(arraySizeXC - count + i);
                        endSelectionXC->push_back(arraySizeXC + targetGapXC - 1 + i);
                    }
                }

                // Add indices along sorting channel
                for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
                {
                    bool indexRequired = false;
                    for(int i = 0; i < (int)count; i++)
                    {
                        int indexXC = atLowIndex ? i : arraySizeXC - 1 - i;
                        if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWith, vertical))
                        {
                            indexRequired = true;
                        }
                    }
                    if(indexRequired)
                    {
                        startSelectionAC->push_back(nextIndexToDealWith);
                        endSelectionAC->push_back(nextIndexToDealWith);
                    }
                }
                if(startSelectionAC->size() > 0)
                {
                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(end));
                    move.execute(stateArray, logger);
                    moveList.push_back(move);
                }
            }
        }
        else
        {
            // Remove atoms index by index as there are not enough AOD tones or 
            // the spacing does not allow for arbitrary targeting in the outer indices
            for(int i = 0; i < (int)count; i++)
            {
                nextIndexToDealWith = 0;
                while(nextIndexToDealWith < (int)arraySizeAC)
                {
                    // Create move to remove atoms from sorting channel
                    ParallelMove move;
                    ParallelMove::Step start;
                    ParallelMove::Step end;
                    std::vector<double> *startSelectionXC, *endSelectionXC, *startSelectionAC, *endSelectionAC;
                    if(vertical)
                    {
                        startSelectionXC = &start.colSelection;
                        endSelectionXC = &end.colSelection;
                        startSelectionAC = &start.rowSelection;
                        endSelectionAC = &end.rowSelection;
                    }
                    else
                    {
                        startSelectionXC = &start.rowSelection;
                        endSelectionXC = &end.rowSelection;
                        startSelectionAC = &start.colSelection;
                        endSelectionAC = &end.colSelection;
                    }
                    if(atLowIndex)
                    {
                        startSelectionXC->push_back(i);
                        endSelectionXC->push_back(-(int)count - targetGapXC + 1 + i);
                    }
                    else
                    {
                        startSelectionXC->push_back(arraySizeXC - count + i);
                        endSelectionXC->push_back(arraySizeXC + targetGapXC - 1 + i);
                    }

                    // Add indices along sorting channel
                    for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
                    {
                        int indexXC = atLowIndex ? i : arraySizeXC - 1 - i;
                        if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWith, vertical))
                        {
                            startSelectionAC->push_back(nextIndexToDealWith);
                            endSelectionAC->push_back(nextIndexToDealWith);
                        }
                    }
                    if(startSelectionAC->size() > 0)
                    {
                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(end));
                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));
                    }
                }
            }
        }
    }
    else
    {
        for(int i = startIndex - (int)sortingChannelWidth - 1; i <= (int)startIndex + (int)sortingChannelWidth; i++)
        {
            usableAtoms[i].clear();
            unusableAtoms[i].clear();
        }

        // Remove atoms for sorting channels that start from within the array
        std::vector<double> targetIndicesXC;
        unsigned int desiredIndices = 2 * (sortingChannelWidth + 1);
        if(vertical)
        {
            if(Config::getInstance().aodColLimit < desiredIndices)
            {
                desiredIndices = Config::getInstance().aodColLimit;
            }
        }
        else
        {
            if(Config::getInstance().aodRowLimit < desiredIndices)
            {
                desiredIndices = Config::getInstance().aodRowLimit;
            }
        }
        if(desiredIndices * maxTones > Config::getInstance().aodTotalLimit)
        {
            desiredIndices = Config::getInstance().aodTotalLimit / maxTones;
        }
        if(desiredIndices < 1)
        {
            desiredIndices = 1;
        }

        if(desiredIndices >= 2 * (sortingChannelWidth + 1) && spacingXC >= Config::getInstance().minDistFromOccSites)
        {
            // Remove all at once if max index count allows for it
            int nextIndexToDealWith = 0;
            while(nextIndexToDealWith < (int)arraySizeAC)
            {
                ParallelMove move;
                ParallelMove::Step start;
                ParallelMove::Step end;
                std::vector<double> *startSelectionXC, *endSelectionXC, *startSelectionAC, *endSelectionAC;
                if(vertical)
                {
                    startSelectionXC = &start.colSelection;
                    endSelectionXC = &end.colSelection;
                    startSelectionAC = &start.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    startSelectionXC = &start.rowSelection;
                    endSelectionXC = &end.rowSelection;
                    startSelectionAC = &start.colSelection;
                    endSelectionAC = &end.colSelection;
                }
                for(int i = startIndex - (int)sortingChannelWidth - 1; i <= (int)startIndex + (int)sortingChannelWidth; i++)
                {
                    startSelectionXC->push_back(i);
                    endSelectionXC->push_back(i);
                }

                // Add indices along sorting channel
                for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
                {
                    bool indexRequired = false;
                    for(int i = startIndex - (int)sortingChannelWidth - 1; i <= (int)startIndex + (int)sortingChannelWidth; i++)
                    {
                        if(accessStateArrayDimIndepedent(stateArray, i, nextIndexToDealWith, vertical))
                        {
                            indexRequired = true;
                            break;
                        }
                    }
                    if(indexRequired)
                    {
                        startSelectionAC->push_back(nextIndexToDealWith);
                    }
                }
                if(nextIndexToDealWith >= (int)arraySizeAC)
                {
                    // Remove indices towards side closer side
                    int indicesLowerOrEqualMiddle = 0, indicesHigherMiddle = 0;
                    for(const auto& indexAC : *startSelectionAC)
                    {
                        if(indexAC <= (int)arraySizeAC / 2)
                        {
                            indicesLowerOrEqualMiddle++;
                        }
                        else
                        {
                            indicesHigherMiddle++;
                        }
                    }
                    for(int i = 0; i < indicesLowerOrEqualMiddle; i++)
                    {
                        endSelectionAC->push_back(-indicesLowerOrEqualMiddle - targetGapAC + 1 + i);
                    }
                    for(int i = 0; i < indicesHigherMiddle; i++)
                    {
                        endSelectionAC->push_back(arraySizeAC + targetGapAC - 1 + i);
                    }
                }
                else
                {
                    for(int i = 0; i < (int)startSelectionAC->size(); i++)
                    {
                        endSelectionAC->push_back(-(int)startSelectionAC->size() - targetGapAC + 1 + i);
                    }
                }
                if(startSelectionAC->size() > 0)
                {
                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(end));
                    move.execute(stateArray, logger);
                    moveList.push_back(move);
                }
            }
        }
        else
        {
            // Remove atoms index-by-index
            double channelStart = (double)startIndex - sortingChannelWidth - 2 + Config::getInstance().minDistFromOccSites / spacingXC;
            double channelEnd = (double)startIndex + sortingChannelWidth + 1 - Config::getInstance().minDistFromOccSites / spacingXC;
            unsigned int handledIndicesXC = 0;
            while(handledIndicesXC < 2 * (sortingChannelWidth + 1))
            {
                std::vector<double> usedStartIndicesXC, usedEndIndicesXC;
                bool outsideChannel = false;
                for(size_t i = 0; i < desiredIndices && handledIndicesXC < 2 * (sortingChannelWidth + 1) && !outsideChannel; i++)
                {
                    int nextIndex = startIndex;
                    if(handledIndicesXC % 2 == 0)
                    {
                        nextIndex += handledIndicesXC / 2;
                    }
                    else
                    {
                        nextIndex -= handledIndicesXC / 2 - 1;
                    }
                    if(nextIndex < channelStart - DOUBLE_EQUIVALENCE_THRESHOLD || nextIndex > channelEnd + DOUBLE_EQUIVALENCE_THRESHOLD)
                    {
                        // If we are outside the sorting channel, we only take one index 
                        // at a time as shadow traps could otherwise impact adjacent atoms
                        if(!usedStartIndicesXC.empty())
                        {
                            break;
                        }
                        outsideChannel = true;
                    }
                    usedStartIndicesXC.push_back(nextIndex);
                    handledIndicesXC++;
                }
                if(outsideChannel)
                {
                    usedEndIndicesXC.push_back((double)startIndex - 0.5);
                }
                else
                {
                    usedEndIndicesXC = usedStartIndicesXC;
                }

                int nextIndexToDealWith = 0;
                while(nextIndexToDealWith < (int)arraySizeAC)
                {
                    ParallelMove move;
                    ParallelMove::Step start;
                    ParallelMove::Step end;
                    std::vector<double> *startSelectionAC, *endSelectionAC;
                    if(vertical)
                    {
                        start.colSelection = usedStartIndicesXC;
                        end.colSelection = usedEndIndicesXC;
                        startSelectionAC = &start.rowSelection;
                        endSelectionAC = &end.rowSelection;
                    }
                    else
                    {
                        start.rowSelection = usedStartIndicesXC;
                        end.rowSelection = usedEndIndicesXC;
                        startSelectionAC = &start.colSelection;
                        endSelectionAC = &end.colSelection;
                    }

                    for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
                    {
                        bool indexRequired = false;
                        for(int i = channelStart; i < (int)channelEnd; i++)
                        {
                            if(accessStateArrayDimIndepedent(stateArray, i, nextIndexToDealWith, vertical))
                            {
                                indexRequired = true;
                                break;
                            }
                        }
                        if(indexRequired)
                        {
                            startSelectionAC->push_back(nextIndexToDealWith);
                            endSelectionAC->push_back(nextIndexToDealWith);
                        }
                    }
                    if(startSelectionAC->size() > 0)
                    {
                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(end));
                        move.execute(stateArray, logger);
                        moveList.push_back(move);
                    }
                }
            }
        }
    }
}

// Create move with only one index across the channel direction
bool createSingleIndexMoves(ArrayAccessor& stateArray, bool vertical, std::vector<ParallelMove>& moveList, 
    std::vector<int>& startIndices, int dir, std::optional<std::vector<int>*> endIndices, 
    int endIndexXC, unsigned int targetCount,
    int arraySizeXC, int arraySizeAC, int sortingChannelWidth, int indexXC, int targetGapAC, 
    unsigned int maxTonesXC, unsigned int maxTonesAC, double spacingXC, double spacingAC,
    std::vector<std::vector<int>>& usableAtoms, bool parkingMove, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int count = 0;
    auto indexAC = startIndices.begin();

    std::optional<std::vector<int>::iterator> targetIndexAC = std::nullopt;
    if(endIndices.has_value())
    {
        if(endIndices.value()->size() == 0)
        {
            return true;
        }
        targetIndexAC = endIndices.value()->begin();
    }
    
    // Create moves while there are atoms and only as many as requested
    while(indexAC != startIndices.end() && count < targetCount)
    {
        // Create move data
        ParallelMove move;
        ParallelMove::Step start, elbow1, elbow2, end;
        std::vector<double> *startSelectionAC, *elbow1SelectionAC, *elbow2SelectionAC, *endSelectionAC;
        double channelMiddle = (double)indexXC - (double)dir * (double)(sortingChannelWidth + 1) / 2.;
        if(vertical)
        {
            start.colSelection.push_back(indexXC);
            elbow1.colSelection.push_back(channelMiddle);
            elbow2.colSelection.push_back(channelMiddle);
            end.colSelection.push_back(endIndexXC);
            startSelectionAC = &start.rowSelection;
            elbow1SelectionAC = &elbow1.rowSelection;
            elbow2SelectionAC = &elbow2.rowSelection;
            endSelectionAC = &end.rowSelection;
        }
        else
        {
            start.rowSelection.push_back(indexXC);
            elbow1.rowSelection.push_back(channelMiddle);
            elbow2.rowSelection.push_back(channelMiddle);
            end.rowSelection.push_back(endIndexXC);
            startSelectionAC = &start.colSelection;
            elbow1SelectionAC = &elbow1.colSelection;
            elbow2SelectionAC = &elbow2.colSelection;
            endSelectionAC = &end.colSelection;
        }

        if(targetIndexAC.has_value())
        {
            double sqDist = (endIndexXC - indexXC) * (endIndexXC - indexXC) * spacingXC * spacingXC;
            double minElbow4ToTargetDist = sqrt(Config::getInstance().minDistFromOccSites * Config::getInstance().minDistFromOccSites - 
                (spacingAC / 2) * (spacingAC / 2)) / spacingXC;

            bool needToMoveBetweenTrapsAfterSortingChannel = 
                sqDist > Config::getInstance().maxSubmoveDistInPenalizedArea * Config::getInstance().maxSubmoveDistInPenalizedArea && 
                sqDist > minElbow4ToTargetDist * minElbow4ToTargetDist;

            // If there are target indices, i.e, not a removal move, iterate over start and end indices and add to move accordingly
            while(indexAC != startIndices.end() && targetIndexAC.value() != endIndices.value()->end() && 
                startSelectionAC->size() < maxTonesAC && count < targetCount)
            {
                count++;
                startSelectionAC->push_back(*indexAC);
                elbow1SelectionAC->push_back(*indexAC);
                elbow2SelectionAC->push_back(*targetIndexAC.value());
                endSelectionAC->push_back(*targetIndexAC.value());

                indexAC = startIndices.erase(indexAC);
                if(parkingMove)
                {
                    usableAtoms[endIndexXC].push_back(*targetIndexAC.value());
                }

                // Advance target index iterator
                if(endIndices.has_value())
                {
                    targetIndexAC = endIndices.value()->erase(targetIndexAC.value());
                }
                else
                {
                    // Should never be reached
                    logger->error("End indices don't exist even though its iterator has a value. Aborting.");
                    return false;
                }
            }
            if(startSelectionAC->size() > 0)
            {
                // Execute move if it contains something
                if(parkingMove)
                {
                    std::sort(elbow2SelectionAC->begin(), elbow2SelectionAC->end());
                    std::sort(endSelectionAC->begin(), endSelectionAC->end());
                }

                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(elbow1));

                if(needToMoveBetweenTrapsAfterSortingChannel)
                {
                    // Move between traps if distance longer than allowed
                    for(auto indexAC = startSelectionAC->begin(), elbow2IndexAC = elbow2SelectionAC->begin(); 
                        indexAC != startSelectionAC->end() && elbow2IndexAC != elbow2SelectionAC->end(); 
                        indexAC++, elbow2IndexAC++)
                    {
                        if(*elbow2IndexAC > *indexAC)
                        {
                            *elbow2IndexAC -= 0.5;
                        }
                        else
                        {
                            *elbow2IndexAC += 0.5;
                        }
                    }

                    // With careful consideration of neighboring atoms, one might get away with omitting elbow4
                    ParallelMove::Step elbow3, elbow4;
                    if(vertical)
                    {
                        elbow3.rowSelection = elbow2.rowSelection;
                        elbow3.colSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow4.rowSelection = end.rowSelection;
                        elbow4.colSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                    }
                    else
                    {
                        elbow3.rowSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow3.colSelection = elbow2.colSelection;
                        elbow4.rowSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow4.colSelection = end.colSelection;
                    }
                    move.steps.push_back(std::move(elbow2));
                    move.steps.push_back(std::move(elbow3));
                    move.steps.push_back(std::move(elbow4));
                }
                else
                {
                    move.steps.push_back(std::move(elbow2));
                }

                move.steps.push_back(std::move(end));
                move.execute(stateArray, logger);
                moveList.push_back(std::move(move));
            }
            else
            {
                return true;
            }
        }
        else
        {
            // There are no target indices. This should be a removal move
            int indicesLowerOrEqualMiddle = 0, indicesHigherMiddle = 0;
            for(; indexAC != startIndices.end() && startSelectionAC->size() < maxTonesAC && 
                count < targetCount; indexAC++)
            {
                count++;
                startSelectionAC->push_back(*indexAC);
                elbow1SelectionAC->push_back(*indexAC);
                if(*indexAC <= (int)arraySizeAC / 2)
                {
                    indicesLowerOrEqualMiddle++;
                }
                else
                {
                    indicesHigherMiddle++;
                }
            }
            // Move towards closer side
            if(indicesLowerOrEqualMiddle > 0 || indicesHigherMiddle > 0)
            {
                for(int i = 0; i < indicesLowerOrEqualMiddle; i++)
                {
                    elbow2SelectionAC->push_back(-indicesLowerOrEqualMiddle - targetGapAC + 1 + i);
                }
                for(int i = 0; i < indicesHigherMiddle; i++)
                {
                    elbow2SelectionAC->push_back(arraySizeAC + targetGapAC - 1 + i);
                }
                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(elbow1));
                move.steps.push_back(std::move(elbow2));
                move.execute(stateArray, logger);
                moveList.push_back(std::move(move));
            }
            else
            {
                return true;
            }
        }
    }

    return true;
}

// Create move that combines moves from both sides in a smart way
bool createCombinedMoves(ArrayAccessor& stateArray, bool vertical, std::vector<ParallelMove>& moveList,
    std::set<int>& excludedStartIndices,
    std::vector<int>& startIndicesLowIndex, std::vector<int>& startIndicesHighIndex,
    std::optional<std::vector<int>*> endIndicesLowIndex, int endIndexXCLowIndex, 
    std::optional<std::vector<int>*> endIndicesHighIndex, int endIndexXCHighIndex, 
    unsigned int targetCountLowIndex, unsigned int targetCountHighIndex, 
    int arraySizeXC, int arraySizeAC, int sortingChannelWidth, int indexXC[2], int targetGapAC, 
    unsigned int maxTonesXC, unsigned int maxTonesAC, double spacingXC, double spacingAC,
    std::vector<std::vector<int>>& usableAtoms, bool parkingMove, std::shared_ptr<spdlog::logger> logger)
{
    std::set<int> sharedIndices;

    if(startIndicesLowIndex.size() > 0 && startIndicesHighIndex.size() > 0 &&
        maxTonesXC >= 2 && 2 * maxTonesAC <= Config::getInstance().aodTotalLimit && 
        (targetCountLowIndex > 0 || targetCountHighIndex > 0) && indexXC[0] > 0 && indexXC[1] < arraySizeXC)
    {
        // If spacing is sufficient, use union
        // Otherwise, shadow traps might interfere so in that case only use intersection
        std::set_intersection(startIndicesLowIndex.begin(), startIndicesLowIndex.end(),
            startIndicesHighIndex.begin(), startIndicesHighIndex.end(), std::inserter(sharedIndices, sharedIndices.begin()));
        for(const auto& excluded : excludedStartIndices)
        {
            sharedIndices.erase(excluded);
        }

        std::optional<std::vector<int>> sharedTargetIndices = std::nullopt;
        if(endIndicesLowIndex.has_value() && endIndicesHighIndex.has_value())
        {
            sharedTargetIndices = std::vector<int>();
            std::set_intersection((*endIndicesLowIndex.value()).begin(), (*endIndicesLowIndex.value()).end(),
                (*endIndicesHighIndex.value()).begin(), (*endIndicesHighIndex.value()).end(), 
                std::inserter(sharedTargetIndices.value(), sharedTargetIndices.value().begin()));
        }    

        if(sharedIndices.size() > 0 && (!sharedTargetIndices.has_value() || 
            sharedTargetIndices.value().size() > 0))
        {
            // Calculate move count when using separate moves
            unsigned int lowIndicesSeparate = targetCountLowIndex;
            if(startIndicesLowIndex.size() < lowIndicesSeparate)
            {
                lowIndicesSeparate = startIndicesLowIndex.size();
            }
            if(endIndicesLowIndex.has_value() && endIndicesLowIndex.value()->size() < lowIndicesSeparate)
            {
                lowIndicesSeparate = endIndicesLowIndex.value()->size();
            }
            unsigned int highIndicesSeparate = targetCountHighIndex;
            if(startIndicesHighIndex.size() < highIndicesSeparate)
            {
                highIndicesSeparate = startIndicesHighIndex.size();
            }
            if(endIndicesHighIndex.has_value() && endIndicesHighIndex.value()->size() < highIndicesSeparate)
            {
                highIndicesSeparate = endIndicesHighIndex.value()->size();
            }
            unsigned int moveCountSeparate = (lowIndicesSeparate - 1) / maxTonesAC + 1 + 
                (highIndicesSeparate - 1) / maxTonesAC + 1;

            // Calculate move count when combining moves
            unsigned int usedSharedIndices = sharedIndices.size();
            if(sharedTargetIndices.has_value() && sharedTargetIndices.value().size() < usedSharedIndices)
            {
                usedSharedIndices = sharedTargetIndices.value().size();
            }
            unsigned int moveCountCombined = (lowIndicesSeparate - usedSharedIndices - 1) / maxTonesAC + 1 + 
                (highIndicesSeparate - usedSharedIndices - 1) / maxTonesAC + 1 + 
                (usedSharedIndices - 1) / maxTonesAC + 1;
            unsigned int fullCombinedMovesCount = usedSharedIndices / maxTonesAC;
            unsigned int moveCountSplitUnfilledCombinedMove = moveCountSeparate - fullCombinedMovesCount;

            unsigned int indicesInUnfilledCombinedMove = usedSharedIndices % maxTonesAC;
            unsigned int indicesInUnfilledLowMove = (lowIndicesSeparate - usedSharedIndices) % maxTonesAC;
            unsigned int indicesInUnfilledHighMove = (highIndicesSeparate - usedSharedIndices) % maxTonesAC;
            unsigned int moveCountAddedSinglesToCombinedMove = moveCountCombined;
            if(indicesInUnfilledCombinedMove + indicesInUnfilledLowMove + indicesInUnfilledHighMove <= maxTonesAC &&
                indicesInUnfilledLowMove > 0 && indicesInUnfilledHighMove > 0)
            {
                moveCountAddedSinglesToCombinedMove -= 2;
            }
            else if((indicesInUnfilledCombinedMove + indicesInUnfilledLowMove <= maxTonesAC && 
                indicesInUnfilledLowMove > 0) || 
                (indicesInUnfilledCombinedMove + indicesInUnfilledHighMove <= maxTonesAC && 
                indicesInUnfilledHighMove > 0) )
            {
                moveCountAddedSinglesToCombinedMove--;
            }

            if(spacingXC >= Config::getInstance().minDistFromOccSites && moveCountAddedSinglesToCombinedMove < moveCountCombined && 
                moveCountAddedSinglesToCombinedMove < moveCountSplitUnfilledCombinedMove)
            {
                if(indicesInUnfilledCombinedMove + indicesInUnfilledLowMove <= maxTonesAC)
                {
                    indicesInUnfilledCombinedMove += indicesInUnfilledLowMove;
                    std::set<int> indicesInOnlyLowStartRegister;
                    std::set_difference(startIndicesLowIndex.begin(), startIndicesLowIndex.end(), 
                        startIndicesHighIndex.begin(), startIndicesHighIndex.end(),
                        std::inserter(indicesInOnlyLowStartRegister, indicesInOnlyLowStartRegister.begin()));
                    unsigned int insertCount = 0;
                    for(auto indexInOnlyLowStartRegister = indicesInOnlyLowStartRegister.begin();
                        indexInOnlyLowStartRegister != indicesInOnlyLowStartRegister.end() && 
                        insertCount < indicesInUnfilledLowMove; 
                        indexInOnlyLowStartRegister++)
                    {
                        if(!excludedStartIndices.contains(*indexInOnlyLowStartRegister))
                        {
                            insertCount++;
                            sharedIndices.insert(*indexInOnlyLowStartRegister);
                        }
                    }
                }
                if(indicesInUnfilledCombinedMove + indicesInUnfilledHighMove <= maxTonesAC)
                {
                    std::set<int> indicesInOnlyHighStartRegister;
                    std::set_difference(startIndicesHighIndex.begin(), startIndicesHighIndex.end(), 
                        startIndicesLowIndex.begin(), startIndicesLowIndex.end(),
                        std::inserter(indicesInOnlyHighStartRegister, indicesInOnlyHighStartRegister.begin()));
                    unsigned int insertCount = 0;
                    for(auto indexInOnlyHighStartRegister = indicesInOnlyHighStartRegister.begin();
                        indexInOnlyHighStartRegister != indicesInOnlyHighStartRegister.end() && 
                        insertCount < indicesInUnfilledHighMove; 
                        indexInOnlyHighStartRegister++)
                    {
                        if(!excludedStartIndices.contains(*indexInOnlyHighStartRegister))
                        {
                            insertCount++;
                            sharedIndices.insert(*indexInOnlyHighStartRegister);
                        }
                    }
                }
                moveCountCombined = moveCountAddedSinglesToCombinedMove;
            }
            else if(moveCountSplitUnfilledCombinedMove < moveCountCombined)
            {
                auto endIndex = sharedIndices.begin();
                std::advance(endIndex, sharedIndices.size() - fullCombinedMovesCount * maxTonesAC);
                sharedIndices.erase(sharedIndices.begin(), endIndex);
                moveCountCombined = moveCountSplitUnfilledCombinedMove;
            }

            if(moveCountSeparate <= moveCountCombined)
            {
                sharedIndices.clear();
            }

            std::optional<std::vector<int>::iterator> targetIndexAC = std::nullopt;
            if(sharedTargetIndices.has_value())
            {
                targetIndexAC = sharedTargetIndices.value().begin();
            }
            auto indexAC = sharedIndices.begin();

            double minElbow4ToTargetDist;
            bool needToMoveBetweenTrapsAfterSortingChannel;

            minElbow4ToTargetDist = sqrt(Config::getInstance().minDistFromOccSites * Config::getInstance().minDistFromOccSites - 
                (spacingAC / 2.) * (spacingAC / 2.)) / spacingXC;
            needToMoveBetweenTrapsAfterSortingChannel = 
                (((double)indexXC[0] - (double)(sortingChannelWidth + 1) / 2. - (double)endIndexXCLowIndex) * 
                (double)spacingXC > (double)Config::getInstance().maxSubmoveDistInPenalizedArea ||
                ((double)endIndexXCHighIndex - (double)(sortingChannelWidth + 1) / 2. - (double)indexXC[1]) * 
                (double)spacingXC > (double)Config::getInstance().maxSubmoveDistInPenalizedArea) && 
                (((double)indexXC[0] - (double)(sortingChannelWidth + 1) / 2. - (double)endIndexXCLowIndex) * 
                (double)spacingXC > minElbow4ToTargetDist ||
                ((double)endIndexXCHighIndex - (double)(sortingChannelWidth + 1) / 2. - (double)indexXC[1]) * 
                (double)spacingXC > minElbow4ToTargetDist);

            while(indexAC != sharedIndices.end() && (!targetIndexAC.has_value() || 
                targetIndexAC.value() != sharedTargetIndices.value().end()))
            {
                ParallelMove move;
                ParallelMove::Step start, elbow1, elbow2, end;
                std::vector<double> *startSelectionAC, *elbow1SelectionAC, *elbow2SelectionAC, *endSelectionAC;
                if(vertical)
                {
                    start.colSelection.push_back(indexXC[0]);
                    start.colSelection.push_back(indexXC[1]);
                    elbow1.colSelection.push_back((double)indexXC[0] + (double)(sortingChannelWidth + 1) / 2.);
                    elbow1.colSelection.push_back((double)indexXC[1] - (double)(sortingChannelWidth + 1) / 2.);
                    elbow2.colSelection = elbow1.colSelection;
                    end.colSelection.push_back(endIndexXCLowIndex);
                    end.colSelection.push_back(endIndexXCHighIndex);
                    startSelectionAC = &start.rowSelection;
                    elbow1SelectionAC = &elbow1.rowSelection;
                    elbow2SelectionAC = &elbow2.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(indexXC[0]);
                    start.rowSelection.push_back(indexXC[1]);
                    elbow1.rowSelection.push_back((double)indexXC[0] + (double)(sortingChannelWidth + 1) / 2.);
                    elbow1.rowSelection.push_back((double)indexXC[1] - (double)(sortingChannelWidth + 1) / 2.);
                    elbow2.rowSelection = elbow1.rowSelection;
                    end.rowSelection.push_back(endIndexXCLowIndex);
                    end.rowSelection.push_back(endIndexXCHighIndex);
                    startSelectionAC = &start.colSelection;
                    elbow1SelectionAC = &elbow1.colSelection;
                    elbow2SelectionAC = &elbow2.colSelection;
                    endSelectionAC = &end.colSelection;
                }
                if(targetIndexAC.has_value())
                {
                    for(; indexAC != sharedIndices.end() && targetIndexAC.value() != sharedTargetIndices.value().end() && 
                        startSelectionAC->size() < maxTonesAC; indexAC++, targetIndexAC.value()++)
                    {
                        startSelectionAC->push_back(*indexAC);
                        elbow1SelectionAC->push_back(*indexAC);
                        elbow2SelectionAC->push_back(*targetIndexAC.value());
                        endSelectionAC->push_back(*targetIndexAC.value());

                        bool lowIndexAtom = std::erase(usableAtoms[indexXC[0]], *indexAC) > 0;
                        bool highIndexAtom = std::erase(usableAtoms[indexXC[1]], *indexAC) > 0;
                        std::erase(startIndicesLowIndex, *indexAC);
                        std::erase(startIndicesHighIndex, *indexAC);
                        if(lowIndexAtom && endIndicesLowIndex.has_value())
                        {
                            targetCountLowIndex--;
                            std::erase(*endIndicesLowIndex.value(), *targetIndexAC.value());
                        }
                        if(highIndexAtom && endIndicesHighIndex.has_value())
                        {
                            targetCountHighIndex--;
                            std::erase(*endIndicesHighIndex.value(), *targetIndexAC.value());
                        }
                        if(parkingMove)
                        {
                            if(lowIndexAtom)
                            {
                                usableAtoms[endIndexXCLowIndex].push_back(*targetIndexAC.value());
                            }
                            if(highIndexAtom)
                            {
                                usableAtoms[endIndexXCHighIndex].push_back(*targetIndexAC.value());
                            }
                        }
                    }
                    if(startSelectionAC->size() > 0)
                    {
                        if(parkingMove)
                        {
                            std::sort(elbow2SelectionAC->begin(), elbow2SelectionAC->end());
                            std::sort(endSelectionAC->begin(), endSelectionAC->end());
                        }

                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(elbow1));

                        if(needToMoveBetweenTrapsAfterSortingChannel)
                        {
                            // Move between traps if distance longer than allowed
                            for(auto indexAC = startSelectionAC->begin(), elbow2IndexAC = elbow2SelectionAC->begin(); 
                                indexAC != startSelectionAC->end() && elbow2IndexAC != elbow2SelectionAC->end(); 
                                indexAC++, elbow2IndexAC++)
                            {
                                if(*elbow2IndexAC > *indexAC)
                                {
                                    *elbow2IndexAC -= 0.5;
                                }
                                else
                                {
                                    *elbow2IndexAC += 0.5;
                                }
                            }

                            ParallelMove::Step elbow3, elbow4;
                            if(vertical)
                            {
                                elbow3.rowSelection = elbow2.rowSelection;
                                elbow3.colSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                                elbow3.colSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                                elbow4.rowSelection = end.rowSelection;
                                elbow4.colSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                                elbow4.colSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                            }
                            else
                            {
                                elbow3.rowSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                                elbow3.rowSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                                elbow3.colSelection = elbow2.colSelection;
                                elbow4.rowSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                                elbow4.rowSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                                elbow4.colSelection = end.colSelection;
                            }
                            move.steps.push_back(std::move(elbow2));
                            move.steps.push_back(std::move(elbow3));
                            move.steps.push_back(std::move(elbow4));
                        }
                        else
                        {
                            move.steps.push_back(std::move(elbow2));
                        }

                        move.steps.push_back(std::move(end));
                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));
                    }
                }
                else
                {
                    int indicesLowerOrEqualMiddle = 0, indicesHigherMiddle = 0;
                    for(; indexAC != sharedIndices.end() && startSelectionAC->size() < maxTonesAC; indexAC++)
                    {
                        startSelectionAC->push_back(*indexAC);
                        elbow1SelectionAC->push_back(*indexAC);
                        std::erase(usableAtoms[indexXC[0]], *indexAC);
                        std::erase(usableAtoms[indexXC[1]], *indexAC);
                        std::erase(startIndicesLowIndex, *indexAC);
                        std::erase(startIndicesHighIndex, *indexAC);
                        if(*indexAC <= (int)arraySizeAC / 2)
                        {
                            indicesLowerOrEqualMiddle++;
                        }
                        else
                        {
                            indicesHigherMiddle++;
                        }
                    }
                    if(indicesLowerOrEqualMiddle > 0 || indicesHigherMiddle > 0)
                    {
                        for(int i = 0; i < indicesLowerOrEqualMiddle; i++)
                        {
                            elbow2SelectionAC->push_back(-indicesLowerOrEqualMiddle - targetGapAC + 1 + i);
                        }
                        for(int i = 0; i < indicesHigherMiddle; i++)
                        {
                            elbow2SelectionAC->push_back(arraySizeAC + targetGapAC - 1 + i);
                        }
                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(elbow1));
                        move.steps.push_back(std::move(elbow2));
                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));
                    }
                }
            }
        }
    }

    // For remaining indices, create individual moves
    if(indexXC[0] >= 0 && startIndicesLowIndex.size() > 0 && targetCountLowIndex > 0)
    {
        if(!createSingleIndexMoves(stateArray, vertical, moveList, startIndicesLowIndex, -1, 
            endIndicesLowIndex, endIndexXCLowIndex, 
            targetCountLowIndex, arraySizeXC, arraySizeAC, sortingChannelWidth, 
            indexXC[0], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, 
            usableAtoms, parkingMove, logger))
        {
            return false;
        }
    }
    if(indexXC[1] < arraySizeXC && startIndicesHighIndex.size() > 0 && targetCountHighIndex > 0)
    {
        if(!createSingleIndexMoves(stateArray, vertical, moveList, startIndicesHighIndex, 1, 
            endIndicesHighIndex, endIndexXCHighIndex, 
            targetCountHighIndex, arraySizeXC, arraySizeAC, sortingChannelWidth, 
            indexXC[1], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC,
            usableAtoms, parkingMove, logger))
        {
            return false;
        }
    }

    return true;
}

bool sortRemainingRowsOrCols(ArrayAccessor& stateArray, int startIndex,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeAC, 
    unsigned int arraySizeXC, unsigned int maxTonesXC, unsigned int maxTonesAC, double spacingXC, double spacingAC, std::vector<std::vector<int>>& usableAtoms, 
    std::vector<std::vector<int>>& unusableAtoms, std::vector<std::vector<int>>& targetSites, 
    size_t compZoneXCStart, size_t compZoneXCEnd, size_t compZoneACStart, size_t compZoneACEnd,
    int targetGapXC, int targetGapAC, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<int> parkingSpotsPerSuitableIndexXC;
    for(int dist = targetGapAC; dist < (int)compZoneACStart || compZoneACEnd - 1 + dist < arraySizeAC; dist += targetGapAC)
    {
        if((int)compZoneACStart - dist > 0)
        {
            parkingSpotsPerSuitableIndexXC.push_back(compZoneACStart - dist);
        }
        if(compZoneACEnd - 1 + dist < arraySizeAC)
        {
            parkingSpotsPerSuitableIndexXC.push_back(compZoneACEnd - 1 + dist);
        }
    }
    
    // If sorting channel starts in the middle, there are two indices: one going up and one down, 
    // [0] means index is decreasing, [1] means increasing
    int indexXC[2], currentTargetIndexXC[2];
    if(startIndex == 0)
    {
        indexXC[0] = -1;
        indexXC[1] = sortingChannelWidth + 1;
        currentTargetIndexXC[0] = -1;
        currentTargetIndexXC[1] = compZoneXCStart;
    }
    else if(startIndex == (int)arraySizeXC)
    {
        indexXC[0] = arraySizeXC;
        indexXC[1] = arraySizeXC - sortingChannelWidth - 2;
        currentTargetIndexXC[0] = arraySizeXC;
        currentTargetIndexXC[1] = compZoneXCEnd - 1;
    }
    else
    {
        indexXC[0] = startIndex - sortingChannelWidth - 2;
        indexXC[1] = startIndex + sortingChannelWidth + 1;
        currentTargetIndexXC[0] = startIndex - 1;
        currentTargetIndexXC[1] = startIndex;
    }
    unsigned int requiredAtoms[2] = {0, 0};
    std::vector<int> parkingSpotsRemainingAtCurrentIndexXC[2];
    for(size_t i = 0; i < 2; i++)
    {
        if((currentTargetIndexXC[i] - compZoneXCStart - 1 + targetGapXC) % targetGapXC == 0)
        {
            parkingSpotsRemainingAtCurrentIndexXC[i] = parkingSpotsPerSuitableIndexXC;
        }
    }
    int dir[2] = {-1, 1};

    unsigned int totalRequiredAtoms = std::accumulate(targetSites.begin(), targetSites.end(), 0u, 
        [](unsigned int init, const auto& elem) { return init + elem.size(); });
    size_t targetSitesInCurrentTarget[2] = {0,0};
    if(currentTargetIndexXC[0] >= 0 && currentTargetIndexXC[0] < (int)targetSites.size())
    {
        targetSitesInCurrentTarget[0] = targetSites[currentTargetIndexXC[0]].size();
    }
    if(currentTargetIndexXC[1] >= 0 && currentTargetIndexXC[1] < (int)targetSites.size())
    {
        targetSitesInCurrentTarget[1] = targetSites[currentTargetIndexXC[1]].size();
    }

    while(indexXC[0] >= 0 || indexXC[1] < (int)arraySizeXC)
    {
        logger->debug("indexXC[0]: {}, indexXC[1]: {}", indexXC[0], indexXC[1]);
        std::stringstream unusableAtomsStr;
        if(indexXC[0] >= 0 && indexXC[0] < (int)unusableAtoms.size())
        {
            unusableAtomsStr << "unusableAtoms[indexXC[0]]: ";
            for(auto unusableAtom : unusableAtoms[indexXC[0]])
            {
                unusableAtomsStr << unusableAtom << ", ";
            }
            unusableAtomsStr << "\nusableAtoms[indexXC[0]]: ";
            for(auto usableAtom : usableAtoms[indexXC[0]])
            {
                unusableAtomsStr << usableAtom << ", ";
            }
        }
        if(indexXC[1] >= 0 && indexXC[1] < (int)unusableAtoms.size())
        {
            unusableAtomsStr << "\nunusableAtoms[indexXC[1]]: ";
            for(auto unusableAtom : unusableAtoms[indexXC[1]])
            {
                unusableAtomsStr << unusableAtom << ", ";
            }
            unusableAtomsStr << "\nusableAtoms[indexXC[1]]: ";
            for(auto usableAtom : usableAtoms[indexXC[1]])
            {
                unusableAtomsStr << usableAtom << ", ";
            }
        }
        logger->debug(unusableAtomsStr.str());

        if(indexXC[0] < (int)compZoneXCStart && indexXC[1] >= (int)compZoneXCEnd && totalRequiredAtoms == 0)
        {
            // If both indices are outside computational zone and we don't need more atoms, then we are done
            return true;
        }
        size_t targetIndexXC[2];
        unsigned int parkingSpots[2];
        std::set<int> excludedIndicesForRemovalAndParkingMoves;
        for(size_t i = 0; i < 2; i++)
        {
            if(indexXC[i] >= 0 && indexXC[i] < (int)arraySizeXC)
            {
                targetIndexXC[i] = indexXC[i] - dir[i] * (sortingChannelWidth + 1);
                requiredAtoms[i] = targetSites[currentTargetIndexXC[i]].size();
                parkingSpots[i] = parkingSpotsRemainingAtCurrentIndexXC[i].size();

                for(int indexXCTowardsLastTargetIndex = currentTargetIndexXC[i] + dir[i]; 
                    indexXCTowardsLastTargetIndex * dir[i] <= (int)targetIndexXC[i] * dir[i]; 
                    indexXCTowardsLastTargetIndex += dir[i])
                {
                    requiredAtoms[i] += targetSites[indexXCTowardsLastTargetIndex].size();
                    if((indexXCTowardsLastTargetIndex - compZoneXCStart - 1 + targetGapXC) % targetGapXC == 0)
                    {
                        parkingSpots[i] += parkingSpotsPerSuitableIndexXC.size();
                    }
                }
                logger->debug("i: {}, indexXC: {}, requiredAtoms: {}, totalRequiredAtoms: {}, parkingSpots: {}", 
                    i, indexXC[i], requiredAtoms[i], totalRequiredAtoms, parkingSpots[i]);

                // Excess atoms that cannot be used for filling target sites or parking spots are thrown away
                if(targetIndexXC[i] >= compZoneXCStart && targetIndexXC[i] < compZoneXCEnd)
                {
                    while(usableAtoms[indexXC[i]].size() > requiredAtoms[i] + parkingSpots[i])
                    {
                        if(usableAtoms[indexXC[i]][0] > (int)arraySizeAC - usableAtoms[indexXC[i]].back() - 1)
                        {
                            unusableAtoms[indexXC[i]].insert(std::upper_bound(unusableAtoms[indexXC[i]].begin(), 
                                unusableAtoms[indexXC[i]].end(), usableAtoms[indexXC[i]][0]), usableAtoms[indexXC[i]][0]);
                            usableAtoms[indexXC[i]].erase(usableAtoms[indexXC[i]].begin());
                        }
                        else
                        {
                            unusableAtoms[indexXC[i]].insert(std::upper_bound(unusableAtoms[indexXC[i]].begin(), 
                                unusableAtoms[indexXC[i]].end(), usableAtoms[indexXC[i]].back()), usableAtoms[indexXC[i]].back());
                            usableAtoms[indexXC[i]].pop_back();
                        }
                    }
                }

                for(const auto& usableIndex : usableAtoms[indexXC[i]])
                {
                    for(int excludedIndex = usableIndex - targetGapAC + 1; excludedIndex < usableIndex + targetGapAC; excludedIndex++)
                    {
                        excludedIndicesForRemovalAndParkingMoves.insert(excludedIndex);
                    }
                }
            }
        }
        
        // Remove unusable atoms
        if(indexXC[0] >= 0 && indexXC[1] < (int)arraySizeXC)
        {
            if(!createCombinedMoves(stateArray, vertical, moveList, excludedIndicesForRemovalAndParkingMoves, 
                unusableAtoms[indexXC[0]], unusableAtoms[indexXC[1]], 
                std::nullopt, 0, std::nullopt, 0, unusableAtoms[indexXC[0]].size(), 
                unusableAtoms[indexXC[1]].size(), arraySizeXC, arraySizeAC, sortingChannelWidth, 
                indexXC, targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
            {
                return false;
            }
        }
        else if(indexXC[0] >= 0)
        {
            if(!createSingleIndexMoves(stateArray, vertical, moveList, unusableAtoms[indexXC[0]], -1, 
                std::nullopt, 0, unusableAtoms[indexXC[0]].size(), 
                arraySizeXC, arraySizeAC, sortingChannelWidth, 
                indexXC[0], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
            {
                return false;
            }
        }
        else if(indexXC[1] < (int)arraySizeXC)
        {
            if(!createSingleIndexMoves(stateArray, vertical, moveList, unusableAtoms[indexXC[1]], 1, 
                std::nullopt, 0, unusableAtoms[indexXC[1]].size(), 
                arraySizeXC, arraySizeAC, sortingChannelWidth, 
                indexXC[1], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
            {
                return false;
            }
        };

        while((indexXC[0] >= 0 && !usableAtoms[indexXC[0]].empty()) || 
            (indexXC[1] < (int)arraySizeXC && !usableAtoms[indexXC[1]].empty()))
        {
            bool lowIndexContainsAtoms = indexXC[0] >= 0 && !usableAtoms[indexXC[0]].empty();
            bool highIndexContainsAtoms = indexXC[1] < (int)arraySizeXC && !usableAtoms[indexXC[1]].empty();
            // If there are too many atoms to use, move excess atoms to parking spots
            if(lowIndexContainsAtoms && highIndexContainsAtoms)
            {
                unsigned int atomsToParkLowIndex = usableAtoms[indexXC[0]].size() > requiredAtoms[0] ? 
                    usableAtoms[indexXC[0]].size() - requiredAtoms[0] : 0;
                unsigned int atomsToParkHighIndex = usableAtoms[indexXC[1]].size() > requiredAtoms[1] ? 
                    usableAtoms[indexXC[1]].size() - requiredAtoms[1] : 0;
                if(atomsToParkLowIndex > 0 || atomsToParkHighIndex > 0)
                {
                    logger->debug("Parking {} and {} atoms", atomsToParkLowIndex, atomsToParkHighIndex);
                    if(!createCombinedMoves(stateArray, vertical, moveList, excludedIndicesForRemovalAndParkingMoves,
                        usableAtoms[indexXC[0]], usableAtoms[indexXC[1]], 
                        &parkingSpotsRemainingAtCurrentIndexXC[0], currentTargetIndexXC[0], 
                        &parkingSpotsRemainingAtCurrentIndexXC[1], currentTargetIndexXC[1],
                        atomsToParkLowIndex, atomsToParkHighIndex, arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC, targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, true, logger))
                    {
                        return false;
                    }
                }
            }
            else if(lowIndexContainsAtoms)
            {
                unsigned int atomsToParkLowIndex = usableAtoms[indexXC[0]].size() > requiredAtoms[0] ? 
                    usableAtoms[indexXC[0]].size() - requiredAtoms[0] : 0;
                if(atomsToParkLowIndex > 0)
                {
                    logger->debug("Parking {} atoms", atomsToParkLowIndex);
                    if(!createSingleIndexMoves(stateArray, vertical, moveList, usableAtoms[indexXC[0]], -1, 
                        &parkingSpotsRemainingAtCurrentIndexXC[0], currentTargetIndexXC[0], 
                        atomsToParkLowIndex, arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC[0], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, true, logger))
                    {
                        return false;
                    }
                }
            }
            else if(highIndexContainsAtoms)
            {
                unsigned int atomsToParkHighIndex = usableAtoms[indexXC[1]].size() > requiredAtoms[1] ?
                    usableAtoms[indexXC[1]].size() - requiredAtoms[1] : 0;
                if(atomsToParkHighIndex > 0)
                {
                    logger->debug("Parking {} atoms", atomsToParkHighIndex);
                    if(!createSingleIndexMoves(stateArray, vertical, moveList, usableAtoms[indexXC[1]], 1, 
                        &parkingSpotsRemainingAtCurrentIndexXC[1], currentTargetIndexXC[1], 
                        atomsToParkHighIndex, arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC[1], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, true, logger))
                    {
                        return false;
                    }
                }
            }

            // Increase row if there are no target sites
            bool atLeastOneWasMoved = false;
            bool lowIndexSideExists = indexXC[0] >= 0 && indexXC[0] < (int)arraySizeXC && 
                currentTargetIndexXC[0] >= 0 && currentTargetIndexXC[0] < (int)targetSites.size();
            bool highIndexSideExists = indexXC[1] >= 0 && indexXC[1] < (int)arraySizeXC && 
                currentTargetIndexXC[1] >= 0 && currentTargetIndexXC[1] < (int)targetSites.size();

            if(lowIndexSideExists && targetSites[currentTargetIndexXC[0]].empty())
            {
                totalRequiredAtoms -= targetSitesInCurrentTarget[0];
                targetSitesInCurrentTarget[0] = 0;

                if(currentTargetIndexXC[0] > (int)targetIndexXC[0] && currentTargetIndexXC[0] > 0)
                {
                    atLeastOneWasMoved = true;
                    currentTargetIndexXC[0]--;
                    if((currentTargetIndexXC[0] - compZoneXCStart + 1) % targetGapXC == 0)
                    {
                        parkingSpotsRemainingAtCurrentIndexXC[0] = parkingSpotsPerSuitableIndexXC;
                    }
                    else
                    {
                        parkingSpotsRemainingAtCurrentIndexXC[0].clear();
                    }

                    targetSitesInCurrentTarget[0] = targetSites[currentTargetIndexXC[0]].size();
                }
            }

            if(highIndexSideExists && targetSites[currentTargetIndexXC[1]].empty())
            {
                totalRequiredAtoms -= targetSitesInCurrentTarget[1];
                targetSitesInCurrentTarget[1] = 0;

                if(currentTargetIndexXC[1] < (int)targetIndexXC[1] && currentTargetIndexXC[1] < (int)targetSites.size() - 1)
                {
                    atLeastOneWasMoved = true;
                    currentTargetIndexXC[1]++;
                    if((currentTargetIndexXC[1] - compZoneXCStart + 1) % targetGapXC == 0)
                    {
                        parkingSpotsRemainingAtCurrentIndexXC[1] = parkingSpotsPerSuitableIndexXC;
                    }
                    else
                    {
                        parkingSpotsRemainingAtCurrentIndexXC[1].clear();
                    }

                    targetSitesInCurrentTarget[1] = targetSites[currentTargetIndexXC[1]].size();
                }
            }

            if((!lowIndexSideExists || targetSites[currentTargetIndexXC[0]].empty()) &&
                (!highIndexSideExists || targetSites[currentTargetIndexXC[1]].empty()) && !atLeastOneWasMoved)
            {
                logger->debug("Breaking out of loop as there are no more targets and target can't be moved");
                break;
            }
            
            if(!atLeastOneWasMoved)
            {
                excludedIndicesForRemovalAndParkingMoves.clear();

                if(lowIndexContainsAtoms && highIndexContainsAtoms)
                {
                    if(!createCombinedMoves(stateArray, vertical, moveList, excludedIndicesForRemovalAndParkingMoves, 
                        usableAtoms[indexXC[0]], usableAtoms[indexXC[1]], 
                        &targetSites[currentTargetIndexXC[0]], currentTargetIndexXC[0], 
                        &targetSites[currentTargetIndexXC[1]], currentTargetIndexXC[1],
                        usableAtoms[indexXC[0]].size(), usableAtoms[indexXC[1]].size(), 
                        arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC, targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
                    {
                        return false;
                    }
                }
                else if(lowIndexContainsAtoms)
                {
                    if(!createSingleIndexMoves(stateArray, vertical, moveList, usableAtoms[indexXC[0]], -1, 
                        &targetSites[currentTargetIndexXC[0]], currentTargetIndexXC[0], 
                        usableAtoms[indexXC[0]].size(), arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC[0], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
                    {
                        return false;
                    }
                }
                else if(highIndexContainsAtoms)
                {
                    if(!createSingleIndexMoves(stateArray, vertical, moveList, usableAtoms[indexXC[1]], 1, 
                        &targetSites[currentTargetIndexXC[1]], currentTargetIndexXC[1], 
                        usableAtoms[indexXC[1]].size(), arraySizeXC, arraySizeAC, sortingChannelWidth, 
                        indexXC[1], targetGapAC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtoms, false, logger))
                    {
                        return false;
                    }
                }

                // We need to fill this set again as there may be fewer usable atoms
                for(unsigned int side : {0,1})
                {
                    if(indexXC[side] >= 0 && indexXC[side] < (int)arraySizeXC)
                    {
                        for(const auto& usableIndex : usableAtoms[indexXC[side]])
                        {
                            for(int excludedIndex = usableIndex - targetGapAC + 1; 
                                excludedIndex < usableIndex + targetGapAC; excludedIndex++)
                            {
                                excludedIndicesForRemovalAndParkingMoves.insert(excludedIndex);
                            }
                        }
                    }
                }
            }
        }
        indexXC[0]--;
        indexXC[1]++;
    }

    return totalRequiredAtoms == 0;
}

std::tuple<std::vector<std::vector<int>>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> 
    findUnusableAtoms(ArrayAccessor& stateArray, bool vertical, unsigned int arraySizeXC, unsigned int arraySizeAC,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    ArrayAccessor& targetGeometry)
{
    // Create mask where true existing atoms would prevent an atoms usability
    auto usabilityPreventingNeighborhoodMask = generateMask(Config::getInstance().minDistFromOccSites);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    std::vector<std::vector<int>> unusableAtoms, usableAtoms, targetSites;

    unusableAtoms.resize(arraySizeXC);
    usableAtoms.resize(arraySizeXC);
    targetSites.resize(arraySizeXC);

    // Iterate over array, check for usability-preventing neighbors, and sort into structure accordingly
    for(size_t indexXC = 0; indexXC < arraySizeXC; indexXC++)
    {
        for(size_t indexAC = 0; indexAC < arraySizeAC; indexAC++)
        {
            size_t row = vertical ? indexAC : indexXC;
            size_t col = vertical ? indexXC : indexAC;

            if(row >= compZoneRowStart && row < compZoneRowEnd && col >= compZoneColStart && 
                col < compZoneColEnd && targetGeometry(row - compZoneRowStart, col - compZoneColStart))
            {
                targetSites[indexXC].push_back(indexAC);
            }
            if(stateArray(row,col))
            {
                bool usable = true;
                for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                    usable && rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                {
                    int shiftedRow = (int)row + rowShift;
                    if(shiftedRow >= 0 && shiftedRow < (int)stateArray.rows())
                    {
                        for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                            colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                        {
                            int shiftedCol = (int)col + colShift;
                            if(shiftedCol >= 0 && shiftedCol < (int)stateArray.cols() && usabilityPreventingNeighborhoodMask(
                                rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                colShift + usabilityPreventingNeighborhoodMaskColDist) && 
                                stateArray(shiftedRow, shiftedCol))
                            {
                                usable = false;
                                break;
                            }
                        }
                    }
                }
                if(usable)
                {
                    usableAtoms[indexXC].push_back(indexAC);
                }
                else
                {
                    unusableAtoms[indexXC].push_back(indexAC);
                }
            }
        }
    }

    return std::tuple(std::move(usableAtoms), std::move(unusableAtoms), std::move(targetSites));
}

// Move atoms from parking spots to empty target sites
bool resolveSortingDeficiencies(ArrayAccessor& stateArray, int startIndex,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeAC, 
    unsigned int arraySizeXC, unsigned int maxTones, double spacingXC, std::vector<std::vector<int>>& usableAtoms, 
    std::vector<std::vector<int>>& unusableAtoms, std::vector<std::vector<int>>& targetSites, 
    size_t compZoneXCStart, size_t compZoneXCEnd, size_t compZoneACStart, size_t compZoneACEnd,
    int targetGapXC, int targetGapAC, std::shared_ptr<spdlog::logger> logger)
{    
    for(int i = 0; i < 2; i++)
    {
        if((i == 0 && startIndex < (int)arraySizeXC) || (i == 1 && startIndex > 0))
        {
            int lastIndexXCToPossiblyContainsAtoms, targetIndexXC, lastTargetIndexXCExclusive, 
                targetIndexXCDir, lastIndexXCWithUsableAtoms;
            if(i == 0)
            {
                lastIndexXCWithUsableAtoms = arraySizeXC - 1;
                lastIndexXCToPossiblyContainsAtoms = 0;
                targetIndexXC = startIndex;
                lastTargetIndexXCExclusive = arraySizeXC;
                targetIndexXCDir = 1;
            }
            else
            {
                lastIndexXCWithUsableAtoms = 0;
                lastIndexXCToPossiblyContainsAtoms = arraySizeXC - 1;
                targetIndexXC = startIndex - 1;
                lastTargetIndexXCExclusive = -1;
                targetIndexXCDir = -1;
            }
            for(; targetIndexXC != lastTargetIndexXCExclusive; targetIndexXC += targetIndexXCDir)
            {
                while(!targetSites[targetIndexXC].empty())
                {
                    while(usableAtoms[lastIndexXCWithUsableAtoms].empty())
                    {
                        if(lastIndexXCWithUsableAtoms == lastIndexXCToPossiblyContainsAtoms)
                        {
                            return false;
                        }
                        lastIndexXCWithUsableAtoms -= targetIndexXCDir;
                    }

                    logger->debug("indexXC: {}", lastIndexXCWithUsableAtoms);
                    std::stringstream unusableAtomsStr;
                    unusableAtomsStr << "usableAtoms[lastIndexXCWithUsableAtoms]: ";
                    for(auto usableAtom : usableAtoms[lastIndexXCWithUsableAtoms])
                    {
                        unusableAtomsStr << usableAtom << ", ";
                    }
                    logger->debug(unusableAtomsStr.str());

                    double channelIndexXC = (targetIndexXC * targetIndexXCDir > 
                        lastIndexXCWithUsableAtoms * targetIndexXCDir) ? 
                        targetIndexXC : lastIndexXCWithUsableAtoms;
                    channelIndexXC += (double)(sortingChannelWidth + 1) / 2. * (double)targetIndexXCDir;

                    ParallelMove move;
                    ParallelMove::Step start, elbow1, elbow2, end;
                    std::vector<double> *startSelectionAC, *endSelectionAC;
                    if(vertical)
                    {
                        start.colSelection.push_back(lastIndexXCWithUsableAtoms);
                        elbow1.colSelection.push_back(channelIndexXC);
                        elbow2.colSelection.push_back(channelIndexXC);
                        end.colSelection.push_back(targetIndexXC);
                        startSelectionAC = &start.rowSelection;
                        endSelectionAC = &end.rowSelection;
                    }
                    else
                    {
                        start.rowSelection.push_back(lastIndexXCWithUsableAtoms);
                        elbow1.rowSelection.push_back(channelIndexXC);
                        elbow2.rowSelection.push_back(channelIndexXC);
                        end.rowSelection.push_back(targetIndexXC);
                        startSelectionAC = &start.colSelection;
                        endSelectionAC = &end.colSelection;
                    }
                    unsigned int count = targetSites[targetIndexXC].size() < 
                        usableAtoms[lastIndexXCWithUsableAtoms].size() ? 
                        targetSites[targetIndexXC].size() : usableAtoms[lastIndexXCWithUsableAtoms].size();
                    if(count > maxTones)
                    {
                        count = maxTones;
                    }
                    for(size_t i = 0; i < count; i++)
                    {
                        startSelectionAC->push_back(usableAtoms[lastIndexXCWithUsableAtoms].back());
                        endSelectionAC->push_back(targetSites[targetIndexXC].back());
                        usableAtoms[lastIndexXCWithUsableAtoms].pop_back();
                        targetSites[targetIndexXC].pop_back();
                    }
                    std::sort(startSelectionAC->begin(), startSelectionAC->end());
                    std::sort(endSelectionAC->begin(), endSelectionAC->end());
                    if(vertical)
                    {
                        elbow1.rowSelection = start.rowSelection;
                        elbow2.rowSelection = end.rowSelection;
                    }
                    else
                    {
                        elbow1.colSelection = start.colSelection;
                        elbow2.colSelection = end.colSelection;
                    }
                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(elbow1));
                    move.steps.push_back(std::move(elbow2));
                    move.steps.push_back(std::move(end));
                    move.execute(stateArray, logger);
                    moveList.push_back(std::move(move));
                }
            }
        }
    }

    return true;
}

bool sortArray(ArrayAccessor& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    ArrayAccessor& targetGeometry, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    bool channelVertical = Config::getInstance().columnSpacing > Config::getInstance().rowSpacing;
    unsigned int maxTonesXC = Config::getInstance().aodTotalLimit,
        maxTonesAC = Config::getInstance().aodTotalLimit;

    // Across channel dir will be abbreviated as XC, along channel as AC
    unsigned int arraySizeXC, arraySizeAC;
    double spacingXC, spacingAC;
    size_t compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd;
    // Initialize data independent of whether sorting channel is vertical or horizontal
    if(channelVertical)
    {
        if(Config::getInstance().aodColLimit < maxTonesXC)
        {
            maxTonesXC = Config::getInstance().aodColLimit;
        }
        if(Config::getInstance().aodRowLimit < maxTonesAC)
        {
            maxTonesAC = Config::getInstance().aodRowLimit;
        }
        spacingXC = Config::getInstance().columnSpacing;
        spacingAC = Config::getInstance().rowSpacing;
        arraySizeXC = stateArray.cols();
        arraySizeAC = stateArray.rows();
        compZoneXCStart = compZoneColStart;
        compZoneXCEnd = compZoneColEnd;
        compZoneACStart = compZoneRowStart;
        compZoneACEnd = compZoneRowEnd;
    }
    else
    {
        if(Config::getInstance().aodRowLimit < maxTonesXC)
        {
            maxTonesXC = Config::getInstance().aodRowLimit;
        }
        if(Config::getInstance().aodColLimit < maxTonesAC)
        {
            maxTonesAC = Config::getInstance().aodColLimit;
        }
        spacingXC = Config::getInstance().rowSpacing;
        spacingAC = Config::getInstance().columnSpacing;
        arraySizeXC = stateArray.rows();
        arraySizeAC = stateArray.cols();
        compZoneXCStart = compZoneRowStart;
        compZoneXCEnd = compZoneRowEnd;
        compZoneACStart = compZoneColStart;
        compZoneACEnd = compZoneColEnd;
    }

    int sortingChannelWidth = (int)(ceil(Config::getInstance().minDistFromOccSites / (spacingXC / 2.)) / 2) * 2;
    int targetGapXC = ceil(Config::getInstance().minDistFromOccSites / spacingXC);
    int targetGapAC = ceil(Config::getInstance().minDistFromOccSites / spacingAC);

    // Differentiate between unusable (too close to each other) and usable atoms and add into per-index buffers
    auto [usableAtomsPerXCIndex, unusableAtomsPerXCIndex, targetSitesPerXCIndex] = 
        findUnusableAtoms(stateArray, channelVertical, arraySizeXC, arraySizeAC, compZoneRowStart, 
            compZoneRowEnd, compZoneColStart, compZoneColEnd, targetGeometry);

    // Determine were to start sorting. May either be from one side and iterate one way over the array or 
    // from the middle and outward simultaneously
    auto startingPosition = determineBestStartPosition(stateArray, channelVertical, 
        sortingChannelWidth, moveList, arraySizeXC, compZoneXCStart, compZoneXCEnd, 
        unusableAtomsPerXCIndex, usableAtomsPerXCIndex, targetSitesPerXCIndex);
    if(startingPosition.has_value())
    {
        logger->debug("Best starting position: {}", startingPosition.value());
    }
    else
    {
        logger->error("Best starting position could not be determined!");
        return false;
    }

    // Remove atoms from starting sorting channel
    clearSortingChannel(stateArray, startingPosition.value(), channelVertical, sortingChannelWidth, 
        moveList, arraySizeXC, arraySizeAC, maxTonesAC, spacingXC, targetGapXC, targetGapAC, 
        usableAtomsPerXCIndex, unusableAtomsPerXCIndex, logger);

    // Call main iterating function that sorts atoms row-by-row through sorting channel
    if(!sortRemainingRowsOrCols(stateArray, startingPosition.value(), channelVertical, sortingChannelWidth, moveList, 
        arraySizeAC, arraySizeXC, maxTonesXC, maxTonesAC, spacingXC, spacingAC, usableAtomsPerXCIndex, unusableAtomsPerXCIndex, 
        targetSitesPerXCIndex, compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd, 
        targetGapXC, targetGapAC, logger))
    {
        // Fill remaining positions from parked atoms
        if(!resolveSortingDeficiencies(stateArray, startingPosition.value(), channelVertical, 
            sortingChannelWidth, moveList, arraySizeAC, arraySizeXC, maxTonesAC, spacingXC, 
            usableAtomsPerXCIndex, unusableAtomsPerXCIndex, targetSitesPerXCIndex, 
            compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd, targetGapXC, targetGapAC, logger))
        {
            logger->error("Array could not be sorted. Aborting");
            return false;
        }
        return true;
    }

    return true;
}

// Access function to be bound, Eigen array act as interfaces as they can act on Python array data without reallocation
std::optional<std::vector<ParallelMove>> sortLatticeByRowParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    // Init logger
    std::shared_ptr<spdlog::logger> logger;
    Config& config = Config::getInstance();
    if((logger = spdlog::get(config.latticeByRowLoggerName)) == nullptr)
    {
        logger = spdlog::basic_logger_mt(config.latticeByRowLoggerName, config.logFileName);
    }
    logger->set_level(spdlog::level::debug);

    omp_set_num_threads(NUM_THREADS);

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

    // Log initial state
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

    // Actual sorting call
    EigenArrayAccessor eigenStateArray(stateArray);
    EigenArrayAccessor eigenTargetArray(targetGeometry);
    std::vector<ParallelMove> moveList;
    if(!sortArray(eigenStateArray, compZoneRowStart, compZoneRowEnd, 
        compZoneColStart, compZoneColEnd, eigenTargetArray, moveList, logger))
    {
        return std::nullopt;
    }

    // Log final state if sorting was successful
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