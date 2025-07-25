/***
 * Sorting approach that imposes a fixed movement scheme instead of finding the best move in a greedy fashion
 * Idea by Francisco Romão and Jonas Winklmann, Implemented by Jonas Winklmann
 */

#include "sortParallel.hpp"
#include "sortLatticeGeometries.hpp"

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

bool& accessStateArrayDimIndepedent(StateArrayAccessor& stateArray, 
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

void clearFirstNRowsOrCols(StateArrayAccessor& stateArray,
    bool vertical, unsigned int count, std::vector<ParallelMove>& moveList, unsigned int arraySizeAC,
    unsigned int maxTones, double spacingXC, int targetGapXC, std::shared_ptr<spdlog::logger> logger)
{
    int nextIndexToDealWith = 0;
    if(((vertical && AOD_ROW_LIMIT >= count) || (!vertical && AOD_COL_LIMIT >= count)) && 
        count * maxTones <= AOD_TOTAL_LIMIT && spacingXC >= MIN_DIST_FROM_OCC_SITES)
    {
        while(nextIndexToDealWith < (int)arraySizeAC)
        {
            ParallelMove move;
            ParallelMove::Step start;
            ParallelMove::Step end;
            std::vector<double> *startSelectionAC, *endSelectionAC;
            if(vertical)
            {
                for(int i = 0; i < (int)count; i++)
                {
                    start.colSelection.push_back(i);
                    end.colSelection.push_back(-(int)count - targetGapXC + 1 + i);
                }
                startSelectionAC = &start.rowSelection;
                endSelectionAC = &end.rowSelection;
            }
            else
            {
                for(int i = 0; i < (int)count; i++)
                {
                    start.rowSelection.push_back(i);
                    end.rowSelection.push_back(-(int)count - targetGapXC + 1 + i);
                }
                startSelectionAC = &start.colSelection;
                endSelectionAC = &end.colSelection;
            }
            for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
            {
                bool indexRequired = false;
                for(int i = 0; i < (int)count; i++)
                {
                    if(accessStateArrayDimIndepedent(stateArray, i, nextIndexToDealWith, vertical))
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
        for(int i = 0; i < (int)count; i++)
        {
            nextIndexToDealWith = 0;
            while(nextIndexToDealWith < (int)arraySizeAC)
            {
                ParallelMove move;
                ParallelMove::Step start;
                ParallelMove::Step end;
                std::vector<double> *startSelectionAC, *endSelectionAC;
                if(vertical)
                {
                    start.colSelection.push_back(i);
                    end.colSelection.push_back(-1);
                    startSelectionAC = &start.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(i);
                    end.rowSelection.push_back(-1);
                    startSelectionAC = &start.colSelection;
                    endSelectionAC = &end.colSelection;
                }
                for(; nextIndexToDealWith < (int)arraySizeAC && startSelectionAC->size() < maxTones; nextIndexToDealWith++)
                {
                    if(accessStateArrayDimIndepedent(stateArray, i, nextIndexToDealWith, vertical))
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

bool sortRemainingRowsOrCols(StateArrayAccessor& stateArray,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeAC, 
    unsigned int arraySizeXC, unsigned int maxTones, double spacingXC, std::vector<std::vector<int>>& usableAtoms, 
    std::vector<std::vector<int>>& unusableAtoms, std::vector<std::vector<int>>& targetSites, 
    size_t compZoneXCStart, size_t compZoneXCEnd, size_t compZoneACStart, size_t compZoneACEnd,
    int targetGapXC, int targetGapAC, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<size_t> parkingSpotsPerSuitableIndexXC;
    for(int i = (compZoneACStart - targetGapAC + 1) / targetGapAC; i >= 0; i--)
    {
        parkingSpotsPerSuitableIndexXC.push_back(i * targetGapAC);
    }
    for(size_t i = arraySizeAC; i >= compZoneACEnd + targetGapAC - 1; i -= targetGapAC)
    {
        parkingSpotsPerSuitableIndexXC.push_back(i);
    }
    size_t currentTargetIndexXC = compZoneXCStart;
    unsigned int requiredAtoms = 0;
    std::vector<size_t> parkingSpotsRemainingAtCurrentIndexXC;
    unsigned int totalRequiredAtoms = std::accumulate(targetSites.begin(), targetSites.end(), 0u, [](unsigned int init, const auto& elem) { return init + elem.size(); });
    for(size_t indexXC = sortingChannelWidth + 1; indexXC < arraySizeXC; indexXC++)
    {
        if(indexXC >= compZoneXCEnd && totalRequiredAtoms == 0)
        {
            return true;
        }
        size_t targetIndexXC = indexXC - sortingChannelWidth - 1;

        requiredAtoms += targetSites[targetIndexXC].size();
        unsigned int parkingSpots = parkingSpotsRemainingAtCurrentIndexXC.size();
        for(size_t parkingSpotXC = currentTargetIndexXC + 1; parkingSpotXC <= targetIndexXC; parkingSpotXC++)
        {
            if((parkingSpotXC - compZoneXCStart + 1) % targetGapXC == 0)
            {
                parkingSpots += parkingSpotsPerSuitableIndexXC.size();
            }
        }

        // Excess atoms that cannot be used for filling target sites or parking spots are thrown away
        if(targetIndexXC >= compZoneXCStart)
        {
            while(usableAtoms[indexXC].size() > requiredAtoms + parkingSpots)
            {
                if(usableAtoms[indexXC][0] > (int)arraySizeAC - usableAtoms[indexXC].back() - 1)
                {
                    unusableAtoms[indexXC].insert(std::upper_bound(unusableAtoms[indexXC].begin(), 
                        unusableAtoms[indexXC].end(), usableAtoms[indexXC][0]), usableAtoms[indexXC][0]);
                    usableAtoms[indexXC].erase(usableAtoms[indexXC].begin());
                }
                else
                {
                    unusableAtoms[indexXC].insert(std::upper_bound(unusableAtoms[indexXC].begin(), 
                        unusableAtoms[indexXC].end(), usableAtoms[indexXC].back()), usableAtoms[indexXC].back());
                    usableAtoms[indexXC].pop_back();
                }
            }
        }
        auto indexAC = unusableAtoms[indexXC].begin();
        while(indexAC != unusableAtoms[indexXC].end())
        {
            ParallelMove move;
            ParallelMove::Step start;
            ParallelMove::Step elbow;
            ParallelMove::Step end;
            std::vector<double> *startSelectionAC, *elbowSelectionAC, *endSelectionAC;
            if(vertical)
            {
                start.colSelection.push_back(indexXC);
                elbow.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                end.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                startSelectionAC = &start.rowSelection;
                elbowSelectionAC = &elbow.rowSelection;
                endSelectionAC = &end.rowSelection;
            }
            else
            {
                start.rowSelection.push_back(indexXC);
                elbow.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                end.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                startSelectionAC = &start.colSelection;
                elbowSelectionAC = &elbow.colSelection;
                endSelectionAC = &end.colSelection;
            }
            int indicesLowerOrEqualMiddle = 0, indicesHigherMiddle = 0;
            for(; indexAC != unusableAtoms[indexXC].end() && startSelectionAC->size() < maxTones; indexAC++)
            {
                startSelectionAC->push_back(*indexAC);
                elbowSelectionAC->push_back(*indexAC);
                if(*indexAC <= (int)arraySizeAC / 2)
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
            move.steps.push_back(std::move(start));
            move.steps.push_back(std::move(elbow));
            move.steps.push_back(std::move(end));
            move.execute(stateArray, logger);
            moveList.push_back(std::move(move));
        }
        if(targetIndexXC < compZoneXCStart)
        {
            indexAC = usableAtoms[indexXC].begin();
            while(indexAC != usableAtoms[indexXC].end())
            {
                ParallelMove move;
                ParallelMove::Step start;
                ParallelMove::Step end;
                std::vector<double> *startSelectionAC, *endSelectionAC;
                if(vertical)
                {
                    start.colSelection.push_back(indexXC);
                    end.colSelection.push_back(targetIndexXC);
                    startSelectionAC = &start.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(indexXC);
                    end.rowSelection.push_back(targetIndexXC);
                    startSelectionAC = &start.colSelection;
                    endSelectionAC = &end.colSelection;
                }
                for(; indexAC != usableAtoms[indexXC].end() && startSelectionAC->size() < maxTones; indexAC++)
                {
                    startSelectionAC->push_back(*indexAC);
                    endSelectionAC->push_back(*indexAC);
                }
                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(end));
                move.execute(stateArray, logger);
                moveList.push_back(std::move(move));
            }
            usableAtoms[targetIndexXC] = usableAtoms[indexXC];
            usableAtoms[indexXC].clear();
        }
        else
        {
            while(!usableAtoms[indexXC].empty())
            {
                while(usableAtoms[indexXC].size() > requiredAtoms && !parkingSpotsRemainingAtCurrentIndexXC.empty())
                {
                    ParallelMove move;
                    ParallelMove::Step start, elbow1, elbow2, end;
                    std::vector<double> *startSelectionAC, *endSelectionAC;
                    if(vertical)
                    {
                        start.colSelection.push_back(indexXC);
                        elbow1.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        elbow2.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        end.colSelection.push_back(currentTargetIndexXC);
                        startSelectionAC = &start.rowSelection;
                        endSelectionAC = &end.rowSelection;
                    }
                    else
                    {
                        start.rowSelection.push_back(indexXC);
                        elbow1.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        elbow2.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        end.rowSelection.push_back(currentTargetIndexXC);
                        startSelectionAC = &start.colSelection;
                        endSelectionAC = &end.colSelection;
                    }

                    while(usableAtoms[indexXC].size() > requiredAtoms && !parkingSpotsRemainingAtCurrentIndexXC.empty() && 
                        startSelectionAC->size() < maxTones)
                    {
                        if(abs(usableAtoms[indexXC][0] - (int)(parkingSpotsRemainingAtCurrentIndexXC[0])) <
                            abs(usableAtoms[indexXC].back() - (int)(parkingSpotsRemainingAtCurrentIndexXC.back())))
                        {
                            startSelectionAC->push_back(usableAtoms[indexXC][0]);
                            endSelectionAC->push_back(parkingSpotsRemainingAtCurrentIndexXC[0]);
                            usableAtoms[indexXC].erase(usableAtoms[indexXC].begin());
                            usableAtoms[currentTargetIndexXC].push_back(parkingSpotsRemainingAtCurrentIndexXC[0]);
                            parkingSpotsRemainingAtCurrentIndexXC.erase(parkingSpotsRemainingAtCurrentIndexXC.begin());
                        }
                        else
                        {
                            startSelectionAC->push_back(usableAtoms[indexXC].back());
                            endSelectionAC->push_back(parkingSpotsRemainingAtCurrentIndexXC.back());
                            usableAtoms[indexXC].pop_back();
                            usableAtoms[currentTargetIndexXC].push_back(parkingSpotsRemainingAtCurrentIndexXC.back());
                            parkingSpotsRemainingAtCurrentIndexXC.pop_back();
                        }
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
                if(targetSites[currentTargetIndexXC].empty())
                {
                    if(currentTargetIndexXC < targetIndexXC)
                    {
                        currentTargetIndexXC++;
                        if((currentTargetIndexXC - compZoneXCStart + 1) % targetGapXC == 0)
                        {
                            parkingSpotsRemainingAtCurrentIndexXC = parkingSpotsPerSuitableIndexXC;
                        }
                        else
                        {
                            parkingSpotsRemainingAtCurrentIndexXC.clear();
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    std::vector<int>::iterator middleTargetAC, middleSourceAC;

                    std::vector<int> usedSourceIndices;
                    std::vector<int> usedTargetIndices;

                    if(targetSites[currentTargetIndexXC].size() > usableAtoms[indexXC].size())
                    {
                        middleSourceAC = usableAtoms[indexXC].begin(); 
                        size_t halfUsableAtomCount = usableAtoms[indexXC].size() / 2;
                        std::advance(middleSourceAC, halfUsableAtomCount);
                        std::vector<int>::iterator targetAC = targetSites[currentTargetIndexXC].begin();
                        std::advance(targetAC, halfUsableAtomCount);
                        int minDist = INT_MAX;

                        for(; std::distance(targetAC, targetSites[currentTargetIndexXC].end()) >= 
                            std::distance(middleSourceAC, usableAtoms[indexXC].end()); targetAC++)
                        {
                            int distSourceToTarget = *targetAC - *middleSourceAC;
                            if(distSourceToTarget < 0)
                            {
                                middleTargetAC = targetAC;
                                minDist = -distSourceToTarget;
                            }
                            else
                            {
                                if(distSourceToTarget < minDist)
                                {
                                    middleTargetAC = targetAC;
                                }
                                break;
                            }
                        }

                        usedSourceIndices.push_back(*middleSourceAC);
                        usedTargetIndices.push_back(*middleTargetAC);

                        auto targetACLeftwards = std::make_reverse_iterator(middleTargetAC);
                        auto sourceACLeftwards = std::make_reverse_iterator(middleSourceAC);

                        auto middleTargetACIndex = std::distance(targetSites[currentTargetIndexXC].begin(), middleTargetAC);
                        usableAtoms[indexXC].erase(middleSourceAC);
                        targetSites[currentTargetIndexXC].erase(middleTargetAC);

                        for(; sourceACLeftwards != usableAtoms[indexXC].rend() && 
                            targetACLeftwards != targetSites[currentTargetIndexXC].rend() && 
                            usedSourceIndices.size() < maxTones;)
                        {
                            int minDist = INT_MAX;

                            for(; std::distance(targetACLeftwards, targetSites[currentTargetIndexXC].rend()) > 
                                std::distance(sourceACLeftwards, usableAtoms[indexXC].rend()); targetACLeftwards++)
                            {
                                int distTargetToSource = *sourceACLeftwards - *targetACLeftwards;
                                if(distTargetToSource < 0)
                                {
                                    minDist = -distTargetToSource;
                                }
                                else
                                {
                                    if(distTargetToSource >= minDist)
                                    {
                                        targetACLeftwards--;
                                    }
                                    break;
                                }
                            }
                            usedSourceIndices.push_back(*sourceACLeftwards);
                            usedTargetIndices.push_back(*targetACLeftwards);
                            usableAtoms[indexXC].erase((++sourceACLeftwards).base());
                            targetSites[currentTargetIndexXC].erase((++targetACLeftwards).base());
                        }
                        
                        middleSourceAC = usableAtoms[indexXC].begin(); 
                        std::advance(middleSourceAC, halfUsableAtomCount + 1 - usedSourceIndices.size());
                        middleTargetAC = targetSites[currentTargetIndexXC].begin();
                        std::advance(middleTargetAC, middleTargetACIndex + 1 - usedTargetIndices.size());

                        for(; middleSourceAC != usableAtoms[indexXC].end() && 
                            middleTargetAC != targetSites[currentTargetIndexXC].end() && 
                            usedSourceIndices.size() < maxTones;)
                        {
                            int minDist = INT_MAX;

                            for(; std::distance(middleTargetAC, targetSites[currentTargetIndexXC].end()) >
                                std::distance(middleSourceAC, usableAtoms[indexXC].end()); middleTargetAC++)
                            {
                                int distSourceToTarget = *middleTargetAC - *middleSourceAC;
                                if(distSourceToTarget < 0)
                                {
                                    minDist = -distSourceToTarget;
                                }
                                else
                                {
                                    if(distSourceToTarget >= minDist)
                                    {
                                        middleTargetAC--;
                                    }
                                    break;
                                }
                            }
                            usedSourceIndices.push_back(*middleSourceAC);
                            usedTargetIndices.push_back(*middleTargetAC);
                            middleSourceAC = usableAtoms[indexXC].erase(middleSourceAC);
                            middleTargetAC = targetSites[currentTargetIndexXC].erase(middleTargetAC);
                        }
                    }
                    else
                    {
                        middleTargetAC = targetSites[currentTargetIndexXC].begin(); 
                        size_t halfTargetSiteCount = targetSites[currentTargetIndexXC].size() / 2;
                        std::advance(middleTargetAC, halfTargetSiteCount);
                        std::vector<int>::iterator sourceAC = usableAtoms[indexXC].begin();
                        std::advance(sourceAC, halfTargetSiteCount);
                        int minDist = INT_MAX;

                        for(; std::distance(sourceAC, usableAtoms[indexXC].end()) >=
                            std::distance(middleTargetAC, targetSites[currentTargetIndexXC].end()); sourceAC++)
                        {
                            int distTargetToSource = *sourceAC - *middleTargetAC;
                            if(distTargetToSource < 0)
                            {
                                middleSourceAC = sourceAC;
                                minDist = -distTargetToSource;
                            }
                            else
                            {
                                if(distTargetToSource < minDist)
                                {
                                    middleSourceAC = sourceAC;
                                }
                                break;
                            }
                        }

                        usedSourceIndices.push_back(*middleSourceAC);
                        usedTargetIndices.push_back(*middleTargetAC);

                        auto targetACLeftwards = std::make_reverse_iterator(middleTargetAC);
                        auto sourceACLeftwards = std::make_reverse_iterator(middleSourceAC);

                        auto middleSourceACIndex = std::distance(usableAtoms[indexXC].begin(), middleSourceAC);
                        usableAtoms[indexXC].erase(middleSourceAC);
                        targetSites[currentTargetIndexXC].erase(middleTargetAC);

                        for(; sourceACLeftwards != usableAtoms[indexXC].rend() && 
                            targetACLeftwards != targetSites[currentTargetIndexXC].rend() && 
                            usedSourceIndices.size() < maxTones;)
                        {
                            int minDist = INT_MAX;

                            if(currentTargetIndexXC < targetIndexXC)
                            {
                                for(; std::distance(sourceACLeftwards, usableAtoms[indexXC].rend()) > 
                                    std::distance(targetACLeftwards, targetSites[currentTargetIndexXC].rend()); sourceACLeftwards++)
                                {
                                    int distSourceToTarget = *targetACLeftwards - *sourceACLeftwards;
                                    if(distSourceToTarget < 0)
                                    {
                                        minDist = -distSourceToTarget;
                                    }
                                    else
                                    {
                                        if(distSourceToTarget >= minDist)
                                        {
                                            sourceACLeftwards--;
                                        }
                                        break;
                                    }
                                }
                            }
                            usedSourceIndices.push_back(*sourceACLeftwards);
                            usedTargetIndices.push_back(*targetACLeftwards);
                            usableAtoms[indexXC].erase((++sourceACLeftwards).base());
                            targetSites[currentTargetIndexXC].erase((++targetACLeftwards).base());
                        }
                        
                        middleSourceAC = usableAtoms[indexXC].begin(); 
                        std::advance(middleSourceAC, middleSourceACIndex + 1 - usedSourceIndices.size());
                        middleTargetAC = targetSites[currentTargetIndexXC].begin();
                        std::advance(middleTargetAC, halfTargetSiteCount + 1 - usedTargetIndices.size());

                        for(; middleSourceAC != usableAtoms[indexXC].end() && middleTargetAC != targetSites[currentTargetIndexXC].end() && 
                            usedSourceIndices.size() < maxTones;)
                        {
                            int minDist = INT_MAX;

                            if(currentTargetIndexXC < targetIndexXC)
                            {
                                for(; std::distance(middleSourceAC, usableAtoms[indexXC].end()) > 
                                    std::distance(middleTargetAC, targetSites[currentTargetIndexXC].end()); middleSourceAC++)
                                {
                                    int distTargetToSource = *middleSourceAC - *middleTargetAC;
                                    if(distTargetToSource < 0)
                                    {
                                        minDist = -distTargetToSource;
                                    }
                                    else
                                    {
                                        if(distTargetToSource >= minDist)
                                        {
                                            middleSourceAC--;
                                        }
                                        break;
                                    }
                                }
                            }
                            usedSourceIndices.push_back(*middleSourceAC);
                            usedTargetIndices.push_back(*middleTargetAC);
                            middleSourceAC = usableAtoms[indexXC].erase(middleSourceAC);
                            middleTargetAC = targetSites[currentTargetIndexXC].erase(middleTargetAC);
                        }
                    }

                    std::sort(usedSourceIndices.begin(), usedSourceIndices.end());
                    std::sort(usedTargetIndices.begin(), usedTargetIndices.end());

                    requiredAtoms -= usedSourceIndices.size();
                    totalRequiredAtoms -= usedSourceIndices.size();

                    ParallelMove move;
                    ParallelMove::Step start, elbow1, elbow2, end;
                    std::vector<double> *startSelectionAC, *endSelectionAC;
                    if(vertical)
                    {
                        start.colSelection.push_back(indexXC);
                        elbow1.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        elbow2.colSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        end.colSelection.push_back(currentTargetIndexXC);
                        startSelectionAC = &start.rowSelection;
                        endSelectionAC = &end.rowSelection;
                    }
                    else
                    {
                        start.rowSelection.push_back(indexXC);
                        elbow1.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        elbow2.rowSelection.push_back((double)indexXC - (double)(sortingChannelWidth + 1) / 2.);
                        end.rowSelection.push_back(currentTargetIndexXC);
                        startSelectionAC = &start.colSelection;
                        endSelectionAC = &end.colSelection;
                    }
                    for(int sourceIndexAC : usedSourceIndices)
                    {
                        startSelectionAC->push_back(sourceIndexAC);
                    }
                    for(int targetIndexAC : usedTargetIndices)
                    {
                        endSelectionAC->push_back(targetIndexAC);
                    }
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

    return totalRequiredAtoms == 0;
}

std::tuple<std::vector<std::vector<int>>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> 
    findUnusableAtoms(StateArrayAccessor& stateArray, bool vertical, unsigned int arraySizeXC,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    auto usabilityPreventingNeighborhoodMask = generateMask(MIN_DIST_FROM_OCC_SITES);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    std::vector<std::vector<int>> unusableAtoms, usableAtoms, targetSites;

    unusableAtoms.resize(arraySizeXC);
    usableAtoms.resize(arraySizeXC);
    targetSites.resize(arraySizeXC);

    for(size_t row = 0; row < stateArray.rows(); row++)
    {
        for(size_t col = 0; col < stateArray.cols(); col++)
        {
            if(row >= compZoneRowStart && row < compZoneRowEnd && col >= compZoneColStart && 
                col < compZoneColEnd && targetGeometry(row - compZoneRowStart, col - compZoneColStart))
            {
                if(vertical)
                {
                    targetSites[col].push_back(row);
                }
                else
                {
                    targetSites[row].push_back(col);
                }
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
                    if(vertical)
                    {
                        usableAtoms[col].push_back(row);
                    }
                    else
                    {
                        usableAtoms[row].push_back(col);
                    }
                }
                else
                {
                    if(vertical)
                    {
                        unusableAtoms[col].push_back(row);
                    }
                    else
                    {
                        unusableAtoms[row].push_back(col);
                    }
                }
            }
        }
    }

    return std::tuple(std::move(usableAtoms), std::move(unusableAtoms), std::move(targetSites));
}

bool resolveSortingDeficiencies(StateArrayAccessor& stateArray,
    bool vertical, unsigned int sortingChannelWidth, std::vector<ParallelMove>& moveList, unsigned int arraySizeAC, 
    unsigned int arraySizeXC, unsigned int maxTones, double spacingXC, std::vector<std::vector<int>>& usableAtoms, 
    std::vector<std::vector<int>>& unusableAtoms, std::vector<std::vector<int>>& targetSites, 
    size_t compZoneXCStart, size_t compZoneXCEnd, size_t compZoneACStart, size_t compZoneACEnd,
    int targetGapXC, int targetGapAC, std::shared_ptr<spdlog::logger> logger)
{
    size_t lastIndexXCWithUsableAtoms = arraySizeXC - 1;
    for(size_t targetIndexXC = 0; targetIndexXC < arraySizeXC; targetIndexXC++)
    {
        while(!targetSites[targetIndexXC].empty())
        {
            while(usableAtoms[lastIndexXCWithUsableAtoms].empty())
            {
                if(lastIndexXCWithUsableAtoms == 0)
                {
                    return false;
                }
                lastIndexXCWithUsableAtoms--;
            }
            double channelIndexXC = targetIndexXC > lastIndexXCWithUsableAtoms ? targetIndexXC : lastIndexXCWithUsableAtoms;
            channelIndexXC += (double)(sortingChannelWidth + 1) / 2.;

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
            unsigned int count = targetSites[targetIndexXC].size() < usableAtoms[lastIndexXCWithUsableAtoms].size() ? 
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

    return true;
}

bool sortArray(StateArrayAccessor& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    bool channelVertical = COL_SPACING > ROW_SPACING;
    unsigned int maxTones = AOD_TOTAL_LIMIT;

    // Across channel dir will be abbreviated as XC, along channel as AC
    unsigned int arraySizeXC, arraySizeAC;
    double spacingXC, spacingAC;
    size_t compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd;
    if(channelVertical)
    {
        if(AOD_ROW_LIMIT < maxTones)
        {
            maxTones = AOD_ROW_LIMIT;
        }
        spacingXC = COL_SPACING;
        spacingAC = ROW_SPACING;
        arraySizeXC = stateArray.cols();
        arraySizeAC = stateArray.rows();
        compZoneXCStart = compZoneColStart;
        compZoneXCEnd = compZoneColEnd;
        compZoneACStart = compZoneRowStart;
        compZoneACEnd = compZoneRowEnd;
    }
    else
    {
        if(AOD_COL_LIMIT < maxTones)
        {
            maxTones = AOD_COL_LIMIT;
        }
        spacingXC = ROW_SPACING;
        spacingAC = COL_SPACING;
        arraySizeXC = stateArray.rows();
        arraySizeAC = stateArray.cols();
        compZoneXCStart = compZoneRowStart;
        compZoneXCEnd = compZoneRowEnd;
        compZoneACStart = compZoneColStart;
        compZoneACEnd = compZoneColEnd;
    }

    int sortingChannelWidth = (int)(ceil((double)MIN_DIST_FROM_OCC_SITES / ((double)spacingXC / 2.)) / 2) + 1;
    int targetGapXC = ceil((double)MIN_DIST_FROM_OCC_SITES / spacingXC);
    int targetGapAC = ceil((double)MIN_DIST_FROM_OCC_SITES / spacingAC);

    auto [usableAtomsPerXCIndex, unusableAtomsPerXCIndex, targetSitesPerXCIndex] = 
        findUnusableAtoms(stateArray, channelVertical, arraySizeXC, compZoneRowStart, 
            compZoneRowEnd, compZoneColStart, compZoneColEnd, targetGeometry);

    clearFirstNRowsOrCols(stateArray, channelVertical, sortingChannelWidth + 1, 
        moveList, arraySizeAC, maxTones, spacingXC, targetGapXC, logger);

    if(!sortRemainingRowsOrCols(stateArray, channelVertical, sortingChannelWidth, moveList, 
        arraySizeAC, arraySizeXC, maxTones, spacingXC, usableAtomsPerXCIndex, unusableAtomsPerXCIndex, 
        targetSitesPerXCIndex, compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd, 
        targetGapXC, targetGapAC, logger))
    {
        if(!resolveSortingDeficiencies(stateArray, channelVertical, sortingChannelWidth, moveList, 
            arraySizeAC, arraySizeXC, maxTones, spacingXC, usableAtomsPerXCIndex, unusableAtomsPerXCIndex, 
            targetSitesPerXCIndex, compZoneXCStart, compZoneXCEnd, compZoneACStart, compZoneACEnd, 
            targetGapXC, targetGapAC, logger))
        {
            logger->error("Array could not be sorted. Aborting");
            return false;
        }
    }

    return true;
}

std::optional<std::vector<ParallelMove>> sortLatticeByRowParallel(
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

    omp_set_num_threads(NUM_THREADS);

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

    EigenArrayStateArrayAccessor eigenStateArray(stateArray);
    std::vector<ParallelMove> moveList;
    if(!sortArray(eigenStateArray, compZoneRowStart, compZoneRowEnd, 
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