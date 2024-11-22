#include "sortSequentiallyByRow.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <new>
#include <optional>

#include "config.hpp"
#include "spdlog/sinks/basic_file_sink.h"

int priorityCompare(const void* a, const void* b)
{
    unsigned int int_a = * ( (unsigned int*) a );
    unsigned int int_b = * ( (unsigned int*) b );

    if ( int_a == int_b ) return 0;
    else if ( int_a > int_b ) return -1;
    else return 1;
}

void findPullPriorities(StateArrayAccessor& stateArray, size_t compZone[4], size_t border[4], 
    unsigned int *pullPriority, unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, unsigned int *atomsInColumnRemaining, 
    size_t targetRow, size_t rowsRemaining, int yDir)
{
    for(size_t col = compZone[2]; col <= compZone[3]; col++)
    {
        const auto& nextFilledRow = nextRowWithAtom[targetRow + (yDir > 0)][col];
        unsigned int atomsInNextFilledRow = 0;
        if(!nextFilledRow.has_value())
        {
            pullPriority[col - compZone[2]] = 0;
        }
        else
        {
            atomsInNextFilledRow = atomsPerRow[nextFilledRow.value()];
            unsigned int distance = (nextFilledRow.value() - targetRow) * yDir;
            unsigned int atomsInCurrentColumn = atomsInColumnRemaining[col];
            // Priority:
            // 1. Column with too many atoms
            // 2. Row with too many atoms
            // 3. Location in this row empty
            // 4. Column with many atoms
            // 5. Greater distance
            pullPriority[col - compZone[2]] = atomsInCurrentColumn ? (((unsigned int)(atomsInCurrentColumn > rowsRemaining) << 31) + 
                ((unsigned int)(atomsInNextFilledRow > (compZone[3] - compZone[2])) << 30) + 
                (((unsigned int)(!stateArray(targetRow,col))) << 29) + 
                (atomsInCurrentColumn << 21) + (distance << 13) + ((yDir > 0) << 12) + (col - border[2])) : 0;
        }
    }
}

bool moveAtomVertically(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger, 
    unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, unsigned int *atomsInColumnRemaining[2],
    size_t targetCol, size_t targetRow, int yDir)
{
    if(stateArray(targetRow,targetCol))
    {
        logger->error("Atom already present!");
        return false;
    }
    const auto& sourceRow = nextRowWithAtom[targetRow + (yDir > 0)][targetCol];
    if(!sourceRow.has_value())
    {
        logger->error("Source row for vertical move is not defined!");
        return false;
    }

    if(sourceRow.value() < 0 || !stateArray(sourceRow.value(),targetCol))
    {
        logger->error("nextRowWithAtom expects an atom in row {}, but none has been found! (index {})", sourceRow.value(), targetRow + (yDir > 0));
        return false;
    }
    stateArray(sourceRow.value(),targetCol) = 0;
    stateArray(targetRow,targetCol) = 1;
    
    std::vector<std::pair<double,double>> sites_list;
    sites_list.push_back({sourceRow.value(), targetCol});
    sites_list.push_back({targetRow, targetCol});
    Move m
    {
        .sites_list = sites_list,
        .distance = (double)abs((int)targetRow - (int)sourceRow.value()),
        .init_dir = Direction::VER
    };
    ml.push_back(std::move(m));

    for(size_t row = targetRow + yDir; row != sourceRow.value(); row += yDir)
    {
        nextRowWithAtom[row + (yDir > 0)][targetCol] = nextRowWithAtom[sourceRow.value() + (yDir > 0)][targetCol];
    }
    atomsPerRow[sourceRow.value()]--;
    atomsInColumnRemaining[yDir > 0][targetCol]--;
    
    return true;
}

bool moveAtomHorizontally(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger, 
    std::optional<size_t> **nextRowWithAtom, size_t targetCol, size_t targetRow, int xDir, int *orthogonalStatus)
{
    if(stateArray(targetRow,targetCol))
    {
        logger->error("Atom already present!");
        return false;
    }
    for(int i = 1;;i++)
    {
        size_t tmpCol = targetCol + i * xDir;
        if(stateArray(targetRow,tmpCol))
        {
            stateArray(targetRow,tmpCol) = 0;
            stateArray(targetRow,targetCol) = 1;
    
            std::vector<std::pair<double,double>> sites_list;
            sites_list.push_back({targetRow, tmpCol});
            sites_list.push_back({targetRow, targetCol});
            Move m
            {
                .sites_list = sites_list,
                .distance = (double)abs((int)targetCol - (int)tmpCol),
                .init_dir = Direction::HOR
            };
            ml.push_back(std::move(m));

            return true;
        }
        else if(abs(orthogonalStatus[tmpCol - border[2]]) == 2)
        {
            int yDir = orthogonalStatus[tmpCol - border[2]] / abs(orthogonalStatus[tmpCol - border[2]]);
            orthogonalStatus[tmpCol - border[2]] = 0;
            const auto& rowWithAtom = nextRowWithAtom[targetRow + (yDir > 0)][tmpCol];
            if(!rowWithAtom.has_value())
            {
                logger->error("There should be another atom in column {}, but there does not seem to be!", tmpCol);
                return false;
            }
            if(!stateArray(rowWithAtom.value(),tmpCol))
            {
                logger->error("There should be an atom at row {} col {}, but there does not seem to be!", rowWithAtom.value(), tmpCol);
                return false;
            }
            stateArray(rowWithAtom.value(),tmpCol) = 0;
            stateArray(targetRow,targetCol) = 1;
    
            std::vector<std::pair<double,double>> sites_list;
            sites_list.push_back({rowWithAtom.value(), tmpCol});
            sites_list.push_back({targetRow, tmpCol});
            sites_list.push_back({targetRow, targetCol});
            Move m
            {
                .sites_list = sites_list,
                .distance = (double)abs((int)targetCol - (int)tmpCol) + (int)rowWithAtom.value() - (int)targetRow,
                .init_dir = Direction::VER
            };
            ml.push_back(std::move(m));

            return true;
        }
        if(tmpCol == border[2] || tmpCol == border[3])
        {
            logger->error("Going out of bounds while traversing horizontally!");
            return false;
        }
    }
    logger->error("This state (moving atom horizonally) should never be reached!");
    return false;
}

bool handleCurrentPosition(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger, 
    unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, unsigned int *atomsInColumnRemaining[2], int *orthogonalStatus, 
    size_t targetRow, size_t col, int horizontalDirection)
{
    if(stateArray(targetRow,col))
    {
        return true;
    }
    if(orthogonalStatus[col - border[2]])
    {
        int dir = orthogonalStatus[col - border[2]];
        orthogonalStatus[col - border[2]] = 0;
        return moveAtomVertically(ml, border, stateArray, logger, atomsPerRow, nextRowWithAtom, 
            atomsInColumnRemaining, col, targetRow, dir);
    }
    return moveAtomHorizontally(ml, border, stateArray, logger, nextRowWithAtom, 
        col, targetRow, horizontalDirection, orthogonalStatus);
}

bool findAndExecuteMoveOrder(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, size_t compZone[4], 
    std::shared_ptr<spdlog::logger> logger, unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, 
    unsigned int *atomsInColumnRemaining[2], int *orthogonalStatus, size_t targetRow, int yDir)
{
    // Find move order and execute moves
    int atomsFound = 0;
    int atomsRequired = 0;
    size_t lastSeperator = (size_t)(compZone[2]);
    bool rightPart = false;

    unsigned int totalAtomsInRow = 0;
    for(size_t col = border[2]; col <= border[3]; col++)
    {
        // Add existing atoms
        totalAtomsInRow += stateArray(targetRow,col);
        if(col >= compZone[2] && col <= compZone[3])
        {
            // Add pulled atoms
            totalAtomsInRow += (orthogonalStatus[col - border[2]] != 0);
        }
    }
    if(totalAtomsInRow < compZone[3] - compZone[2] + 1)
    {
        for(size_t col = border[2]; col <= border[3]; col++)
        {
            if(col < compZone[2] || col > compZone[3])
            {
                if(yDir > -1 && nextRowWithAtom[targetRow + 1][col].has_value())
                {
                    orthogonalStatus[col - border[2]] = 2;
                    totalAtomsInRow++;
                    if(totalAtomsInRow >= compZone[3] - compZone[2] + 1)
                    {
                        break;
                    }
                }
                if(yDir < 1 && nextRowWithAtom[targetRow][col].has_value())
                {
                    orthogonalStatus[col - border[2]] = -2;
                    totalAtomsInRow++;
                    if(totalAtomsInRow >= compZone[3] - compZone[2] + 1)
                    {
                        break;
                    }
                }
            }
            if(col == compZone[2] - 1)
            {
                col = compZone[3];
            }
        }
    }

    for(size_t col = border[2]; col <= border[3]; col++)
    {
        // Add existing atoms
        atomsFound += stateArray(targetRow,col);
        if(col >= compZone[2] && col <= compZone[3])
        {
            atomsRequired++;
            // Add pulled atoms
        }
        atomsFound += (orthogonalStatus[col - border[2]] != 0);
        if(!rightPart && atomsRequired > atomsFound)
        {
            for(size_t i = col - 1; i >= lastSeperator; i--)
            {
                if(!handleCurrentPosition(ml, border, stateArray, compZone, logger, atomsPerRow, 
                    nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, targetRow, i, -1))
                {
                    return false;
                }
            }
            rightPart = true;
            lastSeperator = col;
        }
        if(rightPart && atomsFound >= atomsRequired)
        {
            for(size_t i = lastSeperator; i <= col && i <= compZone[3]; i++)
            {
                if(!handleCurrentPosition(ml, border, stateArray, compZone, logger, atomsPerRow, 
                    nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, targetRow, i, 1))
                {
                    return false;
                }
            }
            rightPart = false;
            lastSeperator = col + 1;
        }
    }
    if(rightPart)
    {
        logger->error("Not enough atoms in row");
        return false;
    }
    for(size_t i = compZone[3]; i >= lastSeperator; i--)
    {
        if(!handleCurrentPosition(ml, border, stateArray, compZone, logger, atomsPerRow, 
            nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, targetRow, i, -1))
        {
            return false;
        }
    }
    return true;
}

void removeOwnAtomsFromBuffers(size_t border[4], StateArrayAccessor& stateArray, 
    unsigned int *atomsInColumnRemaining[2], size_t targetRow, int relationMiddleRow)
{
    for(size_t col = border[2]; col < border[3]; col++)
    {
        if(stateArray(targetRow,col))
        {
            if(relationMiddleRow < 1)
            {
                atomsInColumnRemaining[0][col]--;
            }
            if(relationMiddleRow > -1)
            {
                atomsInColumnRemaining[1][col]--;
            }
        }
    }
}

bool pullFromBothDirections(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger, 
    unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, unsigned int *atomsInColumnRemaining[2], int targetRow, size_t rowsRemainingUpper, 
    size_t rowsRemainingLower, int countFromAbove, int countFromBelow)
{
    logger->debug("Sorting middle row {}", targetRow);
    removeOwnAtomsFromBuffers(border, stateArray, atomsInColumnRemaining, targetRow, 0);

    unsigned int *pullPriority = new (std::nothrow) unsigned int[2 * (compZone[3] - compZone[2] + 1)];
    if(pullPriority == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }

    findPullPriorities(stateArray, compZone, border, pullPriority, 
        atomsPerRow, nextRowWithAtom, atomsInColumnRemaining[0], targetRow, rowsRemainingUpper, -1);
    findPullPriorities(stateArray, compZone, border, pullPriority + compZone[3] - compZone[2] + 1, 
        atomsPerRow, nextRowWithAtom, atomsInColumnRemaining[1], targetRow, rowsRemainingLower, 1);
    qsort(pullPriority, 2 * (compZone[3] - compZone[2] + 1), sizeof(int), priorityCompare);

    int *orthogonalStatus = new (std::nothrow) int[border[3] - border[2] + 1]();
    if(orthogonalStatus == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }
    for(size_t i = 0; (countFromAbove > 0 || countFromBelow > 0) && i < 2 * (compZone[3] - compZone[2] + 1); i++)
    {
        int x = pullPriority[i] & 0xfff;
        int dir = pullPriority[i] & 0x1000;
        if(!orthogonalStatus[x] && ((dir && countFromBelow > 0) || (!dir && countFromAbove > 0)))
        {
            if(dir)
            {
                orthogonalStatus[x] = 1;
                countFromBelow--;
            }
            else
            {
                orthogonalStatus[x] = -1;
                countFromAbove--;
            }
        }
    }

    if(!findAndExecuteMoveOrder(ml, border, stateArray, compZone, logger, atomsPerRow, 
        nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, targetRow, 0))
    {
        return false;
    }

    delete[] pullPriority;
    delete[] orthogonalStatus;
    return true;
}

bool pullFromDirection(std::vector<Move>& ml, size_t border[4], StateArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger, 
    unsigned int *atomsPerRow, std::optional<size_t> **nextRowWithAtom, unsigned int *atomsInColumnRemaining[2], size_t targetRow, 
    size_t rowsRemaining, int yDir, int count)
{
    logger->debug("Sorting row {} from {}", targetRow, (yDir > 0) ? "below" : "above");
    int *orthogonalStatus = new (std::nothrow) int[border[3] - border[2] + 1]();
    if(orthogonalStatus == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }
    removeOwnAtomsFromBuffers(border, stateArray, atomsInColumnRemaining, targetRow, yDir);

    unsigned int *pullPriority = new (std::nothrow) unsigned int[compZone[3] - compZone[2] + 1];
    if(pullPriority == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }
    findPullPriorities(stateArray, compZone, border, pullPriority, atomsPerRow, nextRowWithAtom, 
        atomsInColumnRemaining[yDir > 0], targetRow, rowsRemaining, yDir);
    qsort(pullPriority, compZone[3] - compZone[2] + 1, sizeof(unsigned int), priorityCompare);

    for(int i = 0; i < count; i++)
    {
        if(pullPriority[i] == 0)
        {
            logger->info("Length 2 moves required for row {}", targetRow);
            break;
        }
        int x = pullPriority[i] & 0xfff;
        orthogonalStatus[x] = yDir;
    }
    
    if(!findAndExecuteMoveOrder(ml, border, stateArray, compZone, logger, atomsPerRow, 
        nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, targetRow, yDir))
    {
        return false;
    }

    delete[] pullPriority;
    delete[] orthogonalStatus;
    return true;
}

bool findMinimalBounds(StateArrayAccessor& stateArray, size_t compZone[4], 
    std::shared_ptr<spdlog::logger> logger, size_t border[4], size_t& middleRow, int& countFromBelow, bool& allowLength2Moves)
{
    unsigned int *atomsInColumnRemaining[2] = {new (std::nothrow) unsigned int[stateArray.cols()](), new (std::nothrow) unsigned int[stateArray.cols()]()}; // [0] up, [1] down

    unsigned int requiredPerRow = compZone[3] - compZone[2] + 1;
    unsigned int totalRequiredAtoms = requiredPerRow * (compZone[1] - compZone[0] + 1);

    unsigned int requiredAtoms = 0;
    unsigned int atomsFound = 0;
    unsigned int unusableAtoms = 0;

    std::optional<size_t> middleRowDownDir = std::nullopt;
    std::optional<size_t> middleRowUpDir = std::nullopt;

    bool sortable = false;

    for(size_t row = compZone[0]; row <= compZone[1]; row++)
    {
        for(size_t col = compZone[2]; col <= compZone[3]; col++)
        {
            atomsFound += stateArray(row,col);
        }
    }

    for(size_t i = 0; i < 4; i++)
    {
        border[i] = compZone[i];
    }

    // Gradually increase border from compZone until enough atoms are contained
    while(atomsFound < totalRequiredAtoms)
    {
        bool borderChange = false;
        if(border[2] > 0)
        {
            borderChange = true;
            border[2]--;
            for(size_t row = compZone[0]; row <= compZone[1]; row++)
            {
                atomsFound += stateArray(row,border[2]);
            }
        }
        if(border[3] < stateArray.cols() - 1)
        {
            borderChange = true;
            border[3]++;
            for(size_t row = compZone[0]; row <= compZone[1]; row++)
            {
                atomsFound += stateArray(row,border[3]);
            }
        }
        if(border[0] > 0)
        {
            borderChange = true;
            border[0]--;
            for(size_t col = compZone[2]; col <= compZone[3]; col++)
            {
                atomsFound += stateArray(border[0],col);
            }
        }
        if(border[1] < stateArray.rows() - 1)
        {
            borderChange = true;
            border[1]++;
            for(size_t col = compZone[2]; col <= compZone[3]; col++)
            {
                atomsFound += stateArray(border[1],col);
            }
        }

        if(!borderChange)
        {
            logger->info("Only {}/{} atoms in + region, moves of length 2 are required!", atomsFound, totalRequiredAtoms);
            delete[] atomsInColumnRemaining[0];
            delete[] atomsInColumnRemaining[1];
            allowLength2Moves = true;
            return true;
        }
    }
    
    // Try to find middle row for sorting
    // If impossible, increase border further
    while(!sortable)
    {
        // Analyse upper part and find middle
        for(size_t row = border[0]; row <= border[1]; row++)
        {
            unusableAtoms = 0;
            if(row >= compZone[0] && row <= compZone[1])
            {
                requiredAtoms += requiredPerRow;
            }
            for(size_t col = border[2]; col <= border[3]; col++)
            {
                if(stateArray(row,col))
                {
                    atomsInColumnRemaining[0][col]++;
                    if((row >= compZone[0] && row <= compZone[1]) || 
                        (col >= compZone[2] && col <= compZone[3]))
                    {
                        atomsFound++;
                    }
                }
                if((col >= compZone[2] && col <= compZone[3]))
                {
                    unsigned int usableAtoms = stateArray(row,col) + row - compZone[0] + 1;
                    if(row >= compZone[0] && atomsInColumnRemaining[0][col] > usableAtoms)
                    {
                        unusableAtoms += atomsInColumnRemaining[0][col] - usableAtoms;
                    }
                }
            }
            if(atomsFound - unusableAtoms < requiredAtoms)
            {
                middleRowDownDir = row;
                break;
            }
        }
        countFromBelow = requiredAtoms - atomsFound + unusableAtoms;

        requiredAtoms = 0;
        atomsFound = 0;
        // Analyse lower part and find middle
        for(size_t row = stateArray.rows() - 1; row >= 0; row--)
        {
            unusableAtoms = 0;
            if(row >= compZone[0] && row <= compZone[1])
            {
                requiredAtoms += requiredPerRow;
            }
            for(size_t col = 0; col < stateArray.cols(); col++)
            {
                if(stateArray(row,col))
                {
                    atomsInColumnRemaining[1][col]++;
                    if((row >= compZone[0] && row <= compZone[1]) || 
                        (col >= compZone[2] && col <= compZone[3]))
                    {
                        atomsFound++;
                    }
                }
                if(col >= compZone[2] && col <= compZone[3])
                {
                    unsigned int usableAtoms = stateArray(row,col) + row - compZone[0] + 1;
                    if(row >= compZone[0] && atomsInColumnRemaining[1][col] > usableAtoms)
                    {
                        unusableAtoms += atomsInColumnRemaining[1][col] - usableAtoms;
                    }
                }
            }
            if(atomsFound - unusableAtoms < requiredAtoms)
            {
                middleRowUpDir = row;
                break;
            }
        }

        if(!middleRowUpDir.has_value() || !middleRowDownDir.has_value() || middleRowDownDir.value() > middleRowUpDir.value() || 
            (middleRowDownDir.value() == middleRowUpDir.value() && ((int)(atomsFound - unusableAtoms) - (int)requiredAtoms + (int)requiredPerRow > countFromBelow)))
        {
            sortable = true;
        }
        else
        {
            bool borderChange = false;
            if(border[2] > 0)
            {
                borderChange = true;
                border[2]--;
            }
            if(border[3] < stateArray.cols() - 1)
            {
                borderChange = true;
                border[3]++;
            }
            if(border[0] > 0)
            {
                borderChange = true;
                border[0]--;
            }
            if(border[1] < stateArray.rows() - 1)
            {
                borderChange = true;
                border[1]++;
            }

            if(!borderChange)
            {
                logger->info("Not enough atoms in + region, moves of length 2 are required!");
                delete[] atomsInColumnRemaining[0];
                delete[] atomsInColumnRemaining[1];
                allowLength2Moves = true;
                return true;
            }
        }
    }

    /*  Set middle row according to analysis
        If enough atoms from both directions, just take middle
        If enough from one side, just take middle unless
            middle is not possible from other side, then take first row that is possible
        If neither direction has enough atoms, take average of overlap*/
    if(!middleRowDownDir.has_value())
    {
        if(!middleRowUpDir.has_value())
        {
            middleRow = (size_t)((compZone[0] + compZone[1]) / 2);
            countFromBelow = 0;
        }
        else
        {
            middleRow = (size_t)((compZone[0] + compZone[1]) / 2);
            if(middleRow <= middleRowUpDir.value())
            {
                middleRow = middleRowUpDir.value() + 1;
            }
            countFromBelow = 0;
        }
    }
    else if(!middleRowUpDir.has_value())
    {
        middleRow = middleRowDownDir.value() - 1;
        countFromBelow = 0;
    }
    else if(middleRowDownDir.value() > middleRowUpDir.value())
    {
        middleRow = (size_t)((middleRowUpDir.value() + middleRowDownDir.value()) / 2);
        countFromBelow = 0;
    }
    else if(middleRowDownDir.value() == middleRowUpDir.value())
    {
        middleRow = middleRowDownDir.value();
    }
    else
    {
        logger->error("Not enough atoms! Also, this point should never be reached!");
        return false;
    }

    logger->info("Setting border to x: {} - {}, y: {} - {}", border[2], border[3], border[0], border[1]);

    delete[] atomsInColumnRemaining[0];
    delete[] atomsInColumnRemaining[1];

    return true;
}

bool findRowToBalanceExcessAtoms(StateArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger, size_t& middleRow, int& countFromBelow)
{
    unsigned int totalAtoms = 0;
    unsigned int requiredPerRow = compZone[3] - compZone[2] + 1;
    unsigned int rowsInCompZone = compZone[1] - compZone[0] + 1;
    unsigned int totalRequiredAtoms = requiredPerRow * rowsInCompZone;

    int excessAtomsUp = 0;
    for(size_t row = 0; row < stateArray.rows(); row++)
    {
        for(size_t col = 0; col < stateArray.cols(); col++)
        {
            totalAtoms += stateArray(row,col);
            if(row < compZone[0])
            {
                excessAtomsUp += stateArray(row,col);
            }
        }
    }

    if(totalAtoms < totalRequiredAtoms)
    {
        logger->error("Not enough atoms in array!");
        return false;
    }
    int excessAtoms = totalAtoms - totalRequiredAtoms;

    for(size_t row = compZone[0]; row <= compZone[1]; row++)
    {
        unsigned int atomsInRow = 0;
        for(size_t col = 0; col < stateArray.cols(); col++)
        {
            atomsInRow += stateArray(row,col);
        }
        excessAtomsUp += atomsInRow - requiredPerRow;
        
        if(excessAtomsUp < excessAtoms / 2)
        {
            middleRow = row;
            int requiredAtoms = (int)requiredPerRow - (int)atomsInRow;
            if(requiredAtoms < 0)
            {
                countFromBelow = 0;
                return true;
            }
            else
            {
                int excessAbove = excessAtomsUp - atomsInRow + requiredPerRow;
                int excessBelow = excessAtoms - excessAtomsUp;
                if(excessAbove > excessBelow)
                {
                    unsigned int diff = excessAbove - excessBelow;
                    countFromBelow = (requiredAtoms - diff) / 2;
                }
                else
                {
                    unsigned int diff = excessBelow - excessAbove;
                    countFromBelow = diff + (requiredAtoms - diff) / 2;
                }
                if(countFromBelow > excessBelow)
                {
                    logger->error("Not enough atoms below middle row, aborting!");
                    return false;
                }
                else
                {
                    return true;
                }
            }
        }
    }

    logger->error("This should never be reached, probably too few atoms, aborting");
    return false;
}

bool mainSortingLoop(std::vector<Move>& ml, StateArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger)
{
    size_t rows = stateArray.rows();
    size_t cols = stateArray.cols();
    if(compZone[3] < compZone[2] || compZone[2] < 0 || compZone[3] >= cols ||
        compZone[1] < compZone[0] || compZone[0] < 0 || compZone[1] >= rows)
    {
        logger->error("Comp zone does not hold appropriate values");
        return false;
    }

    unsigned int requiredPerRow = compZone[3] - compZone[2] + 1;

    // Allocate memory 
    unsigned int *atomsPerRow = new (std::nothrow) unsigned int[rows]();
    if(atomsPerRow == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }
    std::optional<size_t> *lastRowWithAtomPerColumn = new (std::nothrow) std::optional<size_t>[cols];
    if(lastRowWithAtomPerColumn == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }

    // Middle row is represented twice: once up and once down
    std::optional<size_t> **nextRowWithAtom = new (std::nothrow) std::optional<size_t>*[rows + 1];
    if(nextRowWithAtom == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }
    for(size_t row = 0; row < rows + 1; row++)
    {
        nextRowWithAtom[row] = new (std::nothrow) std::optional<size_t>[cols];
        if(nextRowWithAtom[row] == 0)
        {
            logger->error("Failed to allocate memory");
            return false;
        }
    }
    unsigned int *atomsInColumnRemaining[2] = {new (std::nothrow) unsigned int[cols](), new (std::nothrow) unsigned int[cols]()}; // [0] up, [1] down
    if(atomsInColumnRemaining[0] == 0 || atomsInColumnRemaining[1] == 0)
    {
        logger->error("Failed to allocate memory");
        return false;
    }

    size_t border[4];
    size_t middleRow = 0;
    int countFromBelow;
    bool allowLength2Moves = false;
    if(!findMinimalBounds(stateArray, compZone, logger, border, middleRow, countFromBelow, allowLength2Moves))
    {
        return false;
    }
    if(allowLength2Moves)
    {
        if(!findRowToBalanceExcessAtoms(stateArray, compZone, logger, middleRow, countFromBelow))
        {
            logger->error("Could not find middle row for case with too few atoms in + region!");
            return false;
        }
    }

    // Analyse upper part
    for(size_t col = 0; col < cols; col++)
    {
        // Initialize manually size default constructor may depend on compiler
        lastRowWithAtomPerColumn[col] = std::nullopt;
    }
    for(size_t row = border[0]; row <= middleRow; row++)
    {
        for(size_t col = border[2]; col <= border[3]; col++)
        {
            // Set for each site the last row which contained an atom in the same column
            nextRowWithAtom[row][col] = lastRowWithAtomPerColumn[col];
            if(stateArray(row,col))
            {
                atomsInColumnRemaining[0][col]++;
                atomsPerRow[row]++;
                lastRowWithAtomPerColumn[col] = row;
            }
        }
    }
    // Analyse lower part
    for(size_t col = 0; col < cols; col++)
    {
        lastRowWithAtomPerColumn[col] = std::nullopt;
    }
    for(size_t row = border[1]; row >= middleRow; row--)
    {
        for(size_t col = border[2]; col <= border[3]; col++)
        {
            nextRowWithAtom[row + 1][col] = lastRowWithAtomPerColumn[col];
            if(stateArray(row,col))
            {
                atomsInColumnRemaining[1][col]++;
                lastRowWithAtomPerColumn[col] = row;
                if(row != middleRow)
                {
                    atomsPerRow[row]++;
                }
            }
        }
    }

    if(!pullFromBothDirections(ml, border, stateArray, compZone, logger, atomsPerRow, nextRowWithAtom, 
        atomsInColumnRemaining, middleRow, middleRow - compZone[0], compZone[1] - middleRow, 
        requiredPerRow - atomsPerRow[middleRow] - countFromBelow, countFromBelow))
    {
        return false;
    }

    for(size_t row = middleRow - 1; row >= compZone[0]; row--)
    {
        int count = requiredPerRow - atomsPerRow[row];
        if(!pullFromDirection(ml, border, stateArray, compZone, logger, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, row, row - compZone[0], -1, count))
        {
            return false;
        }
    }
    for(size_t row = middleRow + 1; row <= compZone[1]; row++)
    {
        int count = requiredPerRow - atomsPerRow[row];
        if(!pullFromDirection(ml, border, stateArray, compZone, logger, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, row, compZone[1] - row, 1, count))
        {
            return false;
        }
    }

    delete[] atomsPerRow;
    delete[] lastRowWithAtomPerColumn;
    delete[] atomsInColumnRemaining[0];
    delete[] atomsInColumnRemaining[1];
    for(size_t row = 0; row < rows + 1; row++)
    {
        delete[] nextRowWithAtom[row];
    }
    delete[] nextRowWithAtom;

    return true;
}

bool sortSequentiallyByRowCA(std::vector<Move>& ml, size_t rows, size_t cols, bool** stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger)
{
    CStyleStateArrayAccessor stateArrayAccessor(stateArray, rows, cols);
    return mainSortingLoop(ml, stateArrayAccessor, compZone, logger);
}

std::optional<std::vector<Move>> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd)
{
    std::shared_ptr<spdlog::logger> logger;
    Config& config = Config::getInstance();
    if((logger = spdlog::get(config.sequentialLoggerName)) == nullptr)
    {
        logger = spdlog::basic_logger_mt(config.sequentialLoggerName, config.logFileName);
    }
    logger->set_level(spdlog::level::debug);

    std::vector<Move> moves;

    size_t compZone[4] = {compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd};

    EigenArrayStateArrayAccessor stateArrayAccessor(stateArray);
    bool success = mainSortingLoop(moves, stateArrayAccessor, compZone, logger);

    if(success)
    {
        return moves;
    }
    else
    {
        return std::nullopt;
    }
}