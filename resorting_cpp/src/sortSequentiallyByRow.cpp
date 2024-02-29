#include "sortSequentiallyByRow.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <fstream>

int priorityCompare(const void* a, const void* b)
{
    unsigned int int_a = * ( (unsigned int*) a );
    unsigned int int_b = * ( (unsigned int*) b );

    if ( int_a == int_b ) return 0;
    else if ( int_a > int_b ) return -1;
    else return 1;
}

void findPullPriorities(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    unsigned int *pullPriority, int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining, 
    int targetRow, int rowsRemaining, int yDir, Eigen::Array2i filledShape, int* borderSize)
{
    for(int x = borderSize[1]; x < borderSize[1] + filledShape(1); x++)
    {
        int nextFilledRow = nextRowWithAtom(targetRow + (yDir > 0),x);
        int atomsInNextFilledRow = 0;
        if(nextFilledRow > 0)
        {
            atomsInNextFilledRow = atomsPerRow[nextFilledRow];
        }
        unsigned int distance = (nextFilledRow - targetRow) * yDir;
        unsigned int atomsInCurrentColumn = atomsInColumnRemaining[x];
        // Priority:
        // 1. Column with too many atoms
        // 2. Row with too many atoms
        // 3. Location in this row empty
        // 4. Column with many atoms
        // 5. Greater distance
        pullPriority[x - borderSize[1]] = atomsInCurrentColumn ? (((unsigned int)(atomsInCurrentColumn > (unsigned int)rowsRemaining) << 31) + 
            ((unsigned int)(atomsInNextFilledRow > filledShape(1)) << 30) + 
            (((unsigned int)(!stateArray(targetRow,x)) << 29) + 
            (atomsInCurrentColumn << 21) + (distance << 13) + ((yDir > 0) << 12) + (x - borderSize[1]))) : 0;
    }
}

bool moveAtomVertically(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining[2],
    int targetX, int targetY, int yDir, std::vector<Move>& moves)
{
    if(stateArray(targetY,targetX))
    {
        printf("Atom already present!");
        return false;
    }
    int sourceRow = nextRowWithAtom(targetY + (yDir > 0),targetX);

    if(!stateArray(sourceRow,targetX))
    {
        printf("nextRowWithAtom expects an atom in row sourceRow, but none has been found!");
        return false;
    }
    stateArray(sourceRow,targetX) = 0;
    stateArray(targetY,targetX) = 1;
    moves.push_back(Move{(unsigned int)targetX, (unsigned int)sourceRow, 0, targetY - sourceRow});
    for(int i = targetY + yDir; i != sourceRow; i += yDir)
    {
        nextRowWithAtom(i + (yDir > 0),targetX) = nextRowWithAtom(sourceRow + (yDir > 0),targetX);
    }
    atomsPerRow[sourceRow]--;
    atomsInColumnRemaining[yDir > 0][targetX]--;
    
    return true;
}

bool moveAtomHorizontally(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int targetX, int targetY, int xDir, std::vector<Move>& moves)
{
    if(stateArray(targetY,targetX))
    {
        printf("Atom already present!");
        return false;
    }
    for(int i = 1;;i++)
    {
        int tmpX = targetX + i * xDir;
        if(stateArray(targetY,tmpX))
        {
            stateArray(targetY,tmpX) = 0;
            stateArray(targetY,targetX) = 1;
            moves.push_back(Move{(unsigned int)tmpX, (unsigned int)targetY, targetX - tmpX, 0});
            return true;
        }
        if(tmpX == 0 || tmpX == stateArray.cols() - 1)
        {
            printf("Going out of bounds while traversing horizontally!");
            return false;
        }
    }
}

bool handleCurrentPosition(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining[2], int *orthogonalStatus, 
    int targetRow, int x, int* borderSize, int horizontalDirection, std::vector<Move>& moves)
{
    if(stateArray(targetRow,x))
    {
        return true;
    }
    if(orthogonalStatus[x - borderSize[1]])
    {
        return moveAtomVertically(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, 
            x, targetRow, orthogonalStatus[x - borderSize[1]], moves);
    }
    return moveAtomHorizontally(stateArray, x, targetRow, horizontalDirection, moves);
}

void findAndExecuteMoveOrder(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining[2], int *orthogonalStatus, 
    int targetRow, Eigen::Array2i filledShape, int* borderSize, std::vector<Move>& moves)
{
    // Find move order and execute moves
    int atomsFound = 0;
    int atomsRequired = 0;
    int lastSeperator = borderSize[1];
    bool rightPart = false;
    for(int x = 0; x < stateArray.cols(); x++)
    {
        // Add existing atoms
        atomsFound += stateArray(targetRow,x);
        if(x >= borderSize[1] && x < borderSize[1] + filledShape(1))
        {
            atomsRequired++;
            // Add pulled atoms
            atomsFound += (orthogonalStatus[x - borderSize[1]] != 0);
        }
        if(!rightPart && atomsRequired > atomsFound)
        {
            for(int i = x - 1; i >= lastSeperator; i--)
            {
                handleCurrentPosition(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, 
                    orthogonalStatus, targetRow, i, borderSize, -1, moves);
            }
            rightPart = true;
            lastSeperator = x;
        }
        if(rightPart && atomsFound >= atomsRequired)
        {
            for(int i = lastSeperator; i <= x && i < borderSize[1] + filledShape(1); i++)
            {
                handleCurrentPosition(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, 
                    orthogonalStatus, targetRow, i, borderSize, 1, moves);
            }
            rightPart = false;
            lastSeperator = x + 1;
        }
    }
    if(rightPart)
    {
        printf("Not enough atoms in row");
        return;
    }
    for(int i = borderSize[1] + filledShape(1) - 1; i >= lastSeperator; i--)
    {
        handleCurrentPosition(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, 
            orthogonalStatus, targetRow, i, borderSize, -1, moves);
    }
}

void removeOwnAtomsFromBuffers(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsInColumnRemaining[2], int targetRow, int relationMiddleRow)
{
    for(int x = 0; x < stateArray.cols(); x++)
    {
        if(stateArray(targetRow,x))
        {
            if(relationMiddleRow < 1)
            {
                atomsInColumnRemaining[0][x]--;
            }
            if(relationMiddleRow > -1)
            {
                atomsInColumnRemaining[1][x]--;
            }
        }
    }
}

void pullFromBothDirections(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining[2], int targetRow, int rowsRemainingUpper, 
    int rowsRemainingLower, int countFromAbove, int countFromBelow, 
    Eigen::Array2i filledShape, int* borderSize, std::vector<Move>& moves)
{
    removeOwnAtomsFromBuffers(stateArray, atomsInColumnRemaining, targetRow, 0);

    unsigned int *pullPriority = (unsigned int*)malloc(2 * filledShape(1) * sizeof(int));
    findPullPriorities(stateArray, pullPriority, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining[0], targetRow,
        rowsRemainingUpper, -1, filledShape, borderSize);
    findPullPriorities(stateArray, pullPriority + filledShape(1), atomsPerRow, nextRowWithAtom, atomsInColumnRemaining[1], targetRow,
        rowsRemainingLower, 1, filledShape, borderSize);
    qsort(pullPriority, 2 * filledShape(1), sizeof(int), priorityCompare);

    int *orthogonalStatus = (int*)calloc(filledShape(1), sizeof(int));
    for(int i = 0; countFromAbove > 0 || countFromBelow > 0; i++)
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

    findAndExecuteMoveOrder(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, orthogonalStatus, 
        targetRow, filledShape, borderSize, moves);
    free(pullPriority);
    free(orthogonalStatus);
}

int pullFromDirection(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    int *atomsPerRow, Eigen::Ref<Eigen::ArrayXXi> nextRowWithAtom, int *atomsInColumnRemaining[2], int targetRow, 
    int rowsRemaining, int yDir, int count, Eigen::Array2i filledShape, int* borderSize, std::vector<Move>& moves)
{
    int *orthogonalStatus = (int*)calloc(filledShape(1), sizeof(int));
    removeOwnAtomsFromBuffers(stateArray, atomsInColumnRemaining, targetRow, yDir);

    unsigned int *pullPriority = (unsigned int*)malloc(filledShape(1) * sizeof(unsigned int));
    findPullPriorities(stateArray, pullPriority, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining[yDir > 0], targetRow,
        rowsRemaining, yDir, filledShape, borderSize);
    qsort(pullPriority, filledShape(1), sizeof(unsigned int), priorityCompare);

    for(int i = 0; i < count; i++)
    {
        if(pullPriority[i] == 0)
        {
            printf("Not enough atoms remaining");
            return count - i;
        }
        int x = pullPriority[i] & 0xfff;
        orthogonalStatus[x] = yDir;
    }

    findAndExecuteMoveOrder(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, 
        orthogonalStatus, targetRow, filledShape, borderSize, moves);

    free(pullPriority);
    free(orthogonalStatus);
    return 0;
}

std::vector<Move> sortSequentiallyByRow(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    Eigen::Array2i filledShape)
{
    std::vector<Move> finalMoves;
    int borderSize[2] = {((int)stateArray.rows() - filledShape(0)) / 2, ((int)stateArray.cols() - filledShape(1)) / 2};
    int atomsFound = 0;
    int requiredAtoms = 0;
    int middleRow = 0;
    int unusableAtoms = 0;
    int *atomsPerRow = (int*)calloc(stateArray.rows(), sizeof(int));
    int *lastRowWithAtomPerColumn = (int*)malloc(stateArray.cols() * sizeof(int));

    // Middle row is represented twice: once up and once down
    Eigen::ArrayXXi nextRowWithAtom(stateArray.rows() + 1, stateArray.cols());
    int *atomsInColumnRemaining[2] = {(int*)calloc(stateArray.cols(), sizeof(int)),(int*)calloc(stateArray.cols(), sizeof(int))}; // [0] up, [1] down
    for(int i = 0; i < stateArray.cols(); i++)
    {
        lastRowWithAtomPerColumn[i] = -1;
    }

    // Analyse upper part and find middle
    for(int y = 0; y < stateArray.rows(); y++)
    {
        unusableAtoms = 0;
        if(y >= borderSize[0] && y < borderSize[0] + filledShape(0))
        {
            requiredAtoms += filledShape(1);
        }
        for(int x = 0; x < stateArray.cols(); x++)
        {
            nextRowWithAtom(y,x) = lastRowWithAtomPerColumn[x];
            if(stateArray(y,x))
            {
                atomsInColumnRemaining[0][x]++;
                atomsPerRow[y]++;
                if((y >= borderSize[0] && y < borderSize[0] + filledShape(0)) || 
                    (x >= borderSize[1] && x < borderSize[1] + filledShape(1)))
                {
                    atomsFound++;
                }

                lastRowWithAtomPerColumn[x] = y;
            }
            if(x >= borderSize[1] && x < borderSize[1] + filledShape(1))
            {
                int usableAtoms = stateArray(y,x) + y - borderSize[0] + 1;
                if(y >= borderSize[0] && atomsInColumnRemaining[0][x] > usableAtoms)
                {
                    unusableAtoms += atomsInColumnRemaining[0][x] - usableAtoms;
                }
            }
        }
        if(atomsFound - unusableAtoms < requiredAtoms)
        {
            middleRow = y;
            break;
        }
    }
    int countFromBelow = requiredAtoms - atomsFound + unusableAtoms;

    if(middleRow == 0)
    {
        middleRow = borderSize[1] + filledShape(1) - 1;
        countFromBelow = 0;
    }

    // Analyse lower part
    for(int i = 0; i < stateArray.cols(); i++)
    {
        lastRowWithAtomPerColumn[i] = -1;
    }
    for(int y = stateArray.rows() - 1; y >= middleRow; y--)
    {
        for(int x = 0; x < stateArray.cols(); x++)
        {
            nextRowWithAtom(y+1,x) = lastRowWithAtomPerColumn[x];
            if(stateArray(y,x))
            {
                atomsInColumnRemaining[1][x]++;
                lastRowWithAtomPerColumn[x] = y;
                if(y != middleRow)
                {
                    atomsPerRow[y]++;
                }
            }
        }
    }

    pullFromBothDirections(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, middleRow, middleRow - borderSize[0],
        borderSize[0] + filledShape(0) - middleRow - 1, filledShape(1) - atomsPerRow[middleRow] - countFromBelow, 
        countFromBelow, filledShape, borderSize, finalMoves);

    for(int y = middleRow - 1; y >= borderSize[0]; y--)
    {
        int count = filledShape(1) - atomsPerRow[y];
        pullFromDirection(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, y, y - borderSize[0], -1, 
            count, filledShape, borderSize, finalMoves);
    }
    for(int y = middleRow + 1; y < borderSize[0] + filledShape(0); y++)
    {
        int count = filledShape(1) - atomsPerRow[y];
        pullFromDirection(stateArray, atomsPerRow, nextRowWithAtom, atomsInColumnRemaining, y, 
            borderSize[0] + filledShape(0) - y - 1, 1, 
            count, filledShape, borderSize, finalMoves);
    }

    free(atomsPerRow);
    free(lastRowWithAtomPerColumn);
    for(int i = 0; i < 2; i++)
    {
        free(atomsInColumnRemaining[i]);
    }

    return finalMoves;
}