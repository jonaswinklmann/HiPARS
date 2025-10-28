#pragma once

#include <fstream>
#include <ranges>
#include <string>
#include <spdlog/spdlog.h>
#include <stdexcept>

#define DOUBLE_EQUIVALENCE_THRESHOLD 0.00001
#define M_4TH_ROOT_2 1.1892071150027210667
#define M_4TH_ROOT_1_2 1 / M_4TH_ROOT_2

#define CONFIG_LOG_NAME "configLogger"

class Config
{
    public:
        static Config& getInstance()
        {
            static Config instance;
            return instance;
        }
    private:
        Config() : logFileName("sorting.log"), sequentialLoggerName("sequentialSortingLogger"), parallelLoggerName("parallelSortingLogger"),
            greedyLatticeLoggerName("greedyLatticeSortingLogger"), latticeByRowLoggerName("latticeByRowSortingLogger"),
            rowSpacing(1), columnSpacing(1), allowMovingEmptyTrapOntoOccupied(true), allowDiagonalMovement(true), 
            allowMovesBetweenRows(true), allowMovesBetweenCols(true), allowMultipleMovesPerAtom(false), aodTotalLimit(256), aodRowLimit(16), aodColLimit(16),
            moveCostOffset(150), moveCostOffsetSubmove(0), moveCostScalingSqrt(0), moveCostScalingLinear(0.1),
            recommendedDistFromOccSites(1), recommendedDistFromEmptySites(0.1), minDistFromOccSites(1), maxSubmoveDistInPenalizedArea(1.5) {}
    public:
        Config(Config const&) = delete;
        void operator=(Config const&) = delete;
        void flushLogs()
        {
            std::shared_ptr<spdlog::logger> logger;
            if((logger = spdlog::get(sequentialLoggerName)) != nullptr)
            {
                logger->flush();
            }
            if((logger = spdlog::get(parallelLoggerName)) != nullptr)
            {
                logger->flush();
            }
            if((logger = spdlog::get(greedyLatticeLoggerName)) != nullptr)
            {
                logger->flush();
            }
            if((logger = spdlog::get(latticeByRowLoggerName)) != nullptr)
            {
                logger->flush();
            }
        };
        bool readConfig(std::string filePath)
        {
            std::ifstream configFile(filePath, std::ios_base::openmode::_S_in);
            std::string line;
            
            if(!configFile.is_open())
            {
                return false;
            }

            // Set all given values, ignore all other lines
            while(std::getline(configFile, line))
            {
                std::istringstream configLine(line);
                auto delim = line.find('=');
                if(delim != std::string::npos)
                {
                    std::string key = line.substr(0, delim);
                    std::string::iterator end_pos = std::remove(key.begin(), key.end(), ' ');
                    key.erase(end_pos, key.end());
                    std::string val = line.substr(delim + 1);
                    end_pos = std::remove(val.begin(), val.end(), ' ');
                    val.erase(end_pos, val.end());

                    if(key.compare("logFileName") == 0)
                    {
                        logFileName = val;
                    }
                    else if(key.compare("sequentialLoggerName") == 0)
                    {
                        sequentialLoggerName = val;
                    }
                    else if(key.compare("parallelLoggerName") == 0)
                    {
                        parallelLoggerName = val;
                    }
                    else if(key.compare("greedyLatticeLoggerName") == 0)
                    {
                        greedyLatticeLoggerName = val;
                    }
                    else if(key.compare("latticeByRowLoggerName") == 0)
                    {
                        latticeByRowLoggerName = val;
                    }
                    else if(key.compare("rowSpacing") == 0)
                    {
                        rowSpacing = std::stod(val);
                    }
                    else if(key.compare("columnSpacing") == 0)
                    {
                        columnSpacing = std::stod(val);
                    }
                    else if(key.compare("allowMovingEmptyTrapOntoOccupied") == 0)
                    {
                        std::transform(val.begin(), val.end(), val.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        allowMovingEmptyTrapOntoOccupied = val.compare("true") == 0;
                    }
                    else if(key.compare("allowDiagonalMovement") == 0)
                    {
                        std::transform(val.begin(), val.end(), val.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        allowDiagonalMovement = val.compare("true") == 0;
                    }
                    else if(key.compare("allowMovesBetweenRows") == 0)
                    {
                        std::transform(val.begin(), val.end(), val.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        allowMovesBetweenRows = val.compare("true") == 0;
                    }
                    else if(key.compare("allowMovesBetweenCols") == 0)
                    {
                        std::transform(val.begin(), val.end(), val.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        allowMovesBetweenCols = val.compare("true") == 0;
                    }
                    else if(key.compare("allowMultipleMovesPerAtom") == 0)
                    {
                        std::transform(val.begin(), val.end(), val.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        allowMultipleMovesPerAtom = val.compare("true") == 0;
                    }
                    else if(key.compare("aodTotalLimit") == 0)
                    {
                        aodTotalLimit = std::stoi(val);
                    }
                    else if(key.compare("aodRowLimit") == 0)
                    {
                        aodRowLimit = std::stoi(val);
                    }
                    else if(key.compare("aodColLimit") == 0)
                    {
                        aodColLimit = std::stoi(val);
                    }
                    else if(key.compare("moveCostOffset") == 0)
                    {
                        moveCostOffset = stod(val);
                    }
                    else if(key.compare("moveCostOffsetSubmove") == 0)
                    {
                        moveCostOffsetSubmove = stod(val);
                    }
                    else if(key.compare("moveCostScalingSqrt") == 0)
                    {
                        moveCostScalingSqrt = stod(val);
                    }
                    else if(key.compare("moveCostScalingLinear") == 0)
                    {
                        moveCostScalingLinear = stod(val);
                    }
                    else if(key.compare("recommendedDistFromOccSites") == 0)
                    {
                        recommendedDistFromOccSites = stod(val);
                    }
                    else if(key.compare("recommendedDistFromEmptySites") == 0)
                    {
                        recommendedDistFromEmptySites = stod(val);
                    }
                    else if(key.compare("minDistFromOccSites") == 0)
                    {
                        minDistFromOccSites = stod(val);
                    }
                    else if(key.compare("maxSubmoveDistInPenalizedArea") == 0)
                    {
                        maxSubmoveDistInPenalizedArea = stod(val);
                    }
                }
            }
            return true;
        }
        std::string logFileName;
        std::string sequentialLoggerName;
        std::string parallelLoggerName;
        std::string greedyLatticeLoggerName;
        std::string latticeByRowLoggerName;
        double rowSpacing, columnSpacing;
        bool allowMovingEmptyTrapOntoOccupied, allowDiagonalMovement, allowMovesBetweenRows, allowMovesBetweenCols, allowMultipleMovesPerAtom;
        unsigned int aodTotalLimit, aodRowLimit, aodColLimit;
        double moveCostOffset, moveCostOffsetSubmove, moveCostScalingSqrt, moveCostScalingLinear;
        double recommendedDistFromOccSites, recommendedDistFromEmptySites, minDistFromOccSites, maxSubmoveDistInPenalizedArea;
};