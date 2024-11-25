#include <string>
#include <spdlog/spdlog.h>

#define DEFAULT_LOGFILE_NAME "sorting.log"

#define DEFAULT_PARALLEL_LOGGER_NAME "parallelSortingLogger"
#define DEFAULT_SEQUENTIAL_LOGGER_NAME "sequentialSortingLogger"

class Config
{
    public:
        static Config& getInstance()
        {
            static Config instance;
            return instance;
        }
    private:
        Config() : logFileName(DEFAULT_LOGFILE_NAME), sequentialLoggerName(DEFAULT_SEQUENTIAL_LOGGER_NAME), parallelLoggerName(DEFAULT_PARALLEL_LOGGER_NAME) {}
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
        };
        std::string logFileName;
        std::string sequentialLoggerName;
        std::string parallelLoggerName;
};