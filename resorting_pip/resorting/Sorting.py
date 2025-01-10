import sys
import os
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resorting_cpp
import numpy as np

class Sorting:
    """Python wrapper class for sorting arrays of neutral atoms
    """

    def __init__(self):
        """Constructor method
        """
    
    def configure_log(self, log_file_name : str = None, parallel_logger_name : str = None, sequential_logger_name : str = None):
        """Function for configuring the logger

        :param log_file_name: The name of the log file, defaults to None
        :type log_file_name: str, optional
        :param parallel_logger_name: The name of the parallel logger, defaults to None
        :type parallel_logger_name: str, optional
        :param sequential_logger_name: The name of the sequential logger, defaults to None
        :type sequential_logger_name: str, optional
        """
        config = resorting_cpp.Config()
        if log_file_name:
            config.logFileName = log_file_name
        if parallel_logger_name:
            config.sequentialLoggerName = parallel_logger_name
        if sequential_logger_name:
            config.parallelLoggerName = sequential_logger_name

    def sort_sequentially(self, state_array, comp_zone_row_range, comp_zone_col_range):
        """Function for sorting sequentially

        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param comp_zone_row_range: Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        :type comp_zone_row_range: tuple(int,int)
        :param comp_zone_col_range: Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        :type comp_zone_col_range: tuple(int,int)
        :raises TypeError: state_array must be numpy bool array
        :raises TypeError: state_array must be dtype bool
        :return: A list of moves to sort array or None if sorting has failed. A move contains .distance, .init_dir, and .sites_list, which is a list of coordinate pairs to traverse
        :rtype: list[Move], optional
        """
        if not isinstance(state_array, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        return resorting_cpp.sortSequentiallyByRow(state_array, *comp_zone_row_range, *comp_zone_col_range)
    
    def sort_parallel(self, state_array, comp_zone_row_range, comp_zone_col_range):
        """Function for sorting in parallel
        
        :param state_array: The array of boolean values to be sorted
        :type state_array: np.ndarray[bool]
        :param comp_zone_row_range: Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        :type comp_zone_row_range: tuple(int,int)
        :param comp_zone_col_range: Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        :type comp_zone_col_range: tuple(int,int)
        :raises TypeError: state_array must be numpy bool array
        :raises TypeError: state_array must be dtype bool
        :return: A list of moves to sort array or None if sorting has failed. A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, each containing .colSelection and .rowSelection, which are lists of doubles
        :rtype: list[ParallelMove], optional
        """
        if not isinstance(state_array, np.ndarray):
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == bool:
            raise TypeError("state_array must be dtype bool")
        return resorting_cpp.sortParallel(state_array, *comp_zone_row_range, *comp_zone_col_range)

    def flush_logs(self):
        """Function for flushing the logs
        """
        resorting_cpp.Config().flushLogs()
