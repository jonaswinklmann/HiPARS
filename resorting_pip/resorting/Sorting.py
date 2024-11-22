"""@package ImageGenerator
Module for generating images of neutral atoms in a grid"""

import sys
import os
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resorting_cpp
import numpy as np

class Sorting:
    """Main class for sorting array of neutral atoms"""

    def __init__(self):
        """Constructor"""
    
    def configure_log(self, log_file_name : str = None, parallel_logger_name : str = None, sequential_logger_name : str = None):
        config = resorting_cpp.Config()
        if log_file_name:
            config.logFileName = log_file_name
        if parallel_logger_name:
            config.sequentialLoggerName = parallel_logger_name
        if sequential_logger_name:
            config.parallelLoggerName = sequential_logger_name

    def sort_sequentially(self, state_array, comp_zone_row_range, comp_zone_col_range):
        """Function for sorting sequentially
        @param state_array The array of boolean values to be sorted
        @param comp_zone_row_range Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        @param comp_zone_col_range Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        @return list[Move]|None A list of moves to sort array or None if sorting has failed. 
        A move contains .distance, .init_dir, and .sites_list, which is a list of coordinate pairs to traverse"""
        if not type(state_array) is np.ndarray:
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == np.bool:
            raise TypeError("state_array must be dtype bool")
        return resorting_cpp.sortSequentiallyByRow(state_array, *comp_zone_row_range, *comp_zone_col_range)
    
    def sort_parallel(self, state_array, comp_zone_row_range, comp_zone_col_range):
        """Function for sorting in parallel
        @param state_array The array of boolean values to be sorted
        @param comp_zone_row_range Tuple (start,end) of start(inclusive) and end(exclusive) of rows in computational zone
        @param comp_zone_col_range Tuple (start,end) of start(inclusive) and end(exclusive) of columns in computational zone
        @return list[ParallelMove]|None A list of moves to sort array or None if sorting has failed. 
        A ParallelMove contains .steps, which is a list of ParallelMoveStep objects, 
        each containing .colSelection and .rowSelection, which are lists of doubles"""
        if not type(state_array) is np.ndarray:
            raise TypeError("state_array must be numpy bool array")
        if not state_array.dtype == np.bool:
            raise TypeError("state_array must be dtype bool")
        return resorting_cpp.sortParallel(state_array, *comp_zone_row_range, *comp_zone_col_range)

    def flush_logs(self):
        resorting_cpp.Config().flushLogs()