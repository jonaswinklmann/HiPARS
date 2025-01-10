#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/attr.h>
#include "sortSequentiallyByRow.hpp"
#include "sortParallel.hpp"
#include "config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "Underlying C++ library for sorting neutral atoms\n";

    py::class_<ParallelMove>(m, "ParallelMove", R"pbdoc(
    A class holding information about a parallel move.
)pbdoc")
    .def(py::init())
    .def_readwrite("steps", &ParallelMove::steps, R"pbdoc(
    :list[:class:`.ParallelMoveStep`]: Individual steps of the move
)pbdoc");

    py::class_<ParallelMove::Step>(m, "ParallelMoveStep", R"pbdoc(
    A class holding information about a parallel move step.
)pbdoc")
    .def(py::init())
    .def_readwrite("colSelection", &ParallelMove::Step::colSelection, R"pbdoc(
    :list[double]: List of column tones to activate at this step
)pbdoc")
    .def_readwrite("rowSelection", &ParallelMove::Step::rowSelection, R"pbdoc(
    :list[double]: List of column tones to activate at this step
)pbdoc");

    py::enum_<Direction>(m, "Direction", R"pbdoc(
    An enum holding the different potential starting AOD directions.
)pbdoc")
    .value("HOR", Direction::HOR)
    .value("VER", Direction::VER)
    .value("NONE", Direction::NONE)
    .value("DIAG", Direction::DIAG)
    .export_values();

    py::class_<Move>(m, "Move", R"pbdoc(
    A class holding information about a sequential move.
)pbdoc")
    .def(py::init())
    .def_readwrite("sites_list", &Move::sites_list, R"pbdoc(
    :list[(double,double)]: Coordinates of movable tweezer at each step
)pbdoc")
    .def_readwrite("distance", &Move::distance, R"pbdoc(
    :double: Total move length
)pbdoc")
    .def_readwrite("init_dir", &Move::init_dir, R"pbdoc(
    :class:`.Direction`: Initial move direction
)pbdoc");

    py::class_<Config, std::unique_ptr<Config, py::nodelete>>(m, "Config", R"pbdoc(
    A class for configuring the sorting algorithm. At the moment, this is only used for configuring the logger.
)pbdoc")
    .def(py::init<>([]{return &Config::getInstance();}))
    .def_readwrite("logFileName", &Config::logFileName, R"pbdoc(
    :string: Name of the log file
)pbdoc")
    .def_readwrite("sequentialLoggerName", &Config::sequentialLoggerName, R"pbdoc(
    :string: Name of the sequential logger
)pbdoc")
    .def_readwrite("parallelLoggerName", &Config::parallelLoggerName, R"pbdoc(
    :string: Name of the parallel logger
)pbdoc")
    .def("flushLogs", &Config::flushLogs);

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"), R"pbdoc(
    A function that sorts an array of atoms sequentially row by row

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[Move] | None
)pbdoc");

    m.def("sortParallel", &sortParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"), R"pbdoc(
    A function that sorts an array of atoms in parallel

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: numpy.ndarray[bool[m, n], flags.writeable]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[ParallelMove] | None
)pbdoc");
};