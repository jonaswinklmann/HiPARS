#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/attr.h>
#include "sortSequentiallyByRow.hpp"
#include "sortParallel.hpp"
#include "config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "pybind11 resorting module\n";

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"),R"pbdoc(
    A function that sorts an array of atoms sequentially row by row

    :param stateArray: The array of boolean values to be sorted
    :type stateArray: np.ndarray[bool]
    :param compZoneRowStart: Start row of computational zone (inclusive)
    :type compZoneRowStart: int
    :param compZoneRowEnd: End row of computational zone (exclusive)
    :type compZoneRowEnd: int
    :param compZoneColStart: Start column of computational zone (inclusive)
    :type compZoneColStart: int
    :param compZoneColEnd: End column of computational zone (exclusive)
    :type compZoneColEnd: int
    :return: A list of moves to sort array or None if sorting has failed.
    :rtype: list[Move]
)pbdoc");

    m.def("sortParallel", &sortParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"))
        .doc() = "sortParallelTest\n";

    py::class_<ParallelMove::Step>(m, "ParallelMoveStep")
    .def(py::init())
    .def_readwrite("colSelection", &ParallelMove::Step::colSelection)
    .def_readwrite("rowSelection", &ParallelMove::Step::rowSelection);

    py::class_<ParallelMove>(m, "ParallelMove")
    .def(py::init())
    .def_readwrite("steps", &ParallelMove::steps);

    py::enum_<Direction>(m, "Direction")
    .value("HOR", Direction::HOR)
    .value("VER", Direction::VER)
    .value("NONE", Direction::NONE)
    .value("DIAG", Direction::DIAG)
    .export_values();

    py::class_<Move>(m, "Move")
    .def(py::init())
    .def_readwrite("sites_list", &Move::sites_list)
    .def_readwrite("distance", &Move::distance)
    .def_readwrite("init_dir", &Move::init_dir);

    py::class_<Config, std::unique_ptr<Config, py::nodelete>>(m, "Config")
    .def(py::init<>([]{return &Config::getInstance();}))
    .def_readwrite("logFileName", &Config::logFileName)
    .def_readwrite("sequentialLoggerName", &Config::sequentialLoggerName)
    .def_readwrite("parallelLoggerName", &Config::parallelLoggerName)
    .def("flushLogs", &Config::flushLogs);
};