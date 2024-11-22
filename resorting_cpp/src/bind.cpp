#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sortSequentiallyByRow.hpp"
#include "sortParallel.hpp"
#include "config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "pybind11 resorting module";

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"));

    m.def("sortParallel", &sortParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"));

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
    .def_readwrite("parallelLoggerName", &Config::parallelLoggerName);
};