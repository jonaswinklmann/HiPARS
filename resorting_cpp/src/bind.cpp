#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sortSequentiallyByRow.hpp"
#include "sortParallel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "pybind11 resorting module";

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), py::arg("filledShape"));

    m.def("sortParallel", &sortParallel, "A function that sorts an array of atoms in parallel", py::arg("stateArray"), 
        py::arg("compZoneRowStart"), py::arg("compZoneRowEnd"), py::arg("compZoneColStart"), py::arg("compZoneColEnd"));

    py::class_<ParallelMove::Step>(m, "ParallelMoveStep")
    .def(py::init())
    .def_readwrite("colSelection", &ParallelMove::Step::colSelection)
    .def_readwrite("rowSelection", &ParallelMove::Step::rowSelection);

    py::class_<ParallelMove>(m, "ParallelMove")
    .def(py::init())
    .def_readwrite("steps", &ParallelMove::steps);

    py::class_<Move>(m, "Move")
    .def(py::init())
    .def_readwrite("x", &Move::x)
    .def_readwrite("y", &Move::y)
    .def_readwrite("xDir", &Move::xDir)
    .def_readwrite("yDir", &Move::yDir);
};