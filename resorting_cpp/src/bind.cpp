#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sortSequentiallyByRow.hpp"

namespace py = pybind11;

PYBIND11_MODULE(resorting_cpp, m) {
    m.doc() = "pybind11 resorting module";

    m.def("sortSequentiallyByRow", &sortSequentiallyByRow, "A function that sorts an array of atoms row by row", py::arg("stateArray"), py::arg("filledShape"));

    py::class_<Move>(m, "Move")
    .def(py::init())
    .def_readwrite("x", &Move::x)
    .def_readwrite("y", &Move::y)
    .def_readwrite("xDir", &Move::xDir)
    .def_readwrite("yDir", &Move::yDir);
};