#include "bind_fun_bemflush_rmtx.h"

void bind_bemflush_rmtx(py::module &m)
{
    m.def("_bemflush_rmtx", bemflush_rmtx,
    "Assembles the BEM matrix coefficients",
    py::arg("el_center").noconvert(),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert()
    );
}