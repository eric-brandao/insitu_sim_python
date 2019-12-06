#include "bind_fun_bemflush_pscat.h"

void bind_bemflush_pscat(py::module &m)
{
    m.def("_bemflush_pscat", bemflush_pscat,
    "Calculate scattered pressure at field point",
    py::arg("r_coord"),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert(),
    py::arg("Nweights").noconvert(),
    py::arg("k0"), py::arg("beta"),
    py::arg("ps").noconvert()
    );
}