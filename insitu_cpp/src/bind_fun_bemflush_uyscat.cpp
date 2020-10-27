#include "bind_fun_bemflush_uyscat.h"

void bind_bemflush_uyscat(py::module &m)
{
    m.def("_bemflush_uyscat", bemflush_uyscat,
    "Calculate scattered particle velocity (ydir) at field point",
    py::arg("r_coord"),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert(),
    py::arg("Nweights").noconvert(),
    py::arg("k0"), py::arg("beta"),
    py::arg("ps").noconvert()
    );
}