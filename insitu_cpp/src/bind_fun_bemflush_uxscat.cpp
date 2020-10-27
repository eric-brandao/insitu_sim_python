#include "bind_fun_bemflush_uxscat.h"

void bind_bemflush_uxscat(py::module &m)
{
    m.def("_bemflush_uxscat", bemflush_uxscat,
    "Calculate scattered particle velocity (xdir) at field point",
    py::arg("r_coord"),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert(),
    py::arg("Nweights").noconvert(),
    py::arg("k0"), py::arg("beta"),
    py::arg("ps").noconvert()
    );
}