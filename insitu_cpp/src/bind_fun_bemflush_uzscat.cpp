#include "bind_fun_bemflush_uzscat.h"

void bind_bemflush_uzscat(py::module &m)
{
    m.def("_bemflush_uzscat", bemflush_uzscat,
    "Calculate scattered particle velocity (zdir) at field point",
    py::arg("r_coord"),
    py::arg("node_x").noconvert(),
    py::arg("node_y").noconvert(),
    py::arg("Nzeta").noconvert(),
    py::arg("Nweights").noconvert(),
    py::arg("k0"), py::arg("beta"),
    py::arg("ps").noconvert()
    );
}