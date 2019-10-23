
#include "bind_mod_doc.h"

void bind_doc(py::module &m)
{
    m.doc() = R"pbdoc(Newton Basins (insitu_cpp) plugin)pbdoc";
}
