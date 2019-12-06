#include "bind_mod_main.h"
PYBIND11_MODULE(insitu_cpp, m)
{
    //bind_compute(m);
    bind_doc(m);
    bind_bemflush_rmtx(m);
    bind_bemflush_mtx2(m);
    bind_bemflush_mtx(m);
    bind_bemflush_pscat(m);
    // bind_cls_sourcecpp(m);
    // bind_cls_receivercpp(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
