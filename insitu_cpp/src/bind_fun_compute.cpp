
#include "bind_fun_compute.h"

void bind_compute(py::module &m)
{
    m.def("compute", [](
        const int &imw, const int &imh,
        Eigen::Ref<Eigen::VectorXd> coefs,
        double crmin, double crmax,
        double cimin, double cimax,
        int itmax, float tol) -> py::array_t<double> {

        Eigen::VectorXd deriv_coefs(coefs.size()-1);
        for (int i=0; i<coefs.size()-1; ++i)
            deriv_coefs(i) = (coefs.size() - 1 - i) * coefs(i);
        coefs.reverseInPlace();
        deriv_coefs.reverseInPlace();

        Eigen::PolynomialSolver<double, Eigen::Dynamic> polysolver;
        polysolver.compute(coefs);
        Eigen::VectorXcd roots;
        roots = polysolver.roots();

        size_t size = imh * imw * 3;
        double *data = new double[size]{}; // <-- initialize with zeros
        py::capsule free_when_done(data, [](void *f) {
            double *data = reinterpret_cast<double *>(f);
            delete[] data;
        });

        auto result = py::array_t<
            double,
            py::array::c_style | py::array::forcecast>(
            {imh, imw, 3}, // shape
            {
                3 * imw * sizeof(double),
                3 * sizeof(double),
                sizeof(double)}, // C-style contiguous strides for double
            data,                // the data pointer
            free_when_done       // numpy array references this parent
        );
        auto res = result.mutable_unchecked<3>();

        for (int row = 0; row < imh; ++row)
            for (int col = 0; col < imw; ++col)
            {
                auto r = crmin + (crmax - crmin) / imw * col;
                auto i = cimax - (cimax - cimin) / imh * row;
                auto z = std::complex<double>(r, i);
                auto k = 0;
                while(k < itmax)
                {
                    auto f = poly_eval(coefs, z);
                    auto df = poly_eval(deriv_coefs, z);
                    if (df != 0.0)
                    {
                        z = z - f/df;

                        auto f_curr = poly_eval(coefs, z);
                        if(std::abs(f_curr) <= tol)
                        {
                            Eigen::VectorXd::Index root_idx=0;
                            (roots.array() - z).abs().minCoeff(&root_idx);
                            auto h = (double)root_idx / (double)roots.size();
                            auto s = (double)1;
                            auto v = 1.0 - (double)k / (double)itmax;
                            res(row, col, 0) = h;
                            res(row, col, 1) = s;
                            res(row, col, 2) = v;
                            break;
                        }
                    }

                    ++k;
                }
            }

        return result;
    },
    "Computes a newton basins image in HSV format and outputs a numpy.ndarray"
    " compatible object with shape (nrows: imh, ncols: imw, depth=3).",
    py::arg("imw"), py::arg("imh"),
    py::arg("coefs").noconvert(),
    py::arg("crmin"), py::arg("crmax"),
    py::arg("cimin"), py::arg("cimax"),
    py::arg("itmax"), py::arg("tol"));
}
