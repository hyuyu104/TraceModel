#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

using namespace std;
namespace py = pybind11;

py::array_t<double> scaled_forward(
    const py::array_t<double> density,
    const py::array_t<double> initdist,
    const py::array_t<double> transmat,
    int N, int T, int S
) {
    // unchecked direct access to numpy arrays
    auto den = density.unchecked<3>();
    auto mu = initdist.unchecked<1>();
    auto P = transmat.unchecked<2>();

    // forward array of shape N x T x (S+1)
    // extra position to store row sum/scaling factor
    py::array_t<double> farr({N, T, S+1});
    auto f = farr.mutable_unchecked<3>();

    for (int n = 0; n < N; ++n) {
        for (int t = 0; t < T; ++t) {
            double row_sum = 0.0;
            for (int s = 0; s < S; ++s) {
                if (t == 0) {
                    f(n, t, s) = mu(s)*den(s, n, t);
                } else {
                    double state_sum = 0.0;
                    for (int sp = 0; sp < S; ++sp) {
                        state_sum += f(n, t-1, sp)*P(sp, s);
                    }
                    f(n, t, s) = den(s, n, t)*state_sum;
                }
                row_sum += f(n, t, s);
            }
            f(n, t, S) = row_sum;
            // Rescale each row to avoid underflow
            for (int s = 0; s < S; ++s) {
                f(n, t, s) /= row_sum;
            }
        }
    }
    return farr;
}

py::array_t<double> scaled_backward(
    const py::array_t<double> density,
    const py::array_t<double> transmat,
    int N, int T, int S
) {
    // unchecked direct access to numpy arrays
    auto den = density.unchecked<3>();
    auto P = transmat.unchecked<2>();

    // forward array of shape N x T x S
    py::array_t<double> barr({N, T, S});
    auto b = barr.mutable_unchecked<3>();

    for (int n = 0; n < N; ++n) {
        for (int t = T-1; t > -1; --t) {
            double row_sum = 0.0;
            for (int s = 0; s < S; ++s) {
                if (t == T-1) {
                    b(n, t, s) = 1;
                } else {
                    double state_sum = 0.0;
                    for (int sp = 0; sp < S; ++sp) {
                        state_sum += b(n, t+1, sp)*P(s, sp)*den(sp, n, t+1);
                    }
                    b(n, t, s) = state_sum;
                }
                row_sum += b(n, t, s);
            }

            for (int s = 0; s < S; ++s) {
                b(n, t, s) /= row_sum;
            }
        }
    }
    return barr;
}


py::array_t<double> calculate_v(
    const py::array_t<double> density,
    const py::array_t<double> fval,
    const py::array_t<double> bval,
    const py::array_t<double> alphaval,
    const py::array_t<double> transmat,
    int N, int T, int S
) {
    auto den = density.unchecked<3>(); // N x T x S
    auto P = transmat.unchecked<2>(); // S x S
    auto f = fval.unchecked<3>(); // N x T x S
    auto b = bval.unchecked<3>(); // N x T x S
    auto alpha = alphaval.unchecked<2>(); // N x T

    // v of shape N x T x S x S
    py::array_t<double> vval({N, T, S, S});
    auto v = vval.mutable_unchecked<4>();

    for (int t = 1; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double denominator = .0;
            for (int s = 0; s < S; ++s) {
                denominator += f(n, t, s)*b(n, t, s);
            }
            // the last term was not canceled
            denominator *= alpha(n, t);
            double numerator;
            for (int s1 = 0; s1 < S; ++s1) {
                for (int s2 = 0; s2 < S; ++s2) {
                    numerator = f(n, t-1, s1)*P(s1, s2)*den(s2, n, t)*b(n, t, s2);
                    v(n, t, s1, s2) = numerator/denominator;
                }
            }
        }
    }
    return vval;
}


PYBIND11_MODULE(update, m) {
    m.def("scaled_forward", &scaled_forward, "Scaled forward algorithm.");
    m.def("scaled_backward", &scaled_backward, "Scaled backward algorithm.");
    m.def("calculate_v", &calculate_v, "Calculate conditional expectation (v).");
}