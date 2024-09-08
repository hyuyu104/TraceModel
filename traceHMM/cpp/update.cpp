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

PYBIND11_MODULE(update, m) {
    m.def("scaled_forward", &scaled_forward, "Scaled forward algorithm.");
    m.def("scaled_backward", &scaled_backward, "Scaled backward algorithm.");
}