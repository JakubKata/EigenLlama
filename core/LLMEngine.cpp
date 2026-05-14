#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>

namespace py = pybind11;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

Matrix matmul(const Matrix& x, const Matrix& weight) {
    return x * weight; 
}

Matrix rmsnorm(const Matrix& x, const Vector& weight, float eps = 1e-5f) {
    Matrix out(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); ++i) {
        float squared_sum = x.row(i).squaredNorm(); 
        float mean_squared = squared_sum / x.cols();
        float rsqrt = 1.0f / std::sqrt(mean_squared + eps);
        
        out.row(i) = (x.row(i) * rsqrt).cwiseProduct(weight.transpose());
    }
    return out;
}

PYBIND11_MODULE(tiny_math, m) {
    m.def("matmul", &matmul, "matmul for TinyLlama");
    m.def("rmsnorm", &rmsnorm, "LLaMA RMS Normalization", 
        py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f);
}