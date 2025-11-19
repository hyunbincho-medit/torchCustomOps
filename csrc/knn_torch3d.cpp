#if __cplusplus < 201703L
#error "C++17 is required"
#endif

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "knn/knn.h"

#define STR_(x) #x
#define STR(x) STR_(x)

// Torch library에 custom ops 등록
TORCH_LIBRARY(knn_torch3d, m) {
    m.def("knn_points_idx(Tensor p1, Tensor p2, Tensor lengths1, Tensor lengths2, int norm, int K, int version) -> (Tensor, Tensor)");
    m.def("knn_points_backward(Tensor p1, Tensor p2, Tensor lengths1, Tensor lengths2, Tensor idx, int norm, Tensor grad_dists) -> (Tensor, Tensor)");
}

// CPU 구현 등록
TORCH_LIBRARY_IMPL(knn_torch3d, CPU, m) {
    m.impl("knn_points_idx", &KNearestNeighborIdx);
    m.impl("knn_points_backward", &KNearestNeighborBackward);
}

// CUDA 구현 등록 (나중에 추가 가능)
// TORCH_LIBRARY_IMPL(knn_torch3d, CUDA, m) {
//     m.impl("knn_points_idx", &KNearestNeighborIdxCuda);
//     m.impl("knn_points_backward", &KNearestNeighborBackwardCuda);
// }

// Autograd 지원
TORCH_LIBRARY_IMPL(knn_torch3d, Autograd, m) {
    m.impl("knn_points_idx", torch::autograd::autogradNotImplementedFallback());
    m.impl("knn_points_backward", torch::autograd::autogradNotImplementedFallback());
}

// Pybind11 모듈 바인딩
PYBIND11_MODULE(_core, m) {
    m.doc() = "KNN operations for PyTorch";
    
    m.attr("CPP_VERSION") = __cplusplus;
    m.attr("PYTORCH_VERSION") = STR(TORCH_VERSION_MAJOR) "." STR(
        TORCH_VERSION_MINOR) "." STR(TORCH_VERSION_PATCH);
    m.attr("PYBIND11_VERSION") = STR(PYBIND11_VERSION_MAJOR) "." STR(
        PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);
    
    // 버전 체크 함수도 바인딩
    m.def("knn_check_version", &KnnCheckVersion, "Check if KNN version is supported");
}