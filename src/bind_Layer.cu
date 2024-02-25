#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Layers.h"

namespace py = pybind11;

// Wrap each function template with the PYBIND11_TEMPLATE macro
PYBIND11_MODULE(myLayer, m) {
    // Provide explicit bindings for float
    m.def("fc_forward_Float", &fc_forward<float>);
    m.def("fc_backward_Float", &fc_backward<float>);
    m.def("im2col_Float", &im2col<float>);
    m.def("col2im_Float", &col2im<float>);
    m.def("conv_forward_Float", &conv_forward<float>);
    m.def("conv_backward_Float", &conv_backward<float>);
    m.def("maxpool_forward_Float", &maxpool_forward<float>);
    m.def("maxpool_backward_Float", &maxpool_backward<float>);
    m.def("softmax_forward_Float", &softmax_forward<float>);
    m.def("softmax_loss_Float", &softmax_loss<float>);
    m.def("crossEntropy_forward_Float", &crossEntropy_forward<float>);
    m.def("softmax_crossEntropy_backward_Float", &softmax_corssEntropy_backward<float>);
}
