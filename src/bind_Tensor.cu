#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "Tensor.h"

namespace py = pybind11;

template <typename TContent>
void bind_Tensor(py::module& m, const std::string& suffix) {
    using TensorType = Tensor<TContent>;
    std::string class_name = "Tensor_" + suffix;

    py::class_<TensorType>(m, class_name.c_str())
        // .def_readonly("data", &TensorType::data)
        // .def_readonly("shape", &TensorType::shape)
        // .def_readonly("device", &TensorType::device)
        // .def_readonly("size", &TensorType::size)
        // .def_property(
        //     "data",
        //     [](const TensorType& tensor) { return tensor.data; },
        //     [](TensorType& tensor, TContent* data) { tensor.data = data; })
        // .def_property(
        //     "shape",
        //     [](const TensorType& tensor) { return tensor.shape; },
        //     [](TensorType& tensor, const std::vector<int>& shape) { tensor.shape = shape; })
        // .def_property(
        //     "device",
        //     [](const TensorType& tensor) { return tensor.device; },
        //     [](TensorType& tensor, const std::string& device) { tensor.device = device; })
        // .def_property(
        //     "size",
        //     [](const TensorType& tensor) { return tensor.size; },
        //     [](TensorType& tensor, size_t size) { tensor.size = size; })

        .def(py::init<const std::vector<int>&, std::string>())
        .def(py::init<const std::vector<int>&, std::string, const std::vector<TContent>&>())
        .def("cpu", &TensorType::cpu)
        .def("gpu", &TensorType::gpu)
        .def("__add__", &TensorType::operator+, py::is_operator())
        .def("__sub__", &TensorType::operator-, py::is_operator())
        .def("random", &TensorType::random)
        .def("ones", &TensorType::ones)
        .def("floatKs", &TensorType::floatKs)
        .def("zeros", &TensorType::zeros)
        .def("negative", &TensorType::negative)

        .def("mults", &TensorType::mults)

        .def("resize", &TensorType::resize)
        .def("reluForward", &TensorType::reluForward)
        .def("reluBackward", &TensorType::reluBackward)
        .def("sigmoidForward", &TensorType::sigmoidForward)
        .def("sigmoidBackward", &TensorType::sigmoidBackward)
        .def("__repr__", [](const TensorType& tensor) {
            std::ostringstream oss;
            oss << tensor;
            std::string repr = oss.str();
            return repr;
        })

        .def("__getitem__", [](const TensorType& tensor, const int k) {
            return tensor[k];
        })

        .def("shape", &TensorType::getshape, py::return_value_policy::copy)  // Binding for getshape()
        .def("data", &TensorType::getdata, py::return_value_policy::copy)
        .def("device", &TensorType::getdevice, py::return_value_policy::copy)  // Binding for getshape()
        .def("flat", &TensorType::getflat);
}

PYBIND11_MODULE(myTensor, m) {
    m.doc() = "Tensor module";
    bind_Tensor<float>(m, "Float");
    // Add bindings for other data types if needed
    bind_Tensor<double>(m, "Double");
    bind_Tensor<int>(m, "Int");
}
