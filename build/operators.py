"""
依据所有mylayers里所给出的接口实现的Op
"""

from typing import List, Optional, Tuple, Union


from myTensor import Tensor_Float as myTensor_f
from myTensor import Tensor_Int as myTensor_i

import myLayer

from basic_operator import Op, Value
from autodiff import compute_gradient_of_variables



class Tensor(Value):
    grad: "Tensor"

# should know that the device be always 'GPU'
# and that dtype be only float32
    def __init__(
        self,
        array,
        *,
        device='GPU',
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        # copy initialize a tensor
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                raise ValueError("error in Tensor copy init")

        elif isinstance(array, myTensor_f):
            # initialize from myTensor_f
            cached_data = array

        elif isinstance(array, myTensor_i):
            # initialize from myTensor_f
            cached_data = array
        else:
            # Refuse to initialize myTensor_f from something else, and that should be a python list
            # device = 'GPU'
            # cached_data = Tensor._myTensor_from_list(array, device=device, dtype=dtype)
            raise ValueError("error in Tensor init: from list-or-something")


        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

        

    @staticmethod
    def _myTensor_from_list(list_data, device, dtype):
        return myTensor_f(list_data.shape, "GPU", list_data)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())
    
# should notice that ops include only those inplemented in myLayers
# including: fc, conv, maxpool, softmax-crossEn

    @property
    def shape(self):
        # print(f"@func shape: {type(self.realize_cached_data())} {self.realize_cached_data()}")
        return self.realize_cached_data().shape()


# need change here
    
    def backward(self, out_grad=None):

        ones = myTensor_f(self.shape, "GPU")
        ones.ones()
        if out_grad is None:
            print(f"@TensorBackward {self.op} has no grad, pushing ones backwards")
        out_grad = (out_grad if out_grad else Tensor(ones))
                    
        compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "\n<< Tensor on " + str(self.realize_cached_data()) + " >>\n"

    def __str__(self):
        return self.realize_cached_data().__str__()

    
    def myTensor_f(self):
        data = self.realize_cached_data()
        return data


# doesn't have that many default operators!
    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            raise ValueError("adding to not a Tensor")

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            raise ValueError("adding to not a Tensor")
        
    __rsub__ = __sub__
            
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return FC(use_bias=False)(self, other, Tensor(myTensor_f([1], "GPU")))
        else:
            return MulScalar(other)(self)
        

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            raise ValueError("diving to a Tensor")
        else:
            return MulScalar(1 / other)(self)
        

        
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return FC(use_bias=False)(self, other, Tensor(myTensor_f([1], "GPU")))
        else:
            raise ValueError("matmul to not a Tensor")



class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    

# ops. doesn't need so much
class EWiseAdd(TensorOp):
    def compute(self, a: myTensor_i, b: myTensor_i):
        return a + b

    def gradient(self, out_grad: Tensor, node):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)


class Negate(TensorOp):
    def compute(self, input: myTensor_f):
        output = myTensor_f(input.shape(), "GPU", input.data())
        output.negative()
        return output
        

    def gradient(self, grad_out_tensor: Tensor, node):
        grad_out = grad_out_tensor.realize_cached_data()
        grad_out.negative()
        return Tensor(grad_out)

def negate(input):
    return Negate()(input)

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, input: myTensor_f):
        output = myTensor_f(input.shape(), "GPU", input.data())
        output.mults(self.scalar)
        return output

    def gradient(self, grad_out_tensor: Tensor, node):
        grad_out = grad_out_tensor.realize_cached_data()
        grad_input = myTensor_f(grad_out.shape(), "GPU", grad_out.data())
        grad_input.mults(self.scalar)
        return Tensor(grad_input)


def mul_scalar(input, scalar):
    return MulScalar(scalar)(input)

class Sqrt(TensorOp):
    def compute(self, input: myTensor_f):
        output = input.sqrtForward()
        return output
    
    def gradient(self, grad_out_tensor: Tensor, node):
        # raise ValueError("sqrt calling Backwards")
        print(f"sqrt should not calling Backwards, do actually nothing.")
        return grad_out_tensor
    
def sqrt(input):
    return Sqrt()(input)

class Relu(TensorOp):
    def compute(self, input: myTensor_f):
        # print(f"@reluFoward input {input}")
        output = input.reluForward()
        # print(f"@reluFoward output {output}")
        return output
    
    def gradient(self, grad_out_tensor: Tensor, node):
        input_tensor, = node.inputs
        # print(f"reluBack input_tensor {input_tensor}")
        input = input_tensor.realize_cached_data()

        
        grad_out = grad_out_tensor.realize_cached_data()
        # print(f"@reluBack gradout {grad_out}")
        grad_input = input.reluBackward(grad_out)
        # print(f"@reluBack gradinput {grad_input}")
        return Tensor(grad_input)
    

def relu(input):
    return Relu()(input)


class Sigmoid(TensorOp):
    def compute(self, input: myTensor_f):
        output = input.sigmoidForward()
        return output
    
    def gradient(self, grad_out_tensor: Tensor, node):
        input_tensor, = node.inputs
        input = input_tensor.realize_cached_data()
        grad_out = grad_out_tensor.realize_cached_data()
        grad_input = input.sigmoidBackward(grad_out)
        
        return Tensor(grad_input)
    

def sigmoid(input):
    return Sigmoid()(input)


    
class FC(TensorOp):
    # void fc_forward(
    # const Tensor<TContent>& input,
    # Tensor<TContent>& output,
    # const Tensor<TContent>& weights,
    # const Tensor<TContent>& bias);
    def __init__(self, use_bias=True):
        self.use_bias = use_bias

    def compute(self, input: myTensor_f, weights: myTensor_f, bias:myTensor_f):
        output_shape = input.shape()[:-1] + weights.shape()[1:]

        print(f"@fcForward input shape {input.shape()}")
        print(f"@fcForward weights shape {weights.shape()}")

        output = myTensor_f(output_shape, "GPU")
        if not self.use_bias:
            bias.zeros()

        myLayer.fc_forward_Float(input, output, weights, bias)
        
        # print(f"@fcForward output shape {output.shape}")
        return output
        

    def gradient(self, grad_output_tensor: Tensor, node):

        input_tensor, weights_tensor, bias_tensor, = node.inputs

        input = input_tensor.realize_cached_data()
        weights = weights_tensor.realize_cached_data()
        bias = bias_tensor.realize_cached_data()
        
        grad_output = grad_output_tensor.realize_cached_data()
        empty_output = myTensor_f(grad_output.shape(), "GPU")

        grad_input = myTensor_f(input.shape(), "GPU")
        grad_weights = myTensor_f(weights.shape(), "GPU")
        grad_bias = myTensor_f(bias.shape(), "GPU")

        print(f"@fcBackward grad_output shape {grad_output.shape()}")
        print(f"@fcBackward weights shape {weights.shape()}")
        

        myLayer.fc_backward_Float(
            input, empty_output, weights, bias,
            grad_input, grad_output, grad_weights, grad_bias
        )

        if not self.use_bias:
            grad_bias.zeros()
      
        return Tensor(grad_input), Tensor(grad_weights), Tensor(grad_bias)
        
def fc(input, weights, bias=myTensor_f([1], "GPU"), use_bias=True):
    return FC(use_bias)(input, weights, bias)


class Conv(TensorOp):
    # void conv_forward(
    # const Tensor<TContent>& input,
    # Tensor<TContent>& output,
    # const Tensor<TContent>& weights,
    # const int pad_h,
    # const int pad_w,
    # const int stride_h,
    # const int stride_w,
    # const int dilation_h,
    # const int dilation_w);

    def __init__(self, pad_h=1, pad_w=1, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

    def compute(self, input: myTensor_f, weights: myTensor_f):
        output = myTensor_f(input.shape(), "GPU")
        # print(f"@func convForward, input is {input}")
        myLayer.conv_forward_Float(
            input, output, weights,
            self.pad_h, self.pad_w, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w
        )

        print(f"@convForward input shape {input.shape()}")
        print(f"@convForward weights shape {weights.shape()}")
        print(f"@convForward output shape {output.shape()}")
        
        return output
    
    def gradient(self, grad_output_tensor: Tensor, node):
        input_tensor, weights_tensor, = node.inputs

        input = input_tensor.realize_cached_data()
        weights = weights_tensor.realize_cached_data()
        grad_output = grad_output_tensor.realize_cached_data()

        empty_output = myTensor_f(grad_output.shape(), "GPU")

        grad_input = myTensor_f(input.shape(), "GPU")
        grad_weights = myTensor_f(weights.shape(), "GPU")
        myLayer.conv_backward_Float(
            input, empty_output, weights,
            grad_input, grad_output, grad_weights,
            self.pad_h, self.pad_w, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w
        )

        print(f"@convBackward grad_output shape {grad_output.shape()}")
        print(f"@convBackward weights shape {weights.shape()}")

        
        return Tensor(grad_input), Tensor(grad_weights)

    
def conv(input, weights, pad_h=1, pad_w=1, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    return Conv(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w)(input, weights)

class Maxpool(TensorOp):
    # void maxpool_forward(
    # const Tensor<TContent>& input,
    # Tensor<TContent>& output,
    # Tensor<TContent>& masks,
    # const std::vector<int>& kernel_shape,
    # const int pad_h,
    # const int pad_w,
    # const int stride_h,
    # const int stride_w);

    def __init__(self, kernel_shape=[2,2], pad_h=1, pad_w=1, stride_h=2, stride_w=2):
        self.kernel_shape = kernel_shape
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.masks = None
        return
    
    def compute(self, input: myTensor_f):
        output = myTensor_f(input.shape(), "GPU")
        # print(f"@maxpoolForward input shape {input.shape}")
        masks = myTensor_f(input.shape(), "GPU")
        myLayer.maxpool_forward_Float(
            input, output, masks,
            self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w
        )
        self.masks = masks
        # print(f"@maxpoolForward masks shape {self.masks.shape()}, output shape {output.shape()}")
        print(f"@maxpoolForward input shape {input.shape()}")
        print(f"@maxpoolForward output shape {output.shape()}")

        return output
    
    def gradient(self, grad_output_tensor: Tensor, node):
        input_tensor, = node.inputs

        input = input_tensor.realize_cached_data()
        grad_output = grad_output_tensor.realize_cached_data()

        grad_input = myTensor_f(input.shape(), "GPU")

        # print(f"@maxpoolBackward masks shape {self.masks.shape()}, grad_output shape {grad_output.shape()}")
        myLayer.maxpool_backward_Float(
            grad_output, self.masks, grad_input,
            self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w
        )
        return Tensor(grad_input)
    
def maxpool(input, kernel_shape=[2,2], pad_h=1, pad_w=1, stride_h=2, stride_w=2):
    return Maxpool(kernel_shape, pad_h, pad_w, stride_h, stride_w)(input)


# Ich bin hier! Junges Mädchen.

class SftCrossEn(TensorOp):

    # void softmax_forward(
    # const Tensor<TContent>& input,
    # Tensor<TContent>& output);

    # // change loss_tensor but not return loss as a value.
    # void crossEntropy_forward(
    # const Tensor<TContent>& input,
    # Tensor<TContent>& loss,
    # const Tensor<int>& real_labels);

    # void softmax_corssEntropy_backward(
    # const Tensor<TContent>& softmax_output,
    # Tensor<TContent>& grad_input,
    # const Tensor<int>& real_labels);

    def __init__(self, real_labels):
        if isinstance(real_labels, myTensor_i):
            self.real_labels = real_labels
        elif isinstance(real_labels, Tensor):
            self.real_labels = real_labels.realize_cached_data()
        else:
            self.real_labels = myTensor_i(len(real_labels), "GPU", real_labels)
        self.softmax_output = None

    def compute(self, input: myTensor_f):
        
        softmax_output = myTensor_f(input.shape(), "GPU")
        myLayer.softmax_forward_Float(input, softmax_output)
        self.softmax_output = softmax_output
        print(f"@sftcrossenForward softmax_output {type(softmax_output)}, real_label {type(self.real_labels)})")
        error = myLayer.softmax_loss_Float(softmax_output, self.real_labels)

        loss = myTensor_f([1], "GPU")
        loss.zeros()
        myLayer.crossEntropy_forward_Float(input, loss, self.real_labels)

        # # set-up the loss!
        # self.loss = loss
        return loss, error
    
    def gradient(self, node):

        grad_input = myTensor_f(self.softmax_output.shape(), "GPU")
        myLayer.softmax_crossEntropy_backward_Float(
            self.softmax_output, grad_input, self.real_labels
        )
        return Tensor(grad_input)

def sftcrossen(input, real_labels):
    return SftCrossEn(real_labels)(input)


class Flat(TensorOp):
    def compute(self, input: myTensor_f):
        self.shape_nonflat = input.shape()
        output = input.flat()
        
        return output
    
    def gradient(self, grad_out_tensor: Tensor, node):

        grad_out = grad_out_tensor.realize_cached_data()
        grad_input = grad_out.resize(self.shape_nonflat)

        return Tensor(grad_input)
    
def flat(input):
    return Flat()(input)


