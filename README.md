# pku-aip2023
course labs for ai-programming, 23summer@pku

no course page available




### Goal
Independently implement a simple CNN framework without using existing deep learning frameworks. Utilize CUDA, pybind11, and Python to achieve this.

### Steps & Files
1. **Cuda-based CNN:** Implement the convolutional neural network using Cuda, including forward and backward processes.
2. **pybind11 Integration:** Use pybind11 to create a Python callable .pyd file from the Cuda code.
3. **Python Integration:** Develop automatic differentiation and optimizer in Python to optimize the loss function and classify the MNIST dataset.


Before making, key codes are that:
```bash

$ tree
.
├── build
│   ├── autodiff.py
│   ├── basic_operator.py
│   ├── operators.py
│   └── optimizer.py
├── CMakeLists.txt
└── src
    ├── bind_Layer.cu
    ├── bind_Tensor.cu
    ├── global.h
    ├── Layer_kernels.h
    ├── Layer_kernels.inl
    ├── Layers.h
    ├── Layers.inl
    ├── Tensor.h
    ├── Tensor.inl
    ├── Tensor_kernels.h
    └── Tensor_kernels.inl
```
You should run the below commands to actually run the project.
```bash
git clone https://github.com/pybind/pybind11.git
mkdir build
cd build
cmake ..
make
# after that, you get xxx.cpython-38-x86_64-linux-gnu.so files.
# you should now switch to a conda env that supports pytorch & torchvision 
python optimizer.py # that would download MNIST dataset automatically
```



After making (with the help of `pybind` folders), you should get:
```bash
$ tree
.
├── CMakeLists.txt
├── build
│   ├── Makefile
│   ├── data
│   │   └── MNIST
│   │       └── ...
│   ├── myLayer.cpython-38-x86_64-linux-gnu.so
│   ├── myTensor.cpython-38-x86_64-linux-gnu.so
│   ├── autodiff.py
│   ├── basic_operator.py
│   ├── operators.py
│   ├── optimizer.py
│   ├── ...
├── pybind11
│   ├── ...
└── src
    ├── Layer_kernels.h
    ├── Layer_kernels.inl
    ├── Layers.h
    ├── Layers.inl
    ├── Tensor.h
    ├── Tensor.inl
    ├── Tensor_kernels.h
    ├── Tensor_kernels.inl
    ├── bind_Layer.cu
    ├── bind_Tensor.cu
    └── global.h
```



### Code Analysis
Codes in `src/` folder does the jobs of defining `myTensor` and `myLayers`. After binding, they provide the apis that are used in CNN networks. They've been previously partially written in `H01`, `H02` and binded in `H03`.

The python codes are divided in 4 files, the `autodiff.py`, `basic_operator.py`, `operators.py`, `optimizer.py`. They are a re-written with base type `myTensor` and base api `myLayers` of files in `H05` and `H06`. That keeps the `autodiff` ablity, while `SGD` and `Adam` optimizer have been inplemented.

In file `basic_operator.py`, the original class now be based on `myTensor`. Check:

```python
from myTensor import Tensor_Float as myTensor_f
from myTensor import Tensor_Int as myTensor_i
class Value:
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: myTensor_f # or myTensor_i
    requires_grad: bool

class Tensor(Value):
    grad: "Tensor"
# should know that the device be always 'GPU'
# and that dtype be merely only float32. dtype value unused
    def __init__(
        self,
        array,
        *,
        device='GPU',
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            # copy initialize deom a tensor
            # ...
        elif isinstance(array, myTensor_f):
            # initialize from myTensor_f
            # ...
        elif isinstance(array, myTensor_i):
            # initialize from myTensor_f
            # ...
        else:
            # simply raise error from that
            raise ValueError("error in Tensor init: from something wrong")

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )
```

In file `operators.py` a bunch of Ops have been inplemented based on the new_tensor. Note that we rewrite only those need in CNN later. Check:

```python
# ops. The below is all you need.
class EWiseAdd(TensorOp):
    # needed for __add__
    # ...
class Negate(TensorOp):
    # get negative. used to def __sub__
    # ...
class MulScalar(TensorOp):
    # multiply a scalar
    # if need multiplying between Tensors, use FC(use_bias=False) instead
    # used to def __mul__, and used in SGD and Adam
    # ...
class Sqrt(TensorOp):
    # sqrt of a Tensor, element-wise. 
    # used in Adam optimizer
    # ...
class Relu(TensorOp):
    # relu
    # ...
class Sigmoid(TensorOp):
    # sigmoid
    # ...
class FC(TensorOp):
    # fully-connected layers
    # you can set use_bias to control if there need a bias in FC layers
    # ...
class Conv(TensorOp):
    # convolution
    # ...
class Maxpool(TensorOp):
    # maxpooling
    # ...
class SftCrossEn(TensorOp):
    # softmax & cross entropy
    # ...
class Flat(TensorOp):
    # hold the Tensor flat, just as torch.view(-1) 
    # ...
```


In file `optimizer.py` the below have been integrated inside a python class
```python
# Step 2: define the CNN
class SimpleCNN():
    def __init__(self):
        # CNN parameters initialized
        self.weights = [self.conv1, self.conv2, self.fcw1, self.fcw2, self.fcb1, self.fcb2]
        # initialize the weights, and upload them in the optimizers.
        # ...

    def forward(self, x, real_labels):
        # forward method defined with TensorOps, which have been build upon myLayers
        # that's what makes a CNN
        x = maxpool(relu(conv(x, self.conv1)))
        x = maxpool(relu(conv(x, self.conv2)))
        x = flat(x)
        x = fc(fc(x, self.fcw1, self.fcb1), self.fcw2, self.fcb2)
        x = sftcrossen(x, real_labels)
        return x
    
    def SGD_epoch(self, X, y, lr=0.1, batch=100):
        # SGD inplemented
        # ...
        # when you need autograd, just do the below:
        tr_loss = self.forward(X_batch, y_batch)
        tr_loss.backward()

        # and then update
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - lr * self.weights[i].grad

    def Adam_epoch(self, X, y, lr=0.1, batch=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Adam optimizer inplemented
        # ...

    def train_nn(self, X_tr, y_tr, X_te, y_te, 
                 epochs=10, lr=0.5, batch=100, 
                 beta1=0.9, beta2=0.999, use_Adam=False):
        # almost the same as that in H06
        # ...
```
Just run `python optimizer.py` to get training outputs.

Outputs be like:
```bash
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|     0 |    0.21041 |   0.06142 |   0.21971 |  0.06560 |
|     1 |    0.15278 |   0.04572 |   0.15449 |  0.04760 |
|     2 |    0.13318 |   0.04130 |   0.14078 |  0.04470 |
|     3 |    0.12296 |   0.03733 |   0.14549 |  0.04280 |
|     4 |    0.12083 |   0.03788 |   0.14580 |  0.04470 |
|     5 |    0.08939 |   0.02732 |   0.12120 |  0.03530 |
|     6 |    0.10000 |   0.03217 |   0.12495 |  0.03780 |
|     7 |    0.08524 |   0.02645 |   0.13268 |  0.03930 |
|     8 |    0.10154 |   0.03337 |   0.14009 |  0.03800 |
|     9 |    0.08216 |   0.02562 |   0.13534 |  0.03820 |
|    10 |    0.06083 |   0.01920 |   0.11416 |  0.03230 |
|    11 |    0.05998 |   0.01970 |   0.12281 |  0.03320 |
|    12 |    0.05689 |   0.01882 |   0.12236 |  0.03380 |
|    13 |    0.05597 |   0.01843 |   0.12001 |  0.03130 |
|    14 |    0.06200 |   0.02067 |   0.13465 |  0.03540 |
|    15 |    0.05232 |   0.01740 |   0.12048 |  0.03020 |
|    16 |    0.03867 |   0.01263 |   0.11686 |  0.02760 |
|    17 |    0.04098 |   0.01338 |   0.11175 |  0.02930 |
|    18 |    0.04558 |   0.01523 |   0.13921 |  0.03310 |
|    19 |    0.05883 |   0.01967 |   0.13714 |  0.03420 |
```

That result provides a typical result based on H06 structure, which proves that our strcture is able for CNN training and evaluating.

And the above is how the task is completed.

