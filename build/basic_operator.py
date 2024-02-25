"""
一个基础的运算符类Op
和一个表示计算图中的节点数据的类Value
但是Op类里面的运算符都是实现好的myLayers
"""
# import numpy as np
from typing import List, Optional, Tuple, Union

from myTensor import Tensor_Float as myTensor_f

TENSOR_COUNTER = 0

class Op:
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[myTensor_f]):
        """
        前向过程的虚函数
        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """
        计算梯度的虚函数
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """
        使得计算的梯度总是以tuple的形式返回
        """
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)
        


class Value:
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: myTensor_f
    requires_grad: bool

    def realize_cached_data(self):
        """
        使用cached_data防止动态计算时重复计算
        """
        if self.cached_data is not None:
            return self.cached_data

        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        # print(f"cached_data created with type {type(self.cached_data)} {self.cached_data}")
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op,
        inputs,
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            # print(f"@basic_operator input is {input}")
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not value.requires_grad:
            return value.detach()
        value.realize_cached_data()
        return value

