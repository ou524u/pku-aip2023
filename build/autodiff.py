"""
auto-diffusion
"""

from typing import List, Dict, Tuple
from basic_operator import Op, Value

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    给定一个节点列表，返回以这些节点结束的拓扑排序列表。
    一种简单的算法是对给定的节点进行后序深度优先搜索（DFS）遍历，
    根据输入边向后遍历。由于一个节点是在其所有前驱节点遍历后才被添加到排序中的，
    因此我们得到了一个拓扑排序。
    """

    visited = set()
    topo_sort = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_sort)
    return topo_sort
    

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""

    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
    

def compute_gradient_of_variables(output_tensor, out_grad):
    """
    对输出节点相对于 node_list 中的每个节点求梯度。
    将计算结果存储在每个 Variable 的 grad 字段中。
    """
    # map for 从节点到每个输出节点的梯度贡献列表
    node_to_output_grads_list = {}
    # 我们实际上是在对标量 reduce_sum(output_node) 
    # 而非向量 output_node 取导数。
    # 但这是损失函数的常见情况。
    node_to_output_grads_list[output_tensor] = [out_grad]
    
    # 根据我们要对其求梯度的 output_node，以逆拓扑排序遍历图。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order: 
        autodiff_joints = node_to_output_grads_list[node]
        # print(autodiff_joints, type(autodiff_joints))

        v_i = autodiff_joints[0]
        for k in range(len(autodiff_joints)):
            if k == 0:
                continue
            v_i = v_i + autodiff_joints[i]


        # v_i = sum(autodiff_joints)
        node.grad = v_i 

        if node.op is None: 
            continue 
        node_grad_list = node.op.gradient_as_tuple(v_i, node)
        for node_input, node_grad in zip(node.inputs, node_grad_list): 
            node_to_output_grads_list.setdefault(node_input, list())
            node_to_output_grads_list[node_input].append(node_grad)
    
    
