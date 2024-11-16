import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import queue
import copy

from ...schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ...schema.protobuf.et_def_pb2 import Node as ChakraNode
from ..third_party.utils.protolib import encodeMessage as encode_message
from ..converter.pytorch_node import PyTorchNode, PyTorchNodeType, PyTorchTensor
from ..converter.pytorch_converter import PyTorchConverter


class TensorNode:
    """
    Represents a tensor node in a PyTorch execution trace, initialized based on each tensor generation.
    Attributes:

    id (int):               Identifier of the node.
    extra_id(int):          The Identifier of the node[id]'s copy. (0 means the origin node)
    value(PyTorchTensor):   The value of the tensor node which is initialized by PyTorchTensor
                            with tensor_data (List[int]): Data of the tensor including tensor_id, storage_id, offset, number of elements, and
                            size of each element in bytes.
    shape(List[int]):       The shape of the tensor.
    type(str):              The type of the tensor in chakra node.
    son(List[int]):         The list of future chakra node will use this node as input to operation.
    parent(int):            The chakra node that generate this node.
    """

    def __init__(self, id, extra_id, value, shape, type):
        self.id = id
        self.extra_id = extra_id
        self.value = PyTorchTensor(value)
        self.shape = shape
        self.type = type
        self.son = []
        self.parent = -1

    def add_son(self, x):
        """
        Add a son x, which means this node will be use by x as input in the future.
        """
        self.son.append(x)

    def set_parent(self, x):
        """
        Set parent, which represents the chakra node generate this node.
        """
        self.parent = x


class TraceMap:
    def __init__(
        self, json_metadata: Dict, json_node_map: Dict[int, PyTorchNode]
    ) -> None:
        self.metadata = json_metadata
        self.node_map = json_node_map
        self.oper_tot = json_node_map.keys().max() + 1
        self.id_count = defaultdict(int)
        self.roots = self.parse_root()
        self.relabel_tensor()
        self.rebuild_map()
        self.tensor_node: Dict[int, TensorNode] = {}

    def relabel_tensor(self):
        self.tensor_count = 0
        self.tensor_trans = defaultdict(lambda: -1)
        self.node_inputs = defaultdict(list)
        self.node_outputs = defaultdict(list)
        for id, node in self.node_map.items():
            inputs = []
            outputs = []
            node.extra_node = []
            if not PyTorchConverter().is_root_node(self.node_map[node.parent].name):
                self.ignore = True
                continue
            for input_value, input_shape, input_type in zip(
                node.inputs["values"], node.inputs["shapes"], node.input["types"]
            ):
                if "Tensor" in input_type:
                    tensor = PyTorchTensor(input_value)
                    tensor = (tensor.tensor_id, tensor.storage_id)
                    if self.tensor_trans[tensor] == -1:
                        self.tensor_trans[tensor] = self.tensor_count
                        self.tensor_node[self.tensor_count] = TensorNode(
                            self.tensor_count, 0, input_value, input_shape, input_type
                        )
                        self.tensor_count += 1
                    inputs.append(self.tensor_trans[tensor])
            for output_value, output_shape, output_type in zip(
                node.outpus["values"], node.outputs["shapes"], node.output["types"]
            ):
                if "Tensor" in output_type:
                    tensor = PyTorchTensor(output_value)
                    tensor = (tensor.tensor_id, tensor.storage_id)
                    self.tensor_trans[tensor] = self.tensor_count
                    self.tensor_node[self.tensor_count] = TensorNode(
                        self.tensor_count, 0, output_value, output_shape, output_type
                    )
                    self.tensor_count += 1
                    outputs.append(self.tensor_trans[tensor])
            self.node_inputs[id] = inputs
            self.node_outputs[id] = outputs

    def rebuild_map(self):
        for id, node in self.node_map.items():
            if self.ignore:
                self.node_map[self.node_map[node.parent].name].extra_node.append(id)
                continue
            self.node_inputs[id].sort()
            self.node_outputs[id].sort()
            if "alltoall" in node.name:
                continue
            # we should ignore the node alltoall which will be process later.
            for x in self.node_inputs[id]:
                self.tensor_node[x].add_son(id)
            for x in self.node_outputs[id]:
                self.tensor_node[x].set_parent(id)

    def new_copytensor(self, time, copy_a, copy_b):
        if self.tensor_copy_map[(copy_a, copy_b)]:
            return self.tensor_copy_map[(copy_a, copy_b)]
        self.tensor_copy_map[(copy_a, copy_b)] = self.tensor_count
        self.extend_list.push((time, 2, self.tensor_count, copy_a, copy_b))
        self.tensor_count += 1
        return self.tensor_count - 1

    def new_copyoperation(self, copy_a, copy_b):
        if self.operation_copy_map[(copy_a, copy_b)]:
            return self.operation_copy_map[(copy_a, copy_b)]
        self.operation_copy_map[(copy_a, copy_b)] = self.oper_tot
        time = max(self.node_map[copy_a].id, self.node_map[copy_b].id)
        self.extend_list.push((time, 1, self.oper_tot, copy_a, copy_b))
        self.oper_tot += 1
        return self.oepr_tot - 1

    def extend_front(self, x):
        if x[1] == 0:  # all to all extend
            mid = self.node_outputs[x[2]].size()
            new_input1 = self.new_copytensor(
                x[0], self.node_inputs[x[2]][0], self.node_inputs[x[2]][1]
            )
            new_input2 = self.new_copytensor(
                x[0], self.node_inputs[x[2]][mid], self.node_inputs[x[2]][mid + 1]
            )
            new_output = self.new_copytensor(
                x[0], self.node_outputs[x[2]][0], self.node_outputs[x[2]][1]
            )
            # 2 present tensor explan
            # which means that at least 2 GPU can infer the behavior of the tenso generation.
            self.node_inputs[x[2]].insert(mid, new_input1)
            self.node_inputs[x[2]].append(new_input2)
            self.node_outputs[x[2]].append(new_output)
            # output will be different from input
        elif x[1] == 1:  # operation extend
            pass
        else:  # tensor extend
            # copy all info
            _, _, name, copy_a, copy_b = x
            op_a, op_b = (
                self.tensor_node[copy_a].parent,
                self.tensor_node[copy_b].parent,
            )
            self.tensor_node[name] = copy.deepcopy(copy_a)
            # copy generation process
            self.tensor_node[name].parent = self.new_copyoperation(op_a, op_b)
            # generate son process
            self.tensor_node[name].son = [
                self.new_copyoperation(a, b)
                for (a, b) in zip(
                    self.tensor_node[copy_a].son, self.tensor_node[copy_b].son
                )
            ]

    def extend_back(self, x):
        pass

    def add_GPU_samegroup(self):
        self.extend_list = queue.PriorityQueue()
        self.back = queue.PriorityQueue()
        self.tensor_copy_map = defaultdict(int)
        self.operation_copy_map = defaultdict(int)
        for id, node in self.node_map.item():
            if node.ignore:
                continue
            if "alltoall" in node.name:
                self.extend_list.push(
                    (id, 0, id)
                )  # 0 present alltoall extend operation
        while not self.extend_list.empty():
            x = self.extend_list.top()
            self.extend_list.pop()
            self.extend_list.push(x)

    def output(self, filename):
        pass


class PyTorchExtender:
    """
    Some GPU trace + cluster info ->(pipeline / data / model / tensor parallelism)-> whole info.
    Current stage: only consider tensor / model para.
    """

    def extend(self, id: List[int], input_filenames: List[str]):
        """ """
        trace_maps = []
        for file in input_filenames:
            json_trace = PyTorchConverter().load_json_execution_traces(file)
            json_metadata, json_node_map = PyTorchConverter().parse_json_trace(
                json_trace
            )
            trace_maps.append(
                TraceMap(
                    json_metadata,
                    json_node_map,
                )
            )
        for x in trace_maps:
            x.add_GPU_samegroup()
        for map, filename in zip(trace_maps, input_filenames):
            map.output("ext_" + filename)
