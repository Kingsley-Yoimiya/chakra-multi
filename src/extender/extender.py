import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import queue

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
            for x in self.node_inputs[id]:
                self.tensor_node[x].add_son(id)
            for x in self.node_outputs[id]:
                self.tensor_node[x].set_parent(id)

    def extend_front(self, front, back, tensor_copy_map, x):
        pass

    def extend_back(self, front, back, tensor_copy_map, x):
        pass

    def add_GPU_samegroup(self):
        front = queue.PriorityQueue()
        back = queue.PriorityQueue()
        tensor_copy_map = defaultdict(int)
        for id, node in self.node_map.item():
            if node.ignore:
                continue
            if "alltoall" in node.name:
                front.push((id, 0))  # 0 present operation
                back.push((id, 0))
        while not front.empty() and not back.empty():
            if not front.empty():
                x = front.top()
                front.pop()
                self.extend_front(front, back, tensor_copy_map, x)
            else:
                x = back.top()
                back.pop()
                self.extend_back(front, back, tensor_copy_map, x)

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
