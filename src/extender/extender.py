import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple
from collections import defaultdict

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


class TraceMap:
    def __init__(
        self, json_metadata: Dict, json_node_map: Dict[int, PyTorchNode]
    ) -> None:
        self.metadata = json_metadata
        self.node_map = json_node_map
        self.roots = self.parse_root()
        self.relabel_tensor()
        self.rebuild_map()

    def relabel_tensor(self):
        self.tensor_count = 0
        self.tensor_trans = defaultdict(int)
        self.node_inputs = defaultdict(list)
        self.node_outputs = defaultdict(list)
        for id, node in self.node_map.items():
            inputs = []
            outputs = []
            node.extra_node = []
            if not PyTorchConverter().is_root_node(self.node_map[node.parent].name):
                continue
            for input_value, input_type in zip(
                node.inputs["values"], node.inputs["shapes"]
            ):
                if "Tensor" in input_type:
                    tensor = PyTorchTensor(input_value)
                    tensor = (tensor.tensor_id, tensor.storage_id)
                    if self.tensor_trans[tensor] == 0:
                        self.tensor_count += 1
                        self.tensor_trans[tensor] = self.tensor_count
                    inputs.append(self.tensor_trans[tensor])
            for output_value, output_type in zip(
                node.outpus["values"], node.outputs["shapes"]
            ):
                if "Tensor" in output_type:
                    tensor = PyTorchTensor(output_value)
                    tensor = (tensor.tensor_id, tensor.storage_id)
                    self.tensor_count += 1
                    self.tensor_trans[tensor] = self.tensor_count
                    outputs.append(self.tensor_trans[tensor])
            self.node_inputs[id] = inputs
            self.node_outputs[id] = outputs

    def rebuild_map(self):
        pass

    def add_GPU_samegroup(self):
        pass

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
