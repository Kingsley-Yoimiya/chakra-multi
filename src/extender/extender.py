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
from tensor_node import TensorNode
from operation_node import OperationNode


class TraceMap:
    """
    We will have several chakra trace, and each trace will be processed as TraceMap.

    Attributes:
        metadata(Any):                                  The metadata from the chakra trace.
        oper_tot(int):                                  If we want a new operation node, the new id will be oper_tot.
        id_count(Dict[int]):                            Each id's counter number.
        operation_node(Dict[int, OperationNode]):       All of operation node we use in this Chakra trace.
        tensor_node(Dict[int, TensorNode]):             All tensor node we will relabel and use in this trace.
        tensor_count(int):                              If we want a new tensor copy from a and b, the new tensor id will be tensor_count.
        tensor_trans(Dict[Tuple[int, int], int]):       The dict we will use to relabel the tensors.
        extend_list(Queue.PriorityQueue()):             The nodes we need to copy(0: all2all, 1: opeation, 2: tensor).
        tensor_copy_map(defaultdict(int)):              The map((int, int) -> (int), (copy_a, copy_b) -> (gener_tensor_node)) that we use to check if we have copy this tensor.
        operation_copy_map(defaultdict(int)):           The map((int, int) -> (int), (copy_a, copy_b) -> (gener_operation_node)) that we use to check if we have copy this operation.
    """

    def __init__(
        self, json_metadata: Dict, json_node_map: Dict[int, PyTorchNode]
    ) -> None:
        self.metadata = json_metadata
        self.oper_tot = json_node_map.keys().max() + 1
        self.id_count = defaultdict(int)
        self.operation_node = {
            node_id: OperationNode(old_node)
            for node_id, old_node in json_node_map.items()
        }
        self.relabel_tensor()
        self.rebuild_map()
        self.tensor_node: Dict[int, TensorNode] = {}

    def relabel_tensor(self) -> None:
        self.tensor_count = 0
        self.tensor_trans = defaultdict(lambda: -1)
        for _, node in self.operation_node.items():
            if not PyTorchConverter().is_root_node(
                self.operation_node[node.parent].name
            ):
                node.ignore = True
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
                            self.tensor_count, input_value, input_shape, input_type
                        )
                        self.tensor_count += 1
                    node.input_ids.append(self.tensor_trans[tensor])
            for output_value, output_shape, output_type in zip(
                node.outputs["values"], node.outputs["shapes"], node.outputs["types"]
            ):
                if "Tensor" in output_type:
                    tensor = PyTorchTensor(output_value)
                    tensor = (tensor.tensor_id, tensor.storage_id)
                    self.tensor_trans[tensor] = self.tensor_count
                    self.tensor_node[self.tensor_count] = TensorNode(
                        self.tensor_count, output_value, output_shape, output_type
                    )
                    self.tensor_count += 1
                    node.output_ids.append(self.tensor_trans[tensor])

    def rebuild_map(self) -> None:
        for id, node in self.operation_node.items():
            if node.ignore:
                self.operation_node[
                    self.operation_node[node.parent].name
                ].extra_node.append(id)
                continue
            if "alltoall" in node.name:
                continue
            # we should ignore the node alltoall which will be process later.
            for x in node.input_ids:
                self.tensor_node[x].add_son(id)
            for x in node.output_ids:
                self.tensor_node[x].set_parent(id)

    def new_copytensor(self, time: int, copy_a: int, copy_b: int) -> int:
        if self.tensor_copy_map[(copy_a, copy_b)]:
            return self.tensor_copy_map[(copy_a, copy_b)]
        self.tensor_copy_map[(copy_a, copy_b)] = self.tensor_count
        self.extend_list.put((time, 2, self.tensor_count, copy_a, copy_b))
        self.tensor_count += 1
        return self.tensor_count - 1

    def new_copyoperation(self, copy_a: int, copy_b: int) -> int:
        if self.operation_copy_map[(copy_a, copy_b)]:
            return self.operation_copy_map[(copy_a, copy_b)]
        self.operation_copy_map[(copy_a, copy_b)] = self.oper_tot
        time = max(self.operation_node[copy_a].id, self.operation_node[copy_b].id)
        self.extend_list.put((time, 1, self.oper_tot, copy_a, copy_b))
        self.oper_tot += 1
        return self.oper_tot - 1

    def extend(self, x: Tuple) -> None:
        if x[1] == 0:  # all to all extend
            node = self.operation_node[x[2]]
            mid = len(node.output_ids)
            new_input1 = self.new_copytensor(x[0], node.input_ids[0], node.input_ids[1])
            new_input2 = self.new_copytensor(
                x[0], node.input_ids[mid], node.output_ids[mid + 1]
            )
            new_output = self.new_copytensor(
                x[0], node.input_ids[0], node.output_ids[1]
            )
            # 2 present tensor explan
            # which means that at least 2 GPU can infer the behavior of the tensor generation.
            node.input_ids.insert(mid, new_input1)
            node.input_ids.append(new_input2)
            node.output_ids.append(new_output)
            # output will be different from input
        elif x[1] == 1:  # operation extend
            time, _, name, copy_a, copy_b = x
            self.operation_node[name] = copy.deepcopy(self.operation_node[copy_a])
            node = self.operation_node[name]
            node.input_ids = [
                self.new_copytensor(time, a, b)
                for (a, b) in zip(
                    self.operation_node[copy_a].input_ids,
                    self.operation_node[copy_b].output_ids,
                )
            ]
            node.output_ids = [
                self.new_copytensor(time, a, b)
                for (a, b) in zip(
                    self.operation_node[copy_a].output_ids,
                    self.operation_node[copy_b].output_ids,
                )
            ]
        else:  # tensor extend
            # copy all info
            _, _, name, copy_a, copy_b = x
            op_a, op_b = (
                self.tensor_node[copy_a].parent,
                self.tensor_node[copy_b].parent,
            )
            self.tensor_node[name] = copy.deepcopy(self.tensor_node[copy_a])
            # copy generation process
            self.tensor_node[name].parent = self.new_copyoperation(op_a, op_b)
            # generate son process
            self.tensor_node[name].son = [
                self.new_copyoperation(a, b)
                for (a, b) in zip(
                    self.tensor_node[copy_a].son, self.tensor_node[copy_b].son
                )
            ]

    def add_GPU_samegroup(self):
        self.extend_list = queue.PriorityQueue()
        self.tensor_copy_map = defaultdict(int)
        self.operation_copy_map = defaultdict(int)
        for id, node in self.operation_node.items():
            if node.ignore:
                continue
            if "alltoall" in node.name:
                self.extend_list.put((id, 0, id))  # 0 present alltoall extend operation
        while not self.extend_list.empty():
            x = self.extend_list.get()
            self.extend_list.get()
            self.extend(x)
        self.add_post_process()

    def add_post_process(self):
        pass

    def output(self, filename: str):
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
