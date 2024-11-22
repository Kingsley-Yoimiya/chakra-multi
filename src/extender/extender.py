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
from tensor_node import represent_tensor
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
        tensor_max_info(Tuple[int, int]):               The max tensor id and store id of the trace.
        extend_list(Queue.PriorityQueue()):             The nodes we need to copy(0: all2all, 1: opeation, 2: tensor).
        tensor_copy_map(defaultdict(int)):              The map((int, int) -> (int), (copy_a, copy_b) -> (gener_tensor_node)) that we use to check if we have copy this tensor.
        operation_copy_map(defaultdict(int)):           The map((int, int) -> (int), (copy_a, copy_b) -> (gener_operation_node)) that we use to check if we have copy this operation.
    """

    def __init__(
        self, json_metadata: Dict, json_node_map: Dict[int, PyTorchNode]
    ) -> None:
        """
        Initialize a Tracemap object using the meta data json and node_map json.

        Args:
            json_metadata (Dict):                       The info read from chakra trace.
            Json_node_map (Dict[int, PyTorchNode]):     Dictionary mapping the id and the PyTorchNode.
        """
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
        """
        Before start rebuild the whole map, we should relabel tensors so that we will not confuse in the latter process.
        """
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
                    tensor = represent_tensor(input_value)
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
                    tensor = represent_tensor(output_value)
                    self.tensor_trans[tensor] = self.tensor_count
                    self.tensor_node[self.tensor_count] = TensorNode(
                        self.tensor_count, output_value, output_shape, output_type
                    )
                    self.tensor_count += 1
                    node.output_ids.append(self.tensor_trans[tensor])
        self.tensor_max_info = tuple(map(max, zip(*self.tensor_trans.keys())))

    def rebuild_map(self) -> None:
        """
        After relabel tensors, we can reconstruct the computation map.
        """
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
        """
        In this process, we need to generate a new tensor c, whose behavior can refer to tensor a and b.
        We only generate id, leaving detailed information in the latter process in the priority_queue.

        Args:
            copy_a(int):        The id of the referred tensor A.
            copy_b(int):        The id of the referred tensor B.
        """
        if (
            copy_a == copy_b
        ):  # If the tensors are the same in old behave, then we don't need to generate a new one.
            return copy_a
        if self.tensor_copy_map[(copy_a, copy_b)]:
            return self.tensor_copy_map[(copy_a, copy_b)]
        self.tensor_copy_map[(copy_a, copy_b)] = self.tensor_count
        self.extend_list.put((time, 2, self.tensor_count, copy_a, copy_b))
        self.tensor_count += 1
        return self.tensor_count - 1

    def new_copyoperation(self, copy_a: int, copy_b: int) -> int:
        """
        In this process, we need to generate a new oper c, whose behavior can refer to oper a and b.
        We only generate id, leaving detailed information in the latter process in the priority_queue.

        Args:
            copy_a(int):        The id of the referred operation A.
            copy_b(int):        The id of the referred operation B.
        """
        if (
            copy_a == copy_b
        ):  # If the operations are the same in old behave, then we don't need to generate a new one.
            return copy_a
        if self.operation_copy_map[(copy_a, copy_b)]:
            return self.operation_copy_map[(copy_a, copy_b)]
        self.operation_copy_map[(copy_a, copy_b)] = self.oper_tot
        time = max(self.operation_node[copy_a].id, self.operation_node[copy_b].id)
        self.extend_list.put((time, 1, self.oper_tot, copy_a, copy_b))
        self.oper_tot += 1
        return self.oper_tot - 1

    def extend(self, x: Tuple) -> None:
        """
        We take a tuple x from the priority_queue, handling the copy process.
        x have 3 forms:
            1. (time, 0, id)            (Usually id == time) The origin of the whole process, starting from extending all2all nodes.
            2. (time, 1, c, a, b)       (Usually id == min{time(a), time(b)}) It's a tensor extending process. We need to complete the info of c, which can be inferred by tensor a and b.
            3. (time, 2, c, a, b)       (Usually id == min{time(a), time(b)}) It's a operation extending process. We need to complete the info of c, whichi can be inferred by operation a and b.

        Args:
            x(Tuple):                   The extending node we need to process.
        """
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
            node.copy_from = (copy_a, copy_b)
        else:  # tensor extend
            # copy all info
            _, _, name, copy_a, copy_b = x
            op_a, op_b = (
                self.tensor_node[copy_a].parent,
                self.tensor_node[copy_b].parent,
            )
            self.tensor_node[name] = copy.deepcopy(self.tensor_node[copy_a])
            node = self.tensor_node[name]
            # copy generation process
            node.parent = self.new_copyoperation(op_a, op_b)
            # generate son process
            node.son = [
                self.new_copyoperation(a, b)
                for (a, b) in zip(
                    self.tensor_node[copy_a].son, self.tensor_node[copy_b].son
                )
            ]
            node.copy_from = (copy_a, copy_b)

    def add_GPU_samegroup(self):
        """
        The most important function that we need in this class which can perform adding a node in tensor/model parallelization, where all node do all2all/allreduce/allgather in the same time without group id.
        """
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
        self.restore_tensor()
        self.add_post_process()

    def restore_tensor(self):
        """
        Recover tensors to their trace-style format.
        """
        for _, node in self.tensor_node.items():
            if node.copy_from != (-1, -1):
                continue
            self.tensor_max_info = (
                self.tensor_max_info[0] + 1,
                self.tensor_max_info[1] + 1,
            )
            node.value.tensor_data[0:2] = (
                self.tensor_max_info[0],
                self.tensor_max_info[1],
            )

        for _, node in self.operation_node.items():
            if node.ignore:
                continue
            if node.copy_from != (-1, -1):
                continue
            cur_id = 0
            for input_id in range(len(node.inputs["values"])):
                if "Tensor" in node.inputs["types"][input_id]:
                    node.inputs["values"][input_id] = self.tensor_node[
                        node.input_ids[cur_id]
                    ].value
                    node.old_node.inputs["values"][input_id] = node.inputs["values"][
                        input_id
                    ]
                    cur_id += 1
            cur_id = 0
            for output_id in range(len(node.outputs["values"])):
                if "Tensor" in node.outputs["types"][output_id]:
                    node.outputs["values"][output_id] = self.tensor_node[
                        node.output_ids[cur_id]
                    ].value
                    node.old_node.inputs["values"][output_id] = node.outputs["values"][
                        output_id
                    ]
                    cur_id += 1

    def process_extra_operation(
        self, id: int, relabel_tensor: dict[tuple[int, int], int]
    ) -> None:
        for x in self.operation_node[id].extra_node:
            pass

    def add_post_process(self):
        for id, node in self.operation_node.items():
            if node.ignore:
                continue
            if node.copy_from == (-1, -1):
                continue
            if node.extra_node:
                pass

    def output(self, filename: str):
        pass


class PyTorchExtender:
    """
    Some GPU trace + cluster info ->(pipeline / data / model / tensor parallelism)-> whole info.
    Current stage: only consider tensor / model para.
    """

    def extend(self, id: List[int], input_filenames: List[str]):
        """
        Only consider tensor / model parallelism extending process.
        """
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
