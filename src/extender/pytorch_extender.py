import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import queue
import copy
from pathlib import Path


from ..converter.pytorch_node import PyTorchNode
from ..converter.pytorch_converter import PyTorchConverter
from .tensor_node import TensorNode
from .tensor_node import represent_tensor
from .operation_node import OperationNode


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
        self.oper_tot = max(json_node_map.keys()) + 1
        self.id_count = defaultdict(int)
        self.operation_node = {
            node_id: OperationNode(old_node)
            for node_id, old_node in json_node_map.items()
        }
        logging.debug(
            f"Start process trace map with metadata: {self.metadata}, oper_tot: {self.oper_tot}"
        )
        self.tensor_node: Dict[int, TensorNode] = {}
        self.relabel_tensor()
        self.rebuild_map()

    def relabel_tensor(self) -> None:
        """
        Before start rebuild the whole map, we should relabel tensors so that we will not confuse in the latter process.
        """
        self.tensor_count = 0
        self.tensor_trans = defaultdict(lambda: -1)
        for id, node in self.operation_node.items():
            if (
                id > 0
                and not PyTorchConverter().is_root_node(node.name)
                and not PyTorchConverter().is_root_node(
                    self.operation_node[node.parent].name
                )
            ):
                node.ignore = True
                logging.debug(f"relabel_tensor: ignore {id}")
                continue
            logging.debug(
                f"Processing the node {id} with {node.inputs['values']}, {node.inputs['shapes']}, {node.inputs['types']}"
            )
            for input_value, input_shape, input_type in zip(
                node.inputs["values"], node.inputs["shapes"], node.inputs["types"]
            ):
                if "Tensor" in input_type:
                    if input_type.startswith("GenericList[Tensor"):
                        # tensor_trans_list = []
                        for inner_value in input_value:
                            tensor = represent_tensor(inner_value)
                            logging.debug(
                                f"Current Tensor: from {inner_value} to {tensor}"
                            )
                            if self.tensor_trans[tensor] == -1:
                                self.tensor_trans[tensor] = self.tensor_count
                                self.tensor_node[self.tensor_count] = TensorNode(
                                    self.tensor_count,
                                    inner_value,
                                    input_shape,
                                    input_type,
                                )  # This is bugy, but we can temporarily ignore this.
                                self.tensor_count += 1
                            logging.debug(
                                f"Map tensor {tensor} in old node {id}'s input to {self.tensor_trans[tensor]}"
                            )
                            node.input_ids.append(self.tensor_trans[tensor])
                        # node.input_ids.append(tensor_trans_list)
                    else:
                        tensor = represent_tensor(input_value)
                        logging.debug(f"Current Tensor: from {input_value} to {tensor}")
                        if self.tensor_trans[tensor] == -1:
                            self.tensor_trans[tensor] = self.tensor_count
                            self.tensor_node[self.tensor_count] = TensorNode(
                                self.tensor_count, input_value, input_shape, input_type
                            )
                            self.tensor_count += 1
                        node.input_ids.append(self.tensor_trans[tensor])
                        logging.debug(
                            f"Map tensor {tensor} in old node {id}'s input to {self.tensor_trans[tensor]}"
                        )
            for output_value, output_shape, output_type in zip(
                node.outputs["values"], node.outputs["shapes"], node.outputs["types"]
            ):
                if "Tensor" in output_type:
                    if output_type.startswith("GenericList[Tensor"):
                        # tensor_trans_list = []
                        for inner_value in output_value:
                            tensor = represent_tensor(inner_value)
                            self.tensor_trans[tensor] = self.tensor_count
                            self.tensor_node[self.tensor_count] = TensorNode(
                                self.tensor_count,
                                inner_value,
                                output_shape,
                                output_type,
                            )  # This is bugy, but we can temporarily ignore this.
                            self.tensor_count += 1
                            node.output_ids.append(self.tensor_trans[tensor])
                            logging.debug(
                                f"Map tensor {tensor} in old node {id}'s output to {self.tensor_trans[tensor]}"
                            )
                        # node.output_ids.append(tensor_trans_list)
                    else:
                        tensor = represent_tensor(output_value)
                        self.tensor_trans[tensor] = self.tensor_count
                        self.tensor_node[self.tensor_count] = TensorNode(
                            self.tensor_count, output_value, output_shape, output_type
                        )
                        self.tensor_count += 1
                        node.output_ids.append(self.tensor_trans[tensor])
                        logging.debug(
                            f"Map tensor {tensor} in old node {id}'s output to {self.tensor_trans[tensor]}"
                        )
        self.tensor_max_info: tuple[int, int] = tuple(
            map(max, zip(*self.tensor_trans.keys()))
        )
        logging.debug(f"Get the tensor_max_info: {self.tensor_max_info}")

    def rebuild_map(self) -> None:
        """
        After relabel tensors, we can reconstruct the computation map.
        """
        for id, node in self.operation_node.items():
            if node.ignore:
                self.operation_node[
                    self.operation_node[node.parent].id
                ].extra_node.append(id)
                logging.debug(
                    f"Because ignored, the node id {self.operation_node[node.parent].id} have the extra_node {id}."
                )
                continue
            if "alltoall" in node.name:
                logging.debug(f"The all2all node id {id} is skipped.")
                continue
            # we should ignore the node alltoall which will be process later.
            for x in node.input_ids:
                logging.debug(f"The node {x} have the son which id is {id}")
                if isinstance(x, list):
                    for y in x:
                        self.tensor_node[y].add_son(id)
                        logging.debug(f"The node {y} have the son which id is {id}")
                else:
                    self.tensor_node[x].add_son(id)
            for x in node.output_ids:
                logging.debug(f"The node {x} have the parent which id is {id}")
                if isinstance(x, list):
                    for y in x:
                        self.tensor_node[y].set_parent(id)
                        logging.debug(f"The node {y} have the parent which id is {id}")
                else:
                    self.tensor_node[x].set_parent(id)

    def new_copytensor(self, time: int, copy_a: int, copy_b: int) -> int:
        """
        In this process, we need to generate a new tensor c, whose behavior can refer to tensor a and b.
        We only generate id, leaving detailed information in the latter process in the priority_queue.

        Args:
            copy_a(int):        The id of the referred tensor A.
            copy_b(int):        The id of the referred tensor B.
        """
        logging.debug(f"Start copy tensor node {copy_a} {copy_b}.")
        if (
            copy_a == copy_b
        ):  # If the tensors are the same in old behave, then we don't need to generate a new one.
            logging.debug(f"Tensor copy_a == copy_b : return copy_a {copy_a}.")
            return copy_a
        if self.tensor_copy_map[(copy_a, copy_b)]:
            logging.debug(
                f"Tensor already copied! So the result is {self.tensor_copy_map[(copy_a, copy_b)]}."
            )
            return self.tensor_copy_map[(copy_a, copy_b)]
        self.tensor_copy_map[(copy_a, copy_b)] = self.tensor_count
        logging.debug(f"Generate a new tensor node called {self.tensor_count}")
        self.extend_list.put((time, 2, self.tensor_count, copy_a, copy_b))
        logging.debug(
            f"Put {(time, 2, self.tensor_count, copy_a, copy_b)} into the extender_list, remain to extend."
        )
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
        logging.debug(f"Start copy operation node {copy_a} {copy_b}.")
        if (
            copy_a == copy_b or copy_a == -1 or copy_b == -1
        ):  # If the operations are the same in old behave, then we don't need to generate a new one.
            logging.debug(
                f"Operation copy_a == copy_b or have -1: return copy_a {copy_a}."
            )
            return copy_a
        if self.operation_copy_map[(copy_a, copy_b)]:
            logging.debug(
                f"Operation already copied! So the result is {self.operation_copy_map[(copy_a, copy_b)]}."
            )
            return self.operation_copy_map[(copy_a, copy_b)]
        self.operation_copy_map[(copy_a, copy_b)] = self.oper_tot
        logging.debug(f"Generate a new Opeartion node called {self.oper_tot}")
        time = max(self.operation_node[copy_a].id, self.operation_node[copy_b].id)
        self.extend_list.put((time, 1, self.oper_tot, copy_a, copy_b))
        logging.debug(
            f"Put {(time, 1, self.oper_tot, copy_a, copy_b)} into the extender_list, remain to extend."
        )
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
        logging.debug(f"Start extend node x : {x}.")
        if x[1] == 0:  # all to all extend
            node = self.operation_node[x[2]]
            logging.debug(f"The input_ids of x is: {node.input_ids}.")
            logging.debug(f"The output_ids of x is: {node.output_ids}.")
            mid = len(node.output_ids)
            new_input1 = self.new_copytensor(x[0], node.input_ids[0], node.input_ids[1])
            new_input2 = self.new_copytensor(
                x[0], node.input_ids[mid], node.input_ids[mid + 1]
            )
            new_output = self.new_copytensor(
                x[0], node.output_ids[0], node.output_ids[1]
            )
            # 2 present tensor explan
            # which means that at least 2 GPU can infer the behavior of the tensor generation.
            node.input_ids.insert(mid, new_input1)
            node.input_ids.append(new_input2)
            node.output_ids.append(new_output)
            logging.debug(f"The result input_ids of x is: {node.input_ids}")
            logging.debug(f"The result output_ids of x is: {node.output_ids}")
            # output will be different from input
        elif x[1] == 1:  # operation extend
            time, _, name, copy_a, copy_b = x
            logging.debug(f"Start operation node copy: {name} <- {copy_a}, {copy_b}")
            self.operation_node[name] = copy.deepcopy(self.operation_node[copy_a])
            node = self.operation_node[name]
            logging.debug(
                f"Op {copy_a}'s output_ids: {self.operation_node[copy_a].output_ids}"
            )
            logging.debug(
                f"Op {copy_b}'s output_ids: {self.operation_node[copy_b].output_ids}"
            )
            node.output_ids = [
                self.new_copytensor(time, a, b)
                for (a, b) in zip(
                    self.operation_node[copy_a].output_ids,
                    self.operation_node[copy_b].output_ids,
                )
            ]
            output_map_a = {
                x: y
                for (x, y) in zip(
                    self.operation_node[copy_a].output_ids, node.output_ids
                )
            }
            output_map_b = {
                x: y
                for (x, y) in zip(
                    self.operation_node[copy_b].output_ids, node.output_ids
                )
            }
            logging.debug(
                f"Op {copy_a}'s input_ids: {self.operation_node[copy_a].input_ids}"
            )
            logging.debug(
                f"Op {copy_b}'s input_ids: {self.operation_node[copy_b].input_ids}"
            )
            node.input_ids = []
            for i in range(
                min(
                    len(self.operation_node[copy_a].input_ids),
                    len(self.operation_node[copy_b].input_ids),
                )
            ):
                a, b = (
                    self.operation_node[copy_a].input_ids[i],
                    self.operation_node[copy_b].input_ids[i],
                )
                if a in output_map_a:
                    self.operation_copy_map[copy_a].input_ids[i] = output_map_a[a]
                    node.input_ids.append(a)
                elif b in output_map_b:
                    self.operation_copy_map[copy_b].input_ids[i] = output_map_b[b]
                    node.input_ids.append(b)
                else:
                    node.input_ids.append(self.new_copytensor(time, a, b))
            node.input_ids = [
                self.new_copytensor(time, a, b)
                for (a, b) in zip(
                    self.operation_node[copy_a].input_ids,
                    self.operation_node[copy_b].input_ids,
                )
            ]
            node.copy_from = (copy_a, copy_b)
            logging.debug(f"node.input_ids: {node.input_ids}")
            logging.debug(f"node.output_ids: {node.output_ids}")
            logging.debug(f"node.copy_from: (copy_a, copy_b)")
        else:  # tensor extend
            # copy all info
            _, _, name, copy_a, copy_b = x
            op_a, op_b = (
                self.tensor_node[copy_a].parent,
                self.tensor_node[copy_b].parent,
            )
            logging.debug(f"Tensor extend, extend source is {op_a}, {op_b}")
            self.tensor_node[name] = copy.deepcopy(self.tensor_node[copy_a])
            node = self.tensor_node[name]
            # copy generation process
            node.parent = self.new_copyoperation(op_a, op_b)
            # generate son process
            logging.debug(f"The node copy_a's son is: {self.tensor_node[copy_a].son}")
            logging.debug(f"The node copy_b's son is: {self.tensor_node[copy_b].son}")
            node.son = [
                self.new_copyoperation(a, b)
                for (a, b) in zip(
                    self.tensor_node[copy_a].son, self.tensor_node[copy_b].son
                )
            ]
            node.copy_from = (copy_a, copy_b)
            logging.debug(f"node.parent {node.parent}.")
            logging.debug(f"node.son: {node.son}")
            logging.debug(f"node.copy_from: {node.copy_from}")

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
            self.extend(x)
        self.restore_tensor()
        self.add_post_process()

    def restore_tensor(self):
        """
        Recover tensors to their trace-style format.
        """
        logging.debug("Start restoring tensors.")
        for id, node in self.tensor_node.items():
            if node.copy_from != (-1, -1):
                continue
            logging.debug(
                f"Start processing the node which is copied from: {node.copy_from} called the node {id}"
            )
            self.tensor_max_info = (
                self.tensor_max_info[0] + 1,
                self.tensor_max_info[1] + 1,
            )
            node.value.tensor_data[0:2] = (
                self.tensor_max_info[0],
                self.tensor_max_info[1],
            )
            logging.debug(f"The new tensor_max_info is: {self.tensor_max_info}.")
            logging.debug(
                f"The node.value.tensor of tensor {id} is : {node.value.tensor_data}"
            )

        logging.debug("Start the Operation node's info changing.")
        for id, node in self.operation_node.items():
            if node.ignore:
                continue
            if node.copy_from != (-1, -1):
                continue
            logging.debug(f"Start the old node {id}'s changing.")
            logging.debug(f"values: {node.inputs['values']}")
            logging.debug(f"types: {node.inputs['types']}")
            cur_id = 0
            for input_id in range(len(node.inputs["values"])):
                if "Tensor" in node.inputs["types"][input_id]:
                    if node.inputs["types"][input_id].startswith("GenericList[Tensor"):
                        for tensor_id in range(len(node.inputs["values"][input_id])):
                            node.inputs["values"][input_id][tensor_id] = (
                                self.tensor_node[
                                    node.input_ids[cur_id]
                                ].value.tensor_data
                            )

                            # node.old_node.inputs["values"][input_id][tensor_id] = (
                            #     self.tensor_node[node.input_ids[cur_id]].value
                            # )
                            cur_id += 1
                            logging.debug(
                                f"Get: {node.inputs['values'][input_id][tensor_id]}"
                            )
                    else:
                        node.inputs["values"][input_id] = self.tensor_node[
                            node.input_ids[cur_id]
                        ].value.tensor_data
                        # node.old_node.inputs["values"][input_id] = self.tensor_node[
                        #     node.input_ids[cur_id]
                        # ].value
                        logging.debug(f"Get: {node.inputs['values'][input_id]}")
                        cur_id += 1
            cur_id = 0
            for output_id in range(len(node.outputs["values"])):
                if "Tensor" in node.outputs["types"][output_id]:
                    if node.outputs["types"][output_id].startswith(
                        "GenericList[Tensor"
                    ):
                        for tensor_id in range(len(node.outputs["values"][output_id])):
                            node.outputs["values"][output_id][tensor_id] = (
                                self.tensor_node[
                                    node.output_ids[cur_id]
                                ].value.tensor_data
                            )
                            # node.old_node.outputs["values"][output_id][tensor_id] = (
                            #     self.tensor_node[node.output_ids[cur_id]].value
                            # )
                            cur_id += 1
                    else:
                        node.outputs["values"][output_id] = self.tensor_node[
                            node.output_ids[cur_id]
                        ].value.tensor_data
                        # node.old_node.outputs["values"][output_id] = self.tensor_node[
                        #     node.output_ids[cur_id]
                        # ].value
                        cur_id += 1
            logging.debug(f"values: {node.inputs['values']}")
            logging.debug(f"types: {node.inputs['types']}")

    def process_extra_operation(
        self,
        id: int,
        id_info: tuple[int, int],
        value_tensor: Dict[int, int] = {},
        value_storage: Dict[int, int] = {},
    ) -> tuple[int, tuple[int, int]]:
        """
        Always, for a operation node, such as matmul, there are other specific procedure node have ctrl_deps on its.
        We can call these nodes as extra node, which describe the lower operation of a simple operation.
        The ctrl_deps of these nodes form a tree, which we can process by dfs.
        We need to copy node from the old extra, then we need to regenerate tensor's info (see tensor_node), refer to old node's tensor info(so we use 2 maps).

        Args:
            id(int):                            current processing node's id.
            id_info(Tuple[int, int]):           Max tensor & storage id of the whole trace map.(See tensor_node)
            value_tensor(Dict[int, int]):       The dict that map old_value's tensor to new_value's tensor.
            value_storage(Dict[int, int]):      The dict that map old_value's storage to new_value's storage.

        Returns:
            id(int):                            Excepts the root of the extra_tree, each node will be copyed and regenerated a new id.
            new_id_info(Tuple[int, int]):       Return the new max tensor & storage id of the whole trace map.
        """
        logging.debug(
            f"ID: {id}, Id_info: {id_info}, value_tensor: {value_tensor}, value_storage: {value_storage}"
        )
        if value_tensor == {}:
            cur: OperationNode = self.operation_node[id]
            if cur.copy_from[0] == -1 or cur.copy_from[1] == -1:
                return id, id_info
            cur_node: PyTorchNode = cur.old_node
            logging.debug(f"cur_id: {id}, copy_id: {cur.copy_from}")
            cpy_node: PyTorchNode = self.operation_node[cur.copy_from[0]].old_node
            logging.debug(f"cur_node: {cur_node}, cpy_node: {cpy_node}")
            # The tensor node's first two arguments called tensor and storage.
            logging.debug(f"cur_node.inputs[values]: {cur_node.inputs['values']}")
            logging.debug(f"cpy_node.inputs[values]: {cpy_node.inputs['values']}")
            logging.debug(f"cur_node.inputs[types]: {cur_node.inputs['types']}")
            for cur_t, cpy_t, tp in zip(
                cur_node.inputs["values"],
                cpy_node.inputs["values"],
                cur_node.inputs["types"],
            ):
                if "Tensor" not in tp:
                    continue
                logging.debug(f"cur_t: {cur_t}, cpy_t: {cpy_t}")
                if tp.startswith("GenericList[Tensor"):
                    for cur_t_inner, cpy_t_inner in zip(cur_t, cpy_t):
                        value_tensor[cpy_t_inner[0]] = cur_t_inner[0]
                        value_storage[cpy_t_inner[1]] = cur_t_inner[1]
                else:
                    value_tensor[cpy_t[0]] = cur_t[0]
                    value_storage[cpy_t[1]] = cur_t[1]
            for cur_t, cpy_t, tp in zip(
                cur_node.outputs["values"],
                cpy_node.outputs["values"],
                cur_node.outputs["values"],
            ):
                if "Tensor" not in tp:
                    continue
                if tp.startswith("GenericList[Tensor"):
                    for cur_t_inner, cpy_t_inner in zip(cur_t, cpy_t):
                        value_tensor[cpy_t_inner[0]] = cur_t_inner[0]
                        value_storage[cpy_t_inner[1]] = cur_t_inner[1]
                else:
                    value_tensor[cpy_t[0]] = cur_t[0]
                    value_storage[cpy_t[1]] = cur_t[1]
        else:
            curid: int = self.oper_tot
            self.operation_node[curid] = copy.deepcopy(self.operation_node[id])
            self.oper_tot += 1
            id = curid
            cur: OperationNode = self.operation_node[curid]
            cur_node: PyTorchNode = cur.old_node
            logging.debug(f"cur_node.inputs[values]: {cur_node.inputs['values']}")
            for cur_t, tp in zip(cur_node.inputs["values"], cur_node.inputs["types"]):
                if "Tensor" not in tp:
                    continue
                if tp.startswith("GenericList[Tensor"):
                    for cur_t_inner in cur_t:
                        if cur_t_inner[0] not in value_tensor:
                            value_tensor[cur_t_inner[0]] = id_info[0] + 1
                            id_info = id_info[0] + 1, id_info[1]
                        cur_t_inner[0] = value_tensor[cur_t_inner[0]]
                        if cur_t_inner[1] not in value_storage:
                            value_storage[cur_t_inner[1]] = id_info[1] + 1
                            id_info = id_info[0], id_info[1] + 1
                        cur_t_inner[1] = value_storage[cur_t_inner[1]]
                else:
                    if cur_t[0] not in value_tensor:
                        value_tensor[cur_t[0]] = id_info[0] + 1
                        id_info = id_info[0] + 1, id_info[1]
                    cur_t[0] = value_tensor[cur_t[0]]
                    if cur_t[1] not in value_storage:
                        value_storage[cur_t[1]] = id_info[1] + 1
                        id_info = id_info[0], id_info[1] + 1
                    cur_t[1] = value_storage[cur_t[1]]
            for cur_t, tp in zip(cur_node.outputs["values"], cur_node.outputs["types"]):
                if "Tensor" not in tp:
                    continue
                if tp.startswith("GenericList[Tensor"):
                    for cur_t_inner in cur_t:
                        if cur_t_inner[0] not in value_tensor:
                            value_tensor[cur_t_inner[0]] = id_info[0] + 1
                            id_info = id_info[0] + 1, id_info[1]
                        cur_t_inner[0] = value_tensor[cur_t_inner[0]]
                        if cur_t_inner[1] not in value_storage:
                            value_storage[cur_t_inner[1]] = id_info[1] + 1
                            id_info = id_info[0], id_info[1] + 1
                        cur_t_inner[1] = value_storage[cur_t_inner[1]]
                else:
                    if cur_t[0] not in value_tensor:
                        value_tensor[cur_t[0]] = id_info[0] + 1
                        id_info = id_info[0] + 1, id_info[1]
                    cur_t[0] = value_tensor[cur_t[0]]
                    if cur_t[1] not in value_storage:
                        value_storage[cur_t[1]] = id_info[1] + 1
                        id_info = id_info[0], id_info[1] + 1
                    cur_t[1] = value_storage[cur_t[1]]
        for i in range(len(cur.extra_node)):
            cur.extra_node[i], id_info = self.process_extra_operation(
                cur.extra_node[i], id_info
            )
        return id, id_info

    def add_post_process(self):
        logging.debug("Start the process for the extra node.")
        for id in range(self.oper_tot):
            node = self.operation_node[id]
            if node.ignore:
                continue
            if node.copy_from == (-1, -1):
                continue
            if node.extra_node:
                _, self.tensor_max_info = self.process_extra_operation(
                    id, self.tensor_max_info
                )
                logging.debug(
                    f"After the process of operation_node {id}, the new extra_node info is: {self.tensor_max_info}"
                )
            node.copy_from = (-1, -1)

    def output(self, filename):
        logging.debug(f"Start output the result to {filename}.")
        result = self.metadata
        nodes = [x.__dict__ for _, x in self.operation_node.items()]
        result["nodes"] = nodes
        with open(filename, "w") as f:
            json.dump(result, f)


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
            file = Path(filename)
            map.output(file.with_name(f"ext_{Path(file).name}"))
