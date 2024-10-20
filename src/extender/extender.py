import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple

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
from ..converter.pytorch_node import PyTorchNode, PyTorchNodeType
from ..converter.pytorch_converter import PyTorchConverter


class PyTorchExtender:
    """
    Some GPU trace + cluster info ->(pipeline / data / model / tensor parallelism)-> whole info.
    Current stage: only consider tensor / model para.
    """

    def extend(self, id: List[int], input_filenames: List[str]):
        """ """
        trace_info = []
        for file in input_filenames:
            json_trace = PyTorchConverter().load_json_execution_traces(file)
            json_metadata, json_node_map = PyTorchConverter().parse_json_trace(
                json_trace
            )
            trace_info.append(
                (
                    json_metadata,
                    json_node_map,
                )
            )
