from ..converter.pytorch_node import PyTorchNode


class OperationNode(PyTorchNode):
    """
    Represents an operation node in a PyTorch execution trace which is initilized by a PyTorch node in chakra.
    And is combined with more information.

    Attributes:
        old_node(PyTorchNode):      Represents the PyTorchNode.
        extra_node(int):            Represents the ctrl deps sons which should be ignored.
        ignore(bool):               If the PyTorchNode have ctrl deps which is not the root, then it should be ignored and processed in the post_process.
        input_ids(Any):             Relabeled tensors of the PyTorchNode's inputs.(List[int] or List[List[int]])
        output_ids(Any):            Relabeled tensors of the PyTorchNode's outputs.(List[int] or List[List[int]])
        copy_from(Tuple[int, int]): The Node copy from this tuple(a, b). If not copy, a = b = -1.
        Other Info from old_node
    """

    def __init__(self, old_node: PyTorchNode) -> None:
        """
        Initialize an operation node with PytorchNode called old_node.

        Args:
            old_node(PyTorchNode):  The node of the PyTorchNode.
        """
        self.__dict__.update(old_node.__dict__)
        self.old_node = old_node
        self.extra_node = []
        self.ignore = False
        self.input_ids = []
        self.output_ids = []
        self.copy_from = (-1, -1)
