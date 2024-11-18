from ..converter.pytorch_node import PyTorchNode


class OperationNode(PyTorchNode):
    """
    Represents an operation node in a PyTorch execution trace which is initilized by a PyTorch node in chakra.
    And is combined with more information.
    Attributes:

    old_node(PyTorchNode):      Represents the PyTorchNode.
    extra_node(int):            Represents the ctrl deps sons which should be ignored.
    ignore(bool):               If the PyTorchNode have ctrl deps which is not the root, then it should be ignored and processed in the post_process.
    input_ids(List[int]):       Relabeled tensors of the PyTorchNode's inputs.
    output_ids(List[int]):      Relabeled tensors of the PyTorchNode's outputs.
    Other Info from old_node
    """

    def __init__(self, old_node: PyTorchNode):
        self.__dict__.update(old_node.__dict__)
        self.old_node = old_node
        self.extra_node = []
        self.ignore = False
        self.input_ids = []
        self.output_ids = []
