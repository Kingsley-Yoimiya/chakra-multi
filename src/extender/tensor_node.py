from ..converter.pytorch_node import PyTorchTensor


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
