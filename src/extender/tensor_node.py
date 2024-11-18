from ..converter.pytorch_node import PyTorchTensor


class TensorNode:
    """
    Represents a tensor node in a PyTorch execution trace, initialized based on each tensor generation.

    Attributes:
        id (int):               Identifier of the node.(Also the `time` of the node)
        value(PyTorchTensor):   The value of the tensor node which is initialized by PyTorchTensor
                            with tensor_data (List[int]): Data of the tensor including tensor_id, storage_id, offset, number of elements, and
                            size of each element in bytes.
        shape(List[int]):       The shape of the tensor.
        type(str):              The type of the tensor in chakra node.
        son(List[int]):         The list of future chakra node will use this node as input to operation.
        parent(int):            The chakra node that generate this node.
    """

    def __init__(self, id: int, value: list[int], shape: list[int], type: str):
        """
        Initialize tensor node data, with tensor id, value, shape, type

        Args:
            id(int):                The tensor id.
            value(List[int]):       The detailed info of the tensor storage.
            shape(List[int]):       The tensor's shape info.
            type(str):              The tensor's type info.
        """
        self.id = id
        self.value = PyTorchTensor(value)
        self.shape = shape
        self.type = type
        self.son = []
        self.parent = -1

    def add_son(self, x: int) -> None:
        """
        Add a son x, which means this node will be use by x as input in the future.

        Args:
            x(int):                 The id of the son.
        """
        self.son.append(x)

    def set_parent(self, x: int) -> None:
        """
        Set parent, which represents the chakra node generate this node.

        Args:
            x(int):                 The id of the parent.
        """
        self.parent = x
