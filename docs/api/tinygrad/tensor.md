# [tensor.py](/tinygrad/tensor.py)

## class `Tensor`

### Class Variables

#### `training: bool`

`False` by default. Set to `True` to enable training specific behavior on certain methods.

#### `no_grad: bool`

`False` by default. Set to `True` to disable the autograd engine and in turn the computation of gradients.

#### `default_type: DType`

`float32` by default. Controls the default data type of created tensors.

### Class Methods

#### `__init__(self, data: Union[int, float, list, LazyBuffer, np.ndarray], device:Optional[str]=None, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None) -> Tensor`

Creates a new tensor from the given data.

- `dtype` is the data type of the tensor. If not specified, the default type is used.
- `device` is the device the tensor is on. If not specified, the default device is used.
- `requires_grad` is whether the tensor should be tracked for gradient computation.
    - `None` means that the gradient will be tracked if this tensor is put into an optimizer.

### Properties

#### `.device -> str`

A property that returns the device the tensor is on.

#### `.shape -> Tuple[int, ...]`

A property that returns the shape of the tensor.

#### `.dtype -> DType`

A property that returns the data type of the tensor.

### Methods

#### `.realize() -> Tensor`

Realizes all lazied computations for the tensor, then returns it.

#### `.assign(self, x) -> Tensor`

Sets the underlying buffer of the tensor to the buffer of `x`.

When `self` is a disk tensor, this will instead write the buffer of `x` to the disk tensor.

#### `.detach(self) -> Tensor`

Detaches the tensor from the autograd engine, then returns it.

#### `.numpy(self) -> np.ndarray`

Returns a numpy array of the tensor's data.

#### `.to_(self, device:str)`

Sets the device of the tensor to `device`.

The tensor must not be realized for this to work, otherwise use `.to`.

#### `.to(self, device:str) -> Tensor`

Moves the tensor to `device`. This does not happen immediately, but rather when the tensor is realized.

#### `Tensor.empty(*shape, **kwargs) -> Tensor`

Creates an empty tensor with the given shape.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.manual_seed(seed=0)`

Sets the seed for random operations.

#### `Tensor.rand(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values between the interval `[0, 1)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.full(shape:Tuple[int, ...], fill_value, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with the given value.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.zeros(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.ones(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.arange(start, stop=None, step=1, **kwargs) -> Tensor`

If `stop` is not specified, creates a tensor with the given shape, filled with values from `0` to `start` with the given step size.

If `stop` is specified, creates a tensor with the given shape, filled with values from `start` to `stop` with the given step size.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.full_like(tensor, fill_value, dtype:Optional[DType]=None, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with the given value.
If `dtype` is not specified, the `dtype` of `tensor` is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.zeros_like(tensor, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.ones_like(tensor, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.eye(dim:int, **kwargs) -> Tensor`

Creates an identity matrix of the given dimension.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
If `dtype` is not specified, the default type is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a normal distribution with the given mean and standard deviation.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a uniform distribution with the given lower and upper bounds.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.scaled_uniform(*shape, **kwargs) -> Tensor`

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.glorot_uniform(*shape, **kwargs) -> Tensor`

<https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor`

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `Tensor.kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor`

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

#### `.backward(self)`

Backpropagates the gradients through the computation graph.

#### `.reshape(self, shape, *args) -> Tensor`

Reshapes a tensor to a new shape.

#### `.expand(self, shape, *args) -> Tensor`

Expands a tensor to a specified shape.
