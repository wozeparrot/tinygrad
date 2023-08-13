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

```python
t = Tensor([1, 2, 3], device="gpu")
print(t.device) #-> "GPU"
```

#### `.shape -> Tuple[int, ...]`

A property that returns the shape of the tensor.

```python
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.shape) #-> (2, 3)
```

#### `.dtype -> DType`

A property that returns the data type of the tensor.

```python
from tinygrad.helpers import dtypes
t = Tensor([1, 2, 3], dtype=dtypes.int32)
print(t.dtype) #-> dtypes.int
```

#### `.T -> Tensor`

A property that returns the transpose of the tensor.

```python
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.T.numpy()) #->
"""
[[1. 4.]
 [2. 5.]
 [3. 6.]]
"""
```

### Static Methods

#### `Tensor.empty(*shape, **kwargs) -> Tensor`

Creates an empty tensor with the given shape.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.empty(2, 3)
print(t.shape) #-> (2, 3)
```

#### `Tensor.manual_seed(seed=0)`

Sets the seed for random operations.

```python
Tensor.manual_seed(42)
print(Tensor._seed) #-> 42
```

#### `Tensor.rand(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values between the interval `[0, 1)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.rand(2, 3)
print(t.numpy()) #->
"""
[[0.5053239  0.6522992  0.4013064 ]
 [0.04377532 0.57715255 0.02002954]]
"""
```

#### `Tensor.full(shape:Tuple[int, ...], fill_value, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with the given value.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.full((2, 3), 42)
print(t.numpy()) #->
"""
[[42. 42. 42.]
 [42. 42. 42.]]
"""
```

#### `Tensor.zeros(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.zeros(2, 3)
print(t.numpy()) #->
"""
[[0. 0. 0.]
 [0. 0. 0.]]
"""
```

#### `Tensor.ones(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.ones(2, 3)
print(t.numpy()) #->
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
```

#### `Tensor.arange(start, stop=None, step=1, **kwargs) -> Tensor`

If `stop` is not specified, creates a tensor with the given shape, filled with values from `0` to `start` with the given step size.

If `stop` is specified, creates a tensor with the given shape, filled with values from `start` to `stop` with the given step size.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.arange(5)
print(t.numpy()) #-> [0. 1. 2. 3. 4.]

t = Tensor.arange(5, 10)
print(t.numpy()) #-> [5. 6. 7. 8. 9.]

t = Tensor.arange(5, 10, 2)
print(t.numpy()) #-> [5. 7. 9.]
```

#### `Tensor.full_like(tensor, fill_value, dtype:Optional[DType]=None, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with the given value.
If `dtype` is not specified, the dtype of `tensor` is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
ot = Tensor.ones(2, 3)
t = Tensor.full_like(ot, 42)
print(t.numpy()) #->
"""
[[42. 42. 42.]
 [42. 42. 42.]]
"""
```

#### `Tensor.zeros_like(tensor, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
ot = Tensor.ones(2, 3)
t = Tensor.zeros_like(ot)
print(t.numpy()) #->
"""
[[0. 0. 0.]
 [0. 0. 0.]]
"""
```

#### `Tensor.ones_like(tensor, **kwargs) -> Tensor`

Creates a tensor with the same shape as `tensor`, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
ot = Tensor.zeros(2, 3)
t = Tensor.ones_like(ot)
print(t.numpy()) #->
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
```

#### `Tensor.eye(dim:int, **kwargs) -> Tensor`

Creates an identity matrix of the given dimension.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.eye(3)
print(t.numpy()) #->
"""
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""
```

#### `Tensor.randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
If `dtype` is not specified, the default type is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy()) #->
"""
[[-0.80423534 -1.1013225  -0.90952474]
 [ 1.2801555  -2.288294    0.7078166 ]]
"""
```

#### `Tensor.normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a normal distribution with the given mean and standard deviation.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=10, std=2)
print(t.numpy()) #->
"""
[[ 8.391529   7.7973547  8.18095  ]
 [12.560311   5.423412  11.415633 ]]
"""
```

#### `Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a uniform distribution with the given lower and upper bounds.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.uniform(2, 3, low=2, high=10)
print(t.numpy()) #->
"""
[[6.042591  7.218394  5.210451 ]
 [2.3502026 6.6172204 2.1602364]]
"""
```

#### `Tensor.scaled_uniform(*shape, **kwargs) -> Tensor`

Creates a tensor with the given shape, filled with random values from a uniform distribution with a mean of zero and a standard deviation of `(prod(shape)**-0.5`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.scaled_uniform(2, 3)
print(t.numpy()) #->
"""
[[ 0.00434694  0.1243518  -0.080583  ]
 [-0.3725059   0.06299479 -0.39189425]]
"""
```

#### `Tensor.glorot_uniform(*shape, **kwargs) -> Tensor`

<https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.glorot_uniform(2, 3)
print(t.numpy()) #->
"""
[[ 0.01166405  0.33367088 -0.21622688]
 [-0.99953824  0.16903277 -1.0515627 ]]
"""
```

#### `Tensor.kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor`

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.kaiming_uniform(2, 3)
print(t.numpy()) #->
"""
[[ 0.01505744  0.43074572 -0.27913368]
 [-1.2903337   0.2182095  -1.3574935 ]]
"""
```

#### `Tensor.kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor`

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.kaiming_normal(2, 3)
print(t.numpy()) #->
"""
[[-0.6566226  -0.8991811  -0.74258673]
 [ 1.0451903  -1.8682909   0.57790095]]
"""
```

#### `Tensor.stack(tensors, dim=0) -> Tensor`

Stacks a list of tensors along a new dimension.

```python
t1 = Tensor.ones(3)
t2 = Tensor.zeros(3)
t = Tensor.stack([t1, t2])
print(t.numpy()) #->
"""
[[1. 1. 1.]
 [0. 0. 0.]]
"""
```

### Methods

#### `.realize() -> Tensor`

Realizes all lazied computations for the tensor, then returns it.

```python
t = Tensor.zeros(2, 3) + 2
t.realize()
```

#### `.assign(self, x) -> Tensor`

Sets the underlying buffer of the tensor to the buffer of `x`.

When `self` is a disk tensor, this will instead write the buffer of `x` to the disk tensor.

```python
ot = Tensor.ones(2, 3)
t = Tensor.zeros(2, 3)
t.assign(ot)
print(t.numpy()) #->
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
```

#### `.detach(self) -> Tensor`

Detaches the tensor from the autograd engine, then returns it.

```python
t = Tensor.zeros(2, 3) + 1
dt = t.detach()
```

#### `.numpy(self) -> np.ndarray`

Returns a numpy array of the tensor's data.

```python
t = Tensor.zeros(2, 3) + 1
print(t.numpy()) #->
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
```

#### `.to_(self, device:str)`

Sets the device of the tensor to `device`.

The tensor must not be realized for this to work, otherwise use `.to`.

```python
t = Tensor.zeros(2, 3) + 1
t.to_("cpu")
print(t.device) #-> "cpu"
```

#### `.to(self, device:str) -> Tensor`

Moves the tensor to `device`. This does not happen immediately, but rather when the tensor is realized.

```python
t = Tensor.zeros(2, 3) + 1
t = t.to("cpu").realize()
print(t.device) #-> "CPU"
```

#### `.backward()`

Backpropagates the gradients through the computation graph.

```python
t = Tensor.zeros(2, 3, requires_grad=True) + 1
t.sum().backward()
print(t.grad.numpy()) #->
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
```

#### `.reshape(self, shape, *args) -> Tensor`

Reshapes a tensor to a new shape. New shape can't be of 0-dimension.

```python
t1 = Tensor([1, 2, 3, 4, 5, 6])
t2 = t1.reshape((1,6))
print(t2.numpy()) #->
"""
[[1. 2. 3. 4. 5. 6.]]
"""
```

#### `.expand(self, shape, *args) -> Tensor`

Expands a tensor to a specified shape.

```python
t1 = Tensor([[1], [2], [3]])
t2 = t1.expand(3, 4)
print(t2.numpy()) #->
"""
[[1. 1. 1. 1.]
 [2. 2. 2. 2.]
 [3. 3. 3. 3.]]
"""
```

#### `.linear(weight:Tensor, bias:Optional[Tensor]=None)`

Applies a linear transformation to the current tensor.

```python
t = Tensor.randn(5,2)
w = Tensor.randn(2,3)
print(t.linear(w).numpy()) #->
"""
[[ 1.5833756  -0.91409355 -0.84546614]
 [-0.08422226  2.715486    1.2233448 ]
 [-0.7796457  -0.35701907  0.05967392]
 [-2.797903    0.7940688   1.1311363 ]
 [ 1.8159602  -1.7854995  -1.2953657 ]]
"""
b = Tensor.randn(5,3)
print(t.linear(w, b).numpy()) #->
"""
[[ 3.360947   -0.05869228 -1.0911431 ]
 [ 1.6087391   2.7058008   2.1051977 ]
 [-0.19086033 -0.61030436 -1.898639  ]
 [-3.275361    0.4857852   1.0790952 ]
 [ 1.9292889  -1.3569635  -0.12129784]]
"""
```
