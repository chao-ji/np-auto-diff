# Copyright (c) 2017 Chao Ji

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
import numpy as np

from autodiff.core import base_node
from autodiff.type_utils import isint


class MatMul(base_node.Node):
  """Computes the dot produt between two 2-D tensors with compatible shapes.
  """
  def __init__(self, x, y, graph=None):
    """Constructor.

    Args:
      x: a 2-D Node instance, the first operand.
      y: a 2-D Node instance, the second operand.
      graph: a RunTime instance.
    """
    if not (x._shape._ndims == 2 and y._shape._ndims == 2 and
        x._shape[-1:].compatible_with(y._shape[:1])):
      raise ValueError('shape of x(%s) and y(%s) in MalMul are not compatible.'
          % (x._shape, y._shape))

    shape = x._shape._raw_shape[0], y._shape._raw_shape[1]
    super(MatMul, self).__init__(shape, graph)
    self._arguments['x'] = x
    self._arguments['y'] = y
    self.increment_num_consumers_for_arguments()

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return np.dot(self._arguments['x'].forward(feed_dict),
        self._arguments['y'].forward(feed_dict))

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    y_val = self._arguments['y'].forward(feed_dict)
    dx_val = np.dot(grad_val, y_val.T)
    dy_val = np.dot(x_val.T, grad_val)
    self._arguments['x'].backward(feed_dict, dx_val)
    self._arguments['y'].backward(feed_dict, dy_val)


class Reshape(base_node.Node):
  """Reshapes the input tensor.
  """
  def __init__(self, x, target_shape, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the tensor to be reshaped.
      target_shape: a list (or tuple) of integers >= -1, the target shape that
        the input tensor will be reshaped to. Up to one entry can be -1, and 
        will be inferred based on the shape of input tensor.
      graph: a Graph instance.
    """
    if not isinstance(target_shape, (list, tuple)):
      raise TypeError('shape must be a list or tuple.')
    if not all([isint(i) and i >= -1 for i in target_shape]):
      raise ValueError('shape must contain integers >= -1.')
    if target_shape.count(-1) > 1:
      raise ValueError('shape must not contain more that one -1\'s.')
    shape = [i if i != -1 else None for i in target_shape]
    super(Reshape, self).__init__(shape, graph)
    self._target_shape = target_shape
    self._arguments['x'] = x
    self.increment_num_consumers_for_arguments()

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return np.reshape(self._arguments['x'].forward(feed_dict), 
        self._target_shape)

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    dx_val = np.reshape(grad_val, x_val.shape)
    self._arguments['x'].backward(feed_dict, dx_val)


class _ReductionOp(base_node.Node):
  """Base class of all Reduction operations."""
  def __init__(self, x, axis=None, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor on which the Reduction operation is 
        applied. 
      axis: a list (or tuple) of ints, or an int scalar. Defaults to all 
        dimensions, i.e. `range(x.shape.ndims)`.
      graph: a Graph instance.
    """
    if axis is None:
      axis = tuple(range(x._shape._ndims))
    if isint(axis):
      axis = (axis,)
    if not (isinstance(axis, (list, tuple)) and 
        all([isint(i) and i >= 0 and i < x._shape._ndims for i in axis])):
      raise ValueError('axis must be a tuple or list of ints >= 0 and < ndims.')
    shape = tuple(map(lambda p: p[1], 
        filter(lambda p: p[0] not in axis, enumerate(x._shape._raw_shape))))
    super(_ReductionOp, self).__init__(shape, graph)
    self._axis = tuple(axis)


class ReduceMean(_ReductionOp):
  """Takes the mean of elements in a tensor across dimensions.
  """
  def __init__(self, x, axis=None, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor on which the Reduction operation is 
        applied.
      axis: a list (or tuple) of ints, or an int scalar. Defaults to all 
        dimensions, i.e. `range(x.shape.ndims)`. 
      graph: a Graph instance.
    """
    super(ReduceMean, self).__init__(x, axis, graph)
    self._arguments['x'] = x
    self.increment_num_consumers_for_arguments()

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return self._arguments['x'].forward(feed_dict).mean(axis=self._axis)

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name] / np.prod(
        [x_val.shape[i] for i in self._axis])

    rep = [d if i in self._axis else 1 for i, d in enumerate(x_val.shape)]
    expanded_shape = [1 if i in self._axis else d 
        for i, d in enumerate(x_val.shape)] 
    dx_val = np.tile(grad_val.reshape(expanded_shape), rep)
    self._arguments['x'].backward(feed_dict, dx_val)


class ReduceSum(_ReductionOp):
  """Takes the sum of elements in a tensor across dimensions.
  """
  def __init__(self, x, axis=None, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor on which the Reduction operation is 
        applied.
      axis: a list (or tuple) of ints, or an int scalar. Defaults to all 
        dimensions, i.e. `range(x.shape.ndims)`. 
      graph: a Graph instance.
    """
    super(ReduceSum, self).__init__(x, axis, graph)
    self._arguments['x'] = x
    self.increment_num_consumers_for_arguments()

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return self._arguments['x'].forward(feed_dict).sum(axis=self._axis)

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name]

    rep = [d if i in self._axis else 1 for i, d in enumerate(x_val.shape)]
    expanded_shape = [1 if i in self._axis else d
        for i, d in enumerate(x_val.shape)]
    dx_val = np.tile(grad_val.reshape(expanded_shape), rep)
    self._arguments['x'].backward(feed_dict, dx_val)
    

class Pad(base_node.Node):
  """Pad the input tensor in any dimension(s) with constant value. 
  """
  def __init__(self, x, paddings, constant_value=0, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor to be padded.
      paddings: a 2-D numpy array of shape, [x.shape.ndims, 2]. `paddings[i, 0]`
        indicates the num of values *prepended* to the contents of `x`, while
        `paddings[i, 1]` indicates the num of values *appended* to the contents
        of `x`.
      constant_value: int scalar, the value to padded to input tensor. Defaults
        to 0.
      graph: a Graph instance.
    """
    raw_shape = list(map(int, np.sum(paddings, axis=1)))
    x._shape._check_raw_shape(raw_shape)
    shape = [d1 + d2 if d1 is not None else None 
        for d1, d2 in zip(x._shape._raw_shape, raw_shape)]
    
    super(Pad, self).__init__(shape, graph)
    self._paddings = paddings
    self._constant_value = constant_value
    self._arguments['x'] = x
    self.increment_num_consumers_for_arguments()

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    pad_val = np.ones(x_val.shape + np.sum(self._paddings, axis=1), 
        dtype=np.float32) * self._constant_value
    mask = self._get_mask(x_val.shape, self._paddings)
    pad_val[mask] = x_val
    return pad_val

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    mask = self._get_mask(x_val.shape, self._paddings)
    dx_val = grad_val[mask]
    self._arguments['x'].backward(feed_dict, dx_val)

  def _get_mask(self, x_shape, paddings):
    """Get the mask indicating the original contents of input tensor in the
    padded tensor.

    Args:
      x_shape: a tuple of ints, holding the dynamic shape of input tensor `x`.
      paddings: a 2-D numpy array of shape, [x.shape.ndims, 2]. `paddings[i, 0]`
        indicates the num of values *prepended* to the contents of `x`, while
        `paddings[i, 1]` indicates the num of values *appended* to the contents
        of `x`.

    Returns:
      a tuple of `slice` objects, used to slice out the original contents of 
        input tensor in the padded tensor.
    """
    if 'mask' not in self._graph.get_runtime()._cache_data[self.name]:
      mask = tuple([slice(p[0], p[0] + i) for p, i in zip(paddings, x_shape)])
      self._graph.get_runtime()._cache_data[self.name]['mask'] = mask
    return self._graph.get_runtime()._cache_data[self.name]['mask']
