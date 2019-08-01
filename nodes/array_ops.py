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
from autodiff.nodes import origin_nodes


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

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    y_val = self._arguments['y'].forward(feed_dict)
    dx_val = np.dot(grad_val, y_val.T)
    dy_val = np.dot(x_val.T, grad_val)
    grad_dict = {self._arguments['x']: dx_val,
                 self._arguments['y']: dy_val}
    return grad_dict


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

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    dx_val = np.reshape(grad_val, x_val.shape)
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict


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

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name] / np.prod(
        [x_val.shape[i] for i in self._axis])

    rep = [d if i in self._axis else 1 for i, d in enumerate(x_val.shape)]
    expanded_shape = [1 if i in self._axis else d 
        for i, d in enumerate(x_val.shape)] 
    dx_val = np.tile(grad_val.reshape(expanded_shape), rep)
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict


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

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name]

    rep = [d if i in self._axis else 1 for i, d in enumerate(x_val.shape)]
    expanded_shape = [1 if i in self._axis else d
        for i, d in enumerate(x_val.shape)]
    dx_val = np.tile(grad_val.reshape(expanded_shape), rep)
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict   


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

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    mask = self._get_mask(x_val.shape, self._paddings)
    dx_val = grad_val[mask]
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict   

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


class Concat(base_node.Node):
  """Concatenate tensors of the same rank along one of the dimensions. 
  """
  def __init__(self, tensors, axis=0, graph=None):
    """Constructor.

    Args:
      tensors: a list (>= 2) of Node instances, holding tensors of the same rank
        (i.e. num of dimension). They must have the same size in all dimension 
        except for the axis `axis`.
      axis: int scalar, the axis that `tensors` will be concatenated along. 
      graph: a Graph instance.
    """
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 2:
      raise ValueError('`tensors` must be either a list or tuple of size >= 2.')
    shapes = [t._shape for t in tensors]
    if len(set([s._ndims for s in shapes])) != 1:
      raise ValueError('inputs of Concat must have the same rank.')
    if not (0 <= axis < shapes[0]._ndims):
      raise ValueError('`axis` must be in `[0, ndims)`.') 

    shape = []
    for i in range(shapes[0]._ndims):
      sizes = [s._raw_shape[i] for s in shapes]
      if i != axis:
        sizes = set(sizes)
        if len(sizes) >= 3 or (len(sizes) == 2 and None not in sizes):
          raise ValueError('input nodes must have the same shape except for '
              ' the `axis` (%d) dimension. Got inconsistent shape at axis %d' % 
              (axis, i))
        sizes = list(sizes)
        shape.append(sizes[0] if len(sizes) == 1 
            or sizes[1] is None else sizes[1])
      else:    
        shape.append(None if None in sizes else sum(sizes))
          
    super(Concat, self).__init__(shape, graph)
    self._axis = axis
    for i, t in enumerate(tensors):
      self._arguments['x_' + str(i)] = t

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_vals = [self._arguments['x_' + str(i)].forward(feed_dict)
        for i in range(len(self._arguments))]
    val = np.concatenate(x_vals, axis=self._axis) 
    return val

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_vals = [self._arguments['x_' + str(i)].forward(feed_dict)
        for i in range(len(self._arguments))]

    sections = np.cumsum([x_val.shape[self._axis] for x_val in x_vals])[:-1]
    dx_vals = np.split(grad_val, sections, axis=self._axis)

    grad_dict = {}
    for i, dx_val in enumerate(dx_vals):
      grad_dict[self._arguments['x_' + str(i)]] = dx_val
    return grad_dict

class Slice(base_node.Node):
  """Slice out a sub-tensor from an input tensor."""
  def __init__(self, x, begin, sizes, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor to be sliced.
      begin: a Node instance of rank 1, or a list or tuple of ints of length
        x.shape.ndims (i.e. rank of `x`).
      sizes: a Node instance of rank 1, or a list or tuple of ints of length
        x.shape.ndims (i.e. rank of `x`).
      graph: a Graph instance.
    """
    begin = self._validate_input(begin, 'begin', graph)
    sizes = self._validate_input(sizes, 'sizes', graph)

    if (x._shape._ndims != begin._shape._raw_shape[0] or 
        x._shape._ndims != sizes._shape._raw_shape[0]):
      raise ValueError('the length of `begin` and `sizes` must be the same as '
          ' the rank of `x`.')

    if isinstance(sizes, origin_nodes.Constant):
      shape = [int(size) if size != -1 else None for size in sizes._val]
    else:
      shape = [None] * x._shape._ndims
    
    super(Slice, self).__init__(shape, graph)
    self._arguments['x'] = x
    self._arguments['begin'] = begin
    self._arguments['sizes'] = sizes

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    begin_val = self._arguments['begin'].forward(feed_dict)
    sizes_val = self._arguments['sizes'].forward(feed_dict)
    slices = self._get_slices(begin_val, sizes_val, x_val)
    slice_val = x_val[slices]
    return slice_val

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    x_val = self._arguments['x'].forward(feed_dict)
    begin_val = self._arguments['begin'].forward(feed_dict)
    sizes_val = self._arguments['sizes'].forward(feed_dict)

    dx_val = np.zeros_like(x_val, dtype='float32')
    slices = self._get_slices(begin_val, sizes_val, x_val)  
    dx_val[slices] = grad_val
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict

  def _get_slices(self, begin_val, sizes_val, x_val):
    """Utility function. Compute the slice indices.

    `sizes_val[i] == -1` indicates, the remaining elements after `begin_val[i]`
     in the `i`-th dimension will be slices.

    For each dimension `i`, we must have 

    0 <= begin_val[i] <= x_val.shape[1] 

    and  begin_val[i] <= begin_val[i] + sizes_val[i] <= x_val.shape[i]

    Args:
      begin_val: Numpy array of rank-1, the begin index in each dimension.
      sizes_val: Numpy array of rank-1, the size to be slices in each dimension.
      x_val: Numpy array of rank `len(begin_val)` or `len(sizes_val)`, the value
        of input tensor `x`.

    Returns:
      a list of python `slice` instances.
    """
    if 'slices' not in self._graph.get_runtime()._cache_data[self.name]:
      slices = []
      for begin, size, x_size in zip(begin_val, sizes_val, x_val.shape):
        if not (0 <= begin <= x_size):
          raise ValueError('for each dimension, `begin` must be >= 0 and '
              '<= x.dim_size[i] (%d), but got %d' % (x_size, begin))
        if size == -1:
          size = x_size - begin
        if not (begin <= begin + size <= x_size):
          raise ValueError('for each dimension, `begin + size` must be >= ' 
              '`begin` and <= x.dim_size[i] (%d), but got %d' % (
              x_size, begin + size))
        slices.append(slice(int(begin), int(begin + size)))
      self._graph.get_runtime()._cache_data[self.name]['slices'] = slices
    return self._graph.get_runtime()._cache_data[self.name]['slices']

  def _validate_input(self, input_, name, graph):
    """Utility function. Validate the input `begin` or `sizes`, and optionally
    convert them into `Constant` nodes if they are not `Node` instances. 
    
    `begin` and `sizes` are passed in as 1-D tensors, or as a list or tuple of 
    ints. In the latter case, they are automatically converted to `Contant` 
    nodes of rank 1.

    Args:
      input_: a Node instance of rank 1, or a list or tuple of integers.
      name: string scalar, name of the input.
      graph: a Graph instance.

    Returns:
      a Node instance of rank 1.
    """
    if not isinstance(input_, base_node.Node):
      if not isinstance(input_, (tuple, list)) or any([
          not isinstance(i, int) for i in input_]):
        raise TypeError('`%s` must be either a tuple or list of integers or a '
            'Node instance.' % name)

      input_ = origin_nodes.Constant(input_, graph) 
    if input_._shape._ndims != 1:
      raise ValueError('`%s` must be 1-D tensor.' % name)
    if input_._shape._raw_shape[0] is None:
      raise ValueError('the shape of `%s` must be determined at graph '
          'construction time.')

    return input_
