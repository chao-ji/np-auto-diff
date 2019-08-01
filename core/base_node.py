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
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from autodiff.type_utils import isint
from autodiff.type_utils import to_numpy

from autodiff import exports 


class TensorShape(object):
  """TensorShape wraps a "raw shape", i.e. a list (or tuple) of integers (or 
  None), which stores the static shape of a `Node` instance. The static shape 
  does not have to be fully defined at graph construction time. 

  Example:

  Shape(None, 224, 224, 3) is the static shape of a 4-D tensor, where the 0-th
  dimension (None) is a wildcard that matches any integer.
  """
  def __init__(self, *raw_shape):
    """Constructor.

    Args:
      raw_shape: a list (or tuple) of integers (or None), size of each 
        dimension.
    """
    self._check_raw_shape(raw_shape)
    self._raw_shape = raw_shape
    self._ndims = len(raw_shape)

  @property
  def ndims(self):
    return self._ndims

  def compatible_with(self, tensor_shape):
    """Checks if the `TensorShape` is compatible with another shape. Example: 

    `Shape(None, 1, 2, 3)` is compatible with `Shape(1, 1, 2, 3)`, while not
    compatible with `Shape(1, 2, 2, 3)`. 

    Args:
      tensor_shape: raw shape, i.e. a list (or tuple) of integers (or None), 
        or a `TensorShape` instance.
    """
    if not isinstance(tensor_shape, (TensorShape, list, tuple)):
      raise TypeError('tensor_shape must be a list or tuple or TensorShape.')
    if not isinstance(tensor_shape, TensorShape):
      tensor_shape = TensorShape(*tensor_shape)

    ndims_compatible = self._ndims == tensor_shape._ndims
    sizes_compatible = all([d1 is None or d2 is None or d1 == d2
        for d1, d2 in zip(self._raw_shape, tensor_shape._raw_shape)])
    return ndims_compatible and sizes_compatible

  def __repr__(self):
    return 'Shape(%s)' % ', '.join(map(str, self._raw_shape))

  def __getitem__(self, k):
    """Allows for indexing and slicing. Example:

    `Shape(1, 2, None)[1] == 2`, and
    `Shape(1, 2, None)[1:]` == TensorShape([2, None])`.
    """
    if isinstance(k, slice):
      return TensorShape(*self._raw_shape[k])
    else:
      return self._raw_shape[k]

  def _check_raw_shape(self, raw_shape):
    """Checks if raw shape is valid, i.e. a list (or tuple) of integers 
    (or None).

    Args:
      raw_shape: a list (or tuple) of integers (or None), size of each 
        dimension.
    """
    if not all([isint(i) or i is None for i in raw_shape]):
      raise TypeError('raw_shape must be a list or tuple of integers or None.')


class Node(object):
  """The base class of all Node types that a computational graph is composed of.
  A Node instance holds the forward pass value and backward pass value (i.e. the
  gradient, with the same shape as the forward pass value) of a tensor-valued 
  variable in a computational graph.

  To kick off the forward and backward pass, call `self.forward()` and 
  `self.backward`, respectively. 

  Subclasses (i.e. specific node types) must implement both `_forward()` and 
  `_backward()` (the only exception is that Nodes with 0 in-degree, e.g. 
  Constant, Variable, Placeholder, only needs to implement `_forward()`). They 
  define the subclass-specific logic to carry out the forward and backward pass, 
  and they are called internally by `forward()` and `backward()`.
  

  The example below explains the logic to be implemented in `_forward()` 
  and `_backward()`. Suppose there is a computational graph,

  a = b + c
  d = a * 1
  e = a^2
  
  where `a` consumes values of `b` and `c` -- `a` is a *consumer* of `b` and 
  `c`, and `b` and `c` each is an *argument* of `a`. Similary, `d` and `e` are 
  *consumers* of `a`, of which `a` is an *argument*.

  The `_forward()` method of `a` must implement the addition operation, which 
  asks for the values of `b` and `c` by calling `b.forward()` and `c.forward()`.

  This `_backward()` method of `a` computes the gradient w.r.t `b` and `c` that
  pass through `a`. When it's called, `a`'s gradient (da_val = da_d_val + 
  da_e_val) should have already been computed. This is guaranteed by traversing
  the nodes in BFS order.
  """
  __metaclass__ = ABCMeta

  def __init__(self, shape, graph):
    """Constructor.

    Args:
      shape: raw shape, i.e. a list (or tuple) of integers (or None), or a 
        `TensorShape` instance.
      graph: a Graph instance.
    """
    self._shape = (shape if isinstance(shape, TensorShape) 
        else TensorShape(*shape))
    self._graph = graph
    self._arguments = dict()

    self._graph.add_node(self)

  @property
  def shape(self):
    return self._shape

  @property
  def graph(self):
    return self._graph

  @property
  def name(self):
    """Returns the name of the node (string) in a computational graph (which is 
    unique).
    """
    return self._graph._node2name_map[self]

  def __repr__(self):
    """Displays the name and shape of a node."""
    return '<%s: shape=%s>' % (self.name, self._shape._raw_shape)

  def check_dynamic_shape(self, val):
    """Checks if the dynamic shape, i.e. the shape of an numpy array computed
    at graph construction time, is compatible with the node's static shape.

    Args:
      val: numpy array of any shape, the numpy array whose shape is to be 
        checked against the static shape.
    """
    if not self._shape.compatible_with(val.shape):
      raise ValueError('%s: static shape (%s) and dynamic shape (%s) are '
          'not compatible.' % (self.name, self._shape, val.shape))

  def forward(self, feed_dict=None):
    """Compute the forward pass value of the node, or retrieve that value if 
    already computed.

    Args:
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    if self.name not in self._graph.get_runtime()._fwval:
      val = self._forward(feed_dict)
      self.check_dynamic_shape(val)
      self._graph.get_runtime()._fwval[self.name] = val
    return self._graph.get_runtime()._fwval[self.name]

  @abstractmethod
  def _forward(self):
    """Compute the forward pass value of the node.

    To be implemented by subclasses.
    """

  def backward(self, feed_dict, bwval=None):
    """Backpropogates gradient to all Nodes reachable from the current node 
    `self`. Example:

    `argument.backward(feed_dict, bwval)`  

    The Nodes are traversed in the order of BFS, i.e., Nodes with smaller 
    distances to `self` are traversed before Nodes with larger distances.

    When a Node is dequeued, its gradient should've already been computed. Then
    tt calls its `_backward` method, which computes the gradient w.r.t to all 
    its child nodes (e.g. for a = b + c, b and c are child node of a) that pass 
    through `self`. It this way, it is easy to accumulate gradients backpropped
    from different paths.

    Args:
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
      bwval: a numpy array whose shape matches `self.shape`, the gradient 
        backpropped from one of its consumer node. If None, defaults to an
        all-one array with shape `self._shape`.
    """
    if self._graph.get_runtime().grad_stopped(self):
      return 

    val = self.forward(feed_dict)
    if bwval is None:
      bwval = np.ones(val.shape, dtype=np.float32)
    else:
      bwval = to_numpy(bwval)
      self.check_dynamic_shape(bwval)

    self._graph.get_runtime()._bwval[self.name] = bwval

    queue = [self]
    while queue:
      node = queue.pop()
      if not hasattr(node, '_backward'):
        continue

      grad_dict = node._backward(feed_dict)
      for child, grad_val in grad_dict.items():
        if self._graph.get_runtime().grad_stopped(child):
          continue

        if child.name not in self._graph.get_runtime()._bwval:
          self._graph.get_runtime()._bwval[child.name] = grad_val 
          queue.append(child) 
        else:
          self._graph.get_runtime()._bwval[child.name] += grad_val

  def _convert_arithmetic_operand(self, other):
    """Convert arithmetic operand to appropriate type.

    Args:
      other: a Node instance, or any type convertable to numpy array.
    """
    if not isinstance(other, Node):
      try:
        other = exports.constant(other)
      except Exception:
        raise TypeError(
            'other must be a Node instance or convertable to numpy array.')
    return other

  def __add__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.add(self, other)

  def __radd__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.add(other, self)

  def __mul__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.multiply(self, other)

  def __rmul__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.multiply(other, self)

  def __sub__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.subtract(self, other)

  def __rsub__(self, other):
    other = self._convert_arithmetic_operand(other)
    return exports.add(other, self)

  def __pos__(self):
    zero = self._convert_arithmetic_operand(0)
    return exports.add(zero, self)

  def __neg__(self):
    zero = self._convert_arithmetic_operand(0) 
    return exports.subtract(zero, self)
