import numpy as np

from autodiff.core import base_node


class _BinaryOp(base_node.Node):
  """Base class for all binary arithmetic operations. Supports broadcasting.
  """
  def __init__(self, x, y, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the first operand.
      y: a Node instance, the second operand.
      graph: a Graph instance.
    """
    shape = self._compute_shape(x._shape, y._shape)
    super(_BinaryOp, self).__init__(shape, graph)
    self._arguments['x'] = x
    self._arguments['y'] = y
    self.increment_num_consumers_for_arguments()

  def _compute_shape(self, x_shape, y_shape):
    """Computs the shape of the resulting Node.

    Args:
      x_shape: a TensorShape instance, shape of the first operand.
      y_shape: a TensorShape instance, shape of the second operand.

    Returns:
      a tuple of integers (or None), shape of the resulting Node.
    """
    if x_shape._ndims == 0:
      return y_shape._raw_shape
    elif y_shape._ndims == 0:
      return x_shape._raw_shape

    shape = []
    min_ndims = np.minimum(x_shape._ndims, y_shape._ndims)
    for i in range(min_ndims):
      dx = x_shape._raw_shape[-1 - i]
      dy = y_shape._raw_shape[-1 - i]
      if dx is not None and dy is not None:
        if dx == dy:
          shape.append(dx)
        elif dx == 1:
          shape.append(dy)
        elif dy == 1:
          shape.append(dx)
        else:
          raise ValueError('operands x and y have incompatible shapes for '
              'broadcasting.' % (x_shape, y_shape))
      elif dx is not None:
        shape.append(None if dx == 1 else dx) 
      elif dy is not None:
        shape.append(None if dy == 1 else dy)
      else:
        shape.append(None)

    if min_ndims == x_shape._ndims:
      return tuple(y_shape._raw_shape[:-min_ndims]) + tuple(shape[::-1])
    else: # min_ndims == y_shape._ndims
      return tuple(x_shape._raw_shape[:-min_ndims]) + tuple(shape[::-1])

  def _compute_reduce_dims(
      self, dynamic_x_shape, dynamic_y_shape, dynamic_shape):
    """Compute list of dimensions to be reduces.

    Args:
      dynamic_x_shape: a tuple of ints, dynamic shape of the first operand.
      dynamic_y_shape: a tuple of ints, dynamic shape of the second operand.
      dynamic_shape: a tuple of ints, dynamic shape of the resulting Node. 
    """
    if len(dynamic_shape) > len(dynamic_x_shape):
      x_shape_expand = tuple(np.ones(len(dynamic_shape) - len(dynamic_x_shape), 
          dtype=int)) + dynamic_x_shape
    else:
      x_shape_expand = dynamic_x_shape
    if len(dynamic_shape) > len(dynamic_y_shape):
      y_shape_expand = tuple(np.ones(len(dynamic_shape) - len(dynamic_y_shape), 
          dtype=int)) + dynamic_y_shape
    else:
      y_shape_expand = dynamic_y_shape

    reduce_dims_x = tuple(np.where(np.array(dynamic_shape) != 
        np.array(x_shape_expand))[0])
    reduce_dims_y = tuple(np.where(np.array(dynamic_shape) != 
        np.array(y_shape_expand))[0])
    return reduce_dims_x, reduce_dims_y


class Add(_BinaryOp):
  """Add operation. Supports broadcasting."""
  def __init__(self, x, y, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the first operand.
      y: a Node instance, the second operand.
      graph: a RunTime instance.
    """
    super(Add, self).__init__(x, y, graph)

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return (self._arguments['x'].forward(feed_dict) +
        self._arguments['y'].forward(feed_dict))

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    
    dynamic_x_shape = self._arguments['x'].forward(feed_dict).shape
    dynamic_y_shape = self._arguments['y'].forward(feed_dict).shape
    dynamic_shape = self.forward(feed_dict).shape

    reduce_dims_x, reduce_dims_y = self._compute_reduce_dims(
        dynamic_x_shape, dynamic_y_shape, dynamic_shape)

    dx_val = grad_val.sum(axis=reduce_dims_x).reshape(dynamic_x_shape)
    dy_val = grad_val.sum(axis=reduce_dims_y).reshape(dynamic_y_shape)
    self._arguments['x'].backward(feed_dict, dx_val)
    self._arguments['y'].backward(feed_dict, dy_val)


class Multiply(_BinaryOp):
  """Multiply operation. Supports broadcasting."""
  def __init__(self, x, y, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the first operand.
      y: a Node instance, the second operand.
      graph: a RunTime instance.
    """
    super(Multiply, self).__init__(x, y, graph)

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    return (self._arguments['x'].forward(feed_dict) *
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
    dynamic_x_shape = x_val.shape
    dynamic_y_shape = y_val.shape
    dynamic_shape = self.forward(feed_dict).shape

    reduce_dims_x, reduce_dims_y = self._compute_reduce_dims(
        dynamic_x_shape, dynamic_y_shape, dynamic_shape)
    
    dx_val = (grad_val * y_val).sum(axis=reduce_dims_x).reshape(dynamic_x_shape)
    dy_val = (grad_val * x_val).sum(axis=reduce_dims_y).reshape(dynamic_y_shape)
    self._arguments['x'].backward(feed_dict, dx_val)
    self._arguments['y'].backward(feed_dict, dy_val)
