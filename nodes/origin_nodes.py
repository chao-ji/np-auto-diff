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
from autodiff.core import base_node
from autodiff.type_utils import to_numpy


class Placeholder(base_node.Node):
  """Placeholder's value can be supplied at graph running time. Example:

  a = Placeholder((2, 3))
  feed_dict = {a: np.array([[1, 2, 3], [4, 5, 6]])}
  a.forward(feed_dict) == np.array([[1, 2, 3], [4, 5, 6]])

  """
  def __init__(self, shape, graph=None):
    """Constructor.

    Args:
      shape: raw shape, i.e. a list (or tuple) of integers (or None), or a 
        `TensorShape` instance.
      graph: a Graph instance. 
    """
    super(Placeholder, self).__init__(shape, graph)

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    """
    return to_numpy(feed_dict[self])


class Variable(Placeholder):
  """Variable stores the value of a node that still persists outside a 
  forward-backward cycle.

  Its value can be initialized in two ways:
  1. Call the `initialize()` method without argument: its value is initialized
    by the Initializer instance passed to the constructor.
  2. Simply pass a numpy array to the `initialize()` method.

  A Variable can be either trainable or not trainable: trainable variables' 
  value can be modified only by an Optimizer instance outside a forward-backward
  cycle; while untrainable variables' value can be modified anywhere.
  """
  def __init__(self, shape, initializer, trainable=True, graph=None):
    """Constructor.

    Args:
      shape: raw shape, i.e. a list (or tuple) of integers (or None), or a 
        `TensorShape` instance.
      initializer: an Initializer instance.
      trainable: bool scalar, whether the variable is trainable.
      graph: a Graph instance.
    """
    super(Variable, self).__init__(shape, graph)
    self._initializer = initializer
    self._val = None
    self._trainable = trainable

  def initialize(self, initial_val=None):
    """Initialize the variable.

    Args:
      initial_val: numpy array, used to initialize the variable. Optional. 
    """
    if initial_val is not None:
      self.set_val(initial_val)
    else:
      self.set_val(self._initializer(shape=self._shape._raw_shape))
    return self

  def set_val(self, val):
    """Set value.

    Args:
      val: numpy array, used to the value of variable.
    """
    self.check_dynamic_shape(val)
    self._val = val

  @property
  def val(self):
    return self._forward()

  @property
  def trainable(self):
    return self._trainable

  def _forward(self, _not_used=None):
    """Compute the value of the variable.
    """
    if self._val is None:
      raise ValueError('variable %s has not been initialized.' % self.name)
    return self._val


class Constant(base_node.Node):
  """Constant nodes store values that are not changed inside a RunTime instance.

  Note: when using arithmetic operators, Python integers are automatically 
  converted to constant nodes. Example:

  a = Placeholder()
  b = a + 1

  `1` is converted to a constant node holding the value 1.
  """
  def __init__(self, val, graph=None):
    """Constructor.

    Args:
      val: any numeric type convertable to numpy array.
      graph: a Graph instance.
    """
    val = to_numpy(val)
    super(Constant, self).__init__(val.shape, graph)
    self._val = val

  def _forward(self, _not_used=None):
    """Compute the value of the variable.
    """
    return self._val
