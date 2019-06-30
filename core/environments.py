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
import collections 
import contextlib

import numpy as np

from autodiff.type_utils import is_variable
from autodiff.core.default_stack import _DEFAULT_STACK


class Graph(object):
  """A `Graph` and a set of `Node` instances, collectively define a 
  computational graph, on which forward-backward cycles are executed. All nodes 
  are bound to some `Graph` instance.

  When executed, the `_runtime` (i.e. the RunTime environment the Graph will be
  launched in) must be set by calling `self.set_runtime()`. 

  Stores the following attributes:    

  nodes: a list of Node instances, holding the nodes in a computational graph.
  name2node_map: a dict mapping from node name (string) to node instance. 
  node2name_map: a dict mapping from node instance to node name (string).
  node_type_count: a defaultdict (defaults to 0) mapping from node type (
    string) to the num of nodes with that type.
  """
  def __init__(self):
    """Constructor
    """
    self._runtime = None 

    self._nodes = []
    self._name2node_map = dict()
    self._node2name_map = dict()
    self._node_type_count = collections.defaultdict(int)

  def add_node(self, node):
    """Add a node to the computational graph.

    The added node will be given a unique name, with the pattern 'NodeType', 
    'NodeType_0', 'NodeType_1', ..., i.e. it starts with the name of the node
    type, and will be appended with the growing count of instances of that type.

    Args:
      node: a Node instance.
    """
    self._nodes.append(node)
    name = type(node).__name__

    if name not in self._name2node_map:
      self._name2node_map[name] = node
      self._node2name_map[node] = name
    else:
      new_name = '_'.join([name, str(self._node_type_count[name])])
      self._name2node_map[new_name] = node
      self._node_type_count[name] += 1
      self._node2name_map[node] = new_name

  def get_variables(self, only_trainable=True):
    """Return the list of variables.

    Args:
      only_trainable: bool scalar, whether to include trainable variables only.

    Returns:
      a list of Node instances represeting variables.
    """
    if only_trainable:
      return list(filter(lambda node: is_variable(node) and
          node._trainable, self._nodes))
    else:
      return list(filter(lambda node: is_variable(node), self._nodes))

  def save_variables(self, filename, only_trainable=False):
    """Save the values of variables to a *.npy file.

    Args:
      filename: string scalar, name of the *.npy file.
      only_trainable: whether to include trainable variables only.
    """
    variables = self.get_variables(only_trainable)
    var_name_map = dict([(self._node2name_map[var], var.val)
        for var in variables])
    np.save(filename, var_name_map)

  def initialize_variables(self, var_dict=None, only_trainable=False):
    """Initialize all variables.

    Args:
      var_dict: a dict mapping from node names (string) to node instances.
      only_trainable: whether to include trainable variables only.
    """
    variables = self.get_variables(only_trainable)
    for var in variables:
      initial_val = (var_dict[var.name] 
          if var_dict is not None and var.name in var_dict else None)
      var.initialize(initial_val)

  def get_runtime(self):
    """Return the RunTime instance the graph will be launched in."""
    if self._runtime is None:
      raise ValueError('runtime must be set.')
    return self._runtime

  def set_runtime(self, runtime):
    """Set the RunTime instance.

    Args:
      runtime: a RunTime instance.
    """
    self._runtime = runtime

  def reset_runtime(self):
    self._runtime = None

  @contextlib.contextmanager
  def as_default_graph(self):
    """Return the context manager that adds a scope in which all nodes defined
    inside have `self` as their graph. Example:

    graph = Graph()

    with graph.as_default_graph():
      a = Node()

    a.graph == graph
    """
    try:
      _add_scoped_graph(self)
      yield self
    finally:
      _remove_scoped_graph()


class RunTime(object):
  """RunTime stores the dynamic data that are valid only inside each graph 
  running session (i.e. forward-backward cycle). 

  Stores the following attributes:    
    fwval: a dict mapping from node name to numpy array, storing the forward
      pass value.
    bwval: a dict mapping from node name to numpy array, storing the backward
      pass value.
    cache_data: a defaultdict (defaults to empty dict) mapping from node name 
      to dict, storing data associated with the node with that name, which might
      be accessed multiple times inside a graph running session.
    backprop_count: a defaultdict (defaults to 0) mapping from node name to
      int, storing the num of consumer nodes that have already backpropped 
      gradients to the node with that name. 

  Note: 
  A RunTime instance's dynamic propreties should be reset at the end of each
  graph running session by calling `runtime.reset()`. Alternatively, one can
  place the code block running the graph within a RunTime context:

  rt = RunTime()
  for i in range(num_steps):
    with rt.forward_backward_cycle():
      node.forward()
      node.backward()

  is equivalent to 

  rt = RunTime()
  for i in range(num_steps):
    node.forward()
    node.backward()
    rt.reset()
  """
  def __init__(self):
    """Constructor."""

    # dynamic properties:
    self._fwval = dict()
    self._bwval = dict()
    self._cache_data = collections.defaultdict(dict)
    self._backprop_count = collections.defaultdict(int)

  def get_bwval(self, name):
    """Get the backward pass value of the node.

    Args:
      name: string scalar, name of the Node instance representing a variable.

    Returns:
      the backward pass value (i.e. gradient) of the node. 
    """
    if name not in self._bwval:
      raise ValueError('node \'%s\' not in bwval' % name)
    return self._bwval[name]
  
  def increment_backprop_count(self, node):
    """Increment the backprop count.

    Args:
      node: a Node instance.
    """
    self._backprop_count[node.name] += 1

  def backprop_initiated(self, node):
    """Returns whether backpropagation is initiated for the given node.

    Args:
      node: a Node instance.
    """
    return node.name in self._backprop_count
  
  def backprop_finished(self, node):
    """Returns whether backprogation is finished for the given node.

    Args:
      node: a Node instance.
    """
    return self._backprop_count[node.name] == node._num_consumers

  def reset(self):
    """Reset the dynamic properties.
    """
    self._fwval = dict()
    self._bwval = dict()
    self._cache_data = collections.defaultdict(dict)
    self._backprop_count = collections.defaultdict(int)

  @contextlib.contextmanager
  def forward_backward_cycle(self):
    """Returns the context manager that clears the dynamic data at the beginning
    and end of each forward-backward cycle.
    """
    try:
      self.reset()
      yield self
    finally:
      self.reset()


def get_default_graph():
  """Return the current graph.

  See docstring of default_stack.py. 
  """
  current = _DEFAULT_STACK[-1]
  if 'graph' not in current:
    current['graph'] = Graph()
  return current['graph']


def _add_scoped_graph(graph):
  """Adds a scoped graph.

  See docstring of default_stack.py.
  """
  _DEFAULT_STACK.append({'graph': graph})


def _remove_scoped_graph():
  """Removes the graph added by `_add_scoped_graph()`.

  See docstring of default_stack.py.
  """
  graph = _DEFAULT_STACK.pop()
