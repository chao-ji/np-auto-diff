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


class Optimizer(object):
  """Base class of all Optimizers.

  Subclasses must implement abstract method `apply_gradients`.
  """
  __metaclass__ = ABCMeta
  def __init__(self, **params):
    """Constructor.

    Args:
      params: a dict mapping from parameter names to parameters.
    """
    self._params = params
    self._params_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items()])

  def __repr__(self):
    """Displays the initializer name and list of parameter name and value pairs.
    """ 
    if self._params_str:
      return '<%s:%s>' % (type(self).__name__, self._params_str)
    else:
      return '<%s>' % type(self).__name__

  def compute_gradients(self, objective, feed_dict, var_list=None):
    """Pass an all-one numpy array to the objective tensor, and collect the 
    backpropped gradient w.r.t each trainable variable.

    Args:
      objective: a Node instance, the objective to be minimized.
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
      var_list: a list of Variable instances, the variables which graident will
        be computed with respect to.

    Returns:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
    """
    graph = objective._graph
    if var_list is None:
      var_list = graph.get_variables(only_trainable=True)

    objective.backward(feed_dict=feed_dict)
    grads_and_vars = [(graph.get_runtime()._bwval[v.name], v) 
        for v in var_list]
    return grads_and_vars

  def optimize(self, objective, feed_dict, var_list=None):
    """Optimize the objective by chaining `compute_gradients()` and 
    `apply_gradients()`.

    Call this function if no gradient postprocessing (e.g. scaling, clipping) 
    is needed.

    Args:
      objective: a Node instance, the objective to be minimized.
      feed_dict: a dict mapping from a `Node` instance to a numpy array.      
      var_list: a list of Variable instances, the variables the `objective`
        will be optimized over.
    """
    grads_and_vars = self.compute_gradients(objective, feed_dict, var_list)
    self.apply_gradients(grads_and_vars)

  @abstractmethod
  def apply_gradients(self):
    """Apply the computed gradient w.r.t. trainable variables.
    """


class GradientDescentOptimizer(Optimizer):
  """The Vanilla Gradient Descent Optimizer."""
  def apply_gradients(self, grads_and_vars):
    """Apply the computed gradient w.r.t. trainable variables.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
    """
    for grad, var in grads_and_vars:
      var.set_val(var.val - self._params['alpha'] * grad) 


class AdamOptimizer(Optimizer):
  """Adam optimizer"""
  def __init__(self, **params):
    """Constructor.

    Args:
      params: a dict mapping from parameter names to parameters.
    """
    self._params = params
    self._params_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items() 
        if k in ('alpha', 'beta1', 'beta2', 'epsilon')])

    self._t = 0
    self._m = None
    self._v = None

  def _initialize_moments(self, grads_and_vars):
    """Initializer the moments of variables as zero-valued tensors.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
    """
    moments = dict([(var.name, np.zeros(var.shape._raw_shape)) 
        for _, var in grads_and_vars])
    return moments

  def apply_gradients(self, grads_and_vars):
    """Apply the computed gradient w.r.t. trainable variables.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
    """
    alpha, beta1, beta2, epsilon = (self._params['alpha'], 
                                    self._params['beta1'], 
                                    self._params['beta2'], 
                                    self._params['epsilon'])
    t = self._t + 1
    m = self._m if self._m is not None else self._initialize_moments(
        grads_and_vars) 
    v = self._v if self._v is not None else self._initialize_moments(
        grads_and_vars)

    alpha_t = alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t))

    for grad, var in grads_and_vars:
      m[var.name] = beta1 * m[var.name] + (1 - beta1) * grad
      v[var.name] = beta2 * v[var.name] + (1 - beta2) * grad * grad
      var.set_val(var.val - 
          alpha_t * m[var.name] / (np.sqrt(v[var.name]) + epsilon))

    self._m = m
    self._v = v
    self._t = t


class RMSPropOptimizer(Optimizer):
  def __init__(self, **params):
    self._params = params
    self._params_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items()
        if k in ('alpha', 'rho', 'momentum', 'epsilon')])

    self._mean_square = None
    self._moment = None 

  def _initialize_moments(self, grads_and_vars):
    moments = dict([(var.name, np.zeros(var.shape._raw_shape)) 
        for _, var in grads_and_vars])
    return moments

  def apply_gradients(self, grads_and_vars):
    alpha, rho, momentum, epsilon = (self._params['alpha'], 
                                     self._params['rho'],
                                     self._params['momentum'], 
                                     self._params['epsilon'])

    mean_square = (self._mean_square if self._mean_square is not None else 
        self._initialize_moments(grads_and_vars))
    moment = (self._moment if self._moment is not None else
        self._initialize_moments(grads_and_vars))

    for grad, var in grads_and_vars:
      mean_square[var.name] = (rho * mean_square[var.name] + 
          (1 - rho) * grad * grad)
      moment[var.name] = momentum * moment[var.name] + alpha * grad / (np.sqrt(
          mean_square[var.name]) + epsilon)
      var.set_val(var.val - moment[var.name])

    self._mean_square = mean_square
    self._moment = moment
