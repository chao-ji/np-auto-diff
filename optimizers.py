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

  def compute_gradients(self, objective, feed_dict):
    """Pass an all-one numpy array to the objective tensor, and collect the 
    backpropped gradient w.r.t each trainable variable.

    Args:
      objective: a Node instance, the objective to be minimized.
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
    """
    graph = objective._graph
    objective.backward(feed_dict=feed_dict)
    grads_and_vars = [(graph.get_runtime()._bwval[v.name], v) 
        for v in graph.get_variables(only_trainable=True)]
    return grads_and_vars

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

  def optimize(self, objective, feed_dict):
    """Optimize the objective by chaining `compute_gradients()` and 
    `apply_gradients()`.

    Call this function if no gradient postprocessing (e.g. scaling, clipping) 
    is needed.

    Args:
      objective: a Node instance, the objective to be minimized.
      feed_dict: a dict mapping from a `Node` instance to a numpy array.      
    """
    grads_and_vars = self.compute_gradients(objective, feed_dict)
    self.apply_gradients(grads_and_vars)


class AdamOptimizer(Optimizer):
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
    moments = dict([(var.name, np.zeros(var.shape._raw_shape)) 
        for _, var in grads_and_vars])
    return moments

  def apply_gradients(self, grads_and_vars):
    alpha, beta1, beta2, epsilon = (self._params['alpha'], 
                                    self._params['beta1'], 
                                    self._params['beta2'], 
                                    self._params['epsilon'])
    t = self._t
    m = self._m if self._m is not None else self._initialize_moments(
        grads_and_vars) 
    v = self._v if self._v is not None else self._initialize_moments(
        grads_and_vars)

    alpha_t = (alpha if t < 1 else 
        alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t)))

    for grad, var in grads_and_vars:
      m[var.name] = beta1 * m[var.name] + (1 - beta1) * grad
      v[var.name] = beta2 * v[var.name] + (1 - beta2) * grad * grad
      var.set_val(var.val - 
          alpha_t * m[var.name] / (np.sqrt(v[var.name]) + epsilon))

    self._m = m
    self._v = v
    self._t += 1

  def optimize(self, objective, feed_dict):
    grads_and_vars = self.compute_gradients(objective, feed_dict)
    self.apply_gradients(grads_and_vars)
    
