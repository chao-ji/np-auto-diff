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

import numpy as np


class SoftmaxCrossEntropyLoss(base_node.Node):
  """Computes the elementwise softmax cross entropy.
  """
  def __init__(self, labels, logits, graph=None):
    """Constructor.

    Args:
      labels: a Node instance, the tensor holding the groundtruth class labels.
        The last dimension is treated as the class dimension.
      logits: a Node instance of same shape as `labels`, the tensor holding the
        logits. The last dimension is treated as the class dimension.
      graph: a Graph instance.
    """
    super(SoftmaxCrossEntropyLoss, self).__init__(
        labels._shape[:-1], graph)
    self._arguments['labels'] = labels
    self._arguments['logits'] = logits

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    logits_probs_val = self._get_softmax(feed_dict)
    labels_val = self._arguments['labels'].forward(feed_dict)
    cross_entropy_val = np.sum(-np.log(logits_probs_val) * labels_val, axis=-1)
    return cross_entropy_val

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
    logits_probs_val = self._get_softmax(feed_dict)
    labels_val = self._arguments['labels'].forward(feed_dict)
    dlogits_val = logits_probs_val - labels_val
    dlogits_val = np.expand_dims(grad_val, -1) * dlogits_val
    grad_dict = {self._arguments['logits']: dlogits_val}
    return grad_dict

  def _get_softmax(self, feed_dict):
    """Utility function. Compute the softmax of logits.

    Args:
      feed_dict: a dict mapping from a `Node` instance to a numpy array.
    
    Returns:
      numpy array of the same shape as the `logits` tensor, holding the softmax
        probability scores in the last (i.e. class) dimension.
    """
    if 'softmax' not in self._graph.get_runtime()._cache_data[self.name]:
      logits_val = self._arguments['logits'].forward(feed_dict)
      # softmax without overflow 
      logits_exp_val = np.exp(logits_val - np.max(logits_val))
      self._graph.get_runtime()._cache_data[self.name]['softmax'] = (
          logits_exp_val / logits_exp_val.sum(axis=-1, keepdims=True))
    return self._graph.get_runtime()._cache_data[self.name]['softmax']


class SigmoidCrossEntropyLoss(base_node.Node):
  """Computes the elementwise sigmoid cross entropy.
  """
  def __init__(self, labels, logits, graph=None):
    """Constructor.

    Args:
      labels: a Node instance, the tensor holding the groundtruth class labels.
      logits: a Node instance of same shape as `labels`, the tensor holding the
        logits. 
      graph: a Graph instance.
    """
    super(SigmoidCrossEntropyLoss, self).__init__(
        labels._shape, graph)
    self._arguments['labels'] = labels
    self._arguments['logits'] = logits

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    logits_val = self._arguments['logits'].forward(feed_dict)
    labels_val = self._arguments['labels'].forward(feed_dict)  

    cross_entropy_val = (np.maximum(logits_val, 0) - logits_val * labels_val + 
        np.log(1 + 1 / self._get_exp_abs(logits_val)))
    return cross_entropy_val

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
    logits_val = self._arguments['logits'].forward(feed_dict)
    labels_val = self._arguments['labels'].forward(feed_dict)

    dlogits_val = np.where(logits_val > 0, 1, 0) - labels_val + 1 / (
        self._get_exp_abs(logits_val) + 1) * np.where(logits_val > 0, -1, 1)
    dlogits_val = dlogits_val.astype('float32') * grad_val
    grad_dict = {self._arguments['logits']: dlogits_val}
    return grad_dict

  def _get_exp_abs(self, logits_val):
    """Utility function. Computes `exp(abs(.))`.

    Args:
      logits_val: Numpy array, holding the value of logits.

    Returns:
      Numpy array of the same shape, holding the element wise `exp(abs(.))`.
    """
    if 'exp_abs' not in self._graph.get_runtime()._cache_data[self.name]:
      self._graph.get_runtime()._cache_data[self.name]['exp_abs'] = np.exp(
          np.abs(logits_val))
    return self._graph.get_runtime()._cache_data[self.name]['exp_abs']
