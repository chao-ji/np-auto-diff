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
    self.increment_num_consumers_for_arguments()

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
    """
    grad_val = self._graph.get_runtime()._bwval[self.name]
    logits_probs_val = self._get_softmax(feed_dict)
    labels_val = self._arguments['labels'].forward(feed_dict)
    dlogits_val = logits_probs_val - labels_val
    dlogits_val = np.expand_dims(grad_val, -1) * dlogits_val
    self._arguments['logits'].backward(feed_dict, dlogits_val)

  def _get_softmax(self, feed_dict):
    """Compute the softmax of logits.

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
  pass
