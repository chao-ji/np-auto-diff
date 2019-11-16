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

import numpy as np

from autodiff.core import base_node
from autodiff.type_utils import isint


class Sigmoid(base_node.Node):
  """Compute elementwise sigmoid of input tensor."""
  def __init__(self, x, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose elementwise sigmiod is to be 
        computed.
      graph: a Graph instance.
    """
    super(Sigmoid, self).__init__(x._shape, graph)
    self._arguments['x'] = x

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    return 1. / (1 + np.exp(-x_val))

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    val = self.forward(feed_dict)
    val = val * (1. - val)
    grad_val = self._graph.get_runtime()._bwval[self.name]
    dx_val = grad_val * val
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict


class Tanh(base_node.Node):
  """Compute elementwise tanh of input tensor."""
  def __init__(self, x, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose elementwise sigmiod is to be 
        computed.
      graph: a Graph instance.
    """
    super(Tanh, self).__init__(x._shape, graph)
    self._arguments['x'] = x

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    return np.tanh(x_val)

  def _backward(self, feed_dict):
    """Retrieve the gradient value of the current node. Then compute and 
    backprop the gradient w.r.t. the argument nodes of the current node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      grad_dict: a dict mapping from a `Node` instance to a numpy array, holding
        the gradient w.r.t. `self`'s arguments that pass through `self`.
    """
    val = self.forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name]
    grad_val = grad_val * (1 - val * val)
    grad_dict = {self._arguments['x']: grad_val}
    return grad_dict


class ReLU(base_node.Node):
  """Compute elementwise ReLU of input tensor."""
  def __init__(self, x, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose elementwise ReLU is to be 
        computed.
      graph: a Graph instance.
    """
    super(ReLU, self).__init__(x._shape, graph)
    self._arguments['x'] = x

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    return np.maximum(0, x_val)

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
    grad_val[x_val <= 0] = 0
    grad_dict = {self._arguments['x']: grad_val}
    return grad_dict


class LeakyReLU(base_node.Node):
  """Compute elementwise Leaky ReLU of input tensor."""
  def __init__(self, x, alpha=0.3, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose elementwise ReLU is to be 
        computed.
      alpha: float scalar, slope of the activation function for x < 0. 
      graph: a Graph instance.
    """
    super(LeakyReLU, self).__init__(x._shape, graph)
    self._arguments['x'] = x
    self._alpha = alpha

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    return np.where(x_val >= 0, x_val, x_val * self._alpha)

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
    grad_val = np.where(x_val <= 0, self._alpha * grad_val, grad_val)
    grad_dict = {self._arguments['x']: grad_val}
    return grad_dict


class Dropout(base_node.Node):
  """Compute elementwise dropout of input tensor.

  In training mode, some random set of elements are zeroed with probability 
  `rate`, while those that are not zeroed are scaled up by `1 / (1 - rate)`. 
  In test mode, all elements are output as is.
  """
  def __init__(self, x, rate, is_training=True, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor.
      rate: float scalar, the probability that an element is dropped.
      is_training: bool scalar, whether dropout is in training mode.
      graph: a Graph instance.
    """
    if not isinstance(rate, float):
      raise TypeError('`rate` must be a float.')
    if rate < 0 or rate > 1:
      raise ValueError('`rate` must be in [0, 1].')
    if not isinstance(is_training, bool):
      raise TypeError('is_training must be a bool.')
    super(Dropout, self).__init__(x._shape, graph)
    self._arguments['x'] = x
    self._rate = rate
    self._is_training = is_training

  def _forward(self, feed_dict):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose elementwise ReLU is to be 
        computed.
      graph: a Graph instance.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    if not self._is_training:
      return x_val
    x_dropout_val = x_val / (1 - self._rate)
    mask = self._get_mask(x_val.shape)
    x_dropout_val[mask] = 0.
    return x_dropout_val

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
    dx_val = grad_val / (1 - self._rate)
    mask = self._get_mask(grad_val.shape)
    dx_val[mask] = 0.
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict

  def _get_mask(self, shape):
    """Get the mask that indicates which elements are zeroed.

    Args:
      shape: a list or tuple of ints, holding the dynamic shape of input tensor.

    Returns:
      the indicator array of the same shape as input tensor, where 1's indicate
        entries to be zeroed.
    """
    if 'mask' not in self._graph.get_runtime()._cache_data[self.name]:
      self._graph.get_runtime()._cache_data[self.name]['mask'] = np.random.binomial(
          1, self._rate, size=shape).astype(np.bool) 
    return self._graph.get_runtime()._cache_data[self.name]['mask']


class FusedBatchNorm(base_node.Node):
  """Fused Batch Normalization where the batch statistics is computed internally
  as opposed to being computed externally and then passed in.
  """
  def __init__(self, 
               x, 
               scale,  
               offset, 
               moving_mean, 
               moving_variance,
               epsilon=0.001,
               decay=0.999,
               is_training=True,
               graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor to be batch normalized.
      scale: a Node instance holding 1-D tensor of shape x.shape[-1], the 
        scaling factor (i.e. `gamma`).
      offset: a Node instance holding 1-D tensor of shape x.shape[-1], the
        bias to be added (i.e. `beta`).
      moving_mean: a Node instance holding 1-D tensor of shape x.shape[-1], the 
        moving average of batch mean.
      moving_variance: a Node instance holding 1-D tensor of shape x.shape[-1], 
        the moving average of batch variance.
      epsilon: float scalar, small value added to variance to avoid dividing 
        by 0.
      decay: float scalar, the decaying factor to compute moving average of 
        batch statistics.
      is_training: bool scalar, whether batch norm is in training mode. 
      graph: a Graph instance.
    """
    if not (scale._shape._ndims == 1 and 
        scale._shape.compatible_with(x._shape[-1:])):
      raise ValueError('last dimension of x must be compatible with the shape '
          'of scale.')
    if not (offset._shape._ndims == 1 and
        offset._shape.compatible_with(x._shape[-1:])):
      raise ValueError('last dimension of x must be compatible with the shape '
          'of offset.')
    if moving_mean is None or moving_variance is None:
      raise ValueError('At inference time, both moving_mean and moving_variance '
          'should be provided.') 
    if not (moving_mean._shape._ndims == 1 and 
        moving_mean._shape.compatible_with(x._shape[-1:])):
      raise ValueError('last dimension of x must be compatible with the shape'  
          ' of moving_mean.')
    if not (moving_variance._shape._ndims == 1 and
        moving_variance._shape.compatible_with(x._shape[-1:])):
      raise ValueError('last dimension of x must be compatible with the shape' 
          ' of moving_variance.')

    super(FusedBatchNorm, self).__init__(x._shape, graph)
    self._dims = tuple(range(x._shape._ndims - 1))
    self._is_training = is_training
    self._epsilon = epsilon
    self._decay = decay
    self._arguments['x'] = x
    self._arguments['scale'] = scale
    self._arguments['offset'] = offset

    self._arguments['moving_mean'] = moving_mean
    self._arguments['moving_variance'] = moving_variance
    
  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Note: This `_forward()` method has the *side effect* of updating the moving
    statistics.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict) 
    if self._is_training:
      mean_val, variance_val, variance_val_ddof1 = self._get_batch_stats(x_val)
      self._update_moving_stats(mean_val, variance_val_ddof1)
    else:
      mean_val = self._arguments['moving_mean'].val
      variance_val = self._arguments['moving_variance'].val

    offset_val = self._arguments['offset'].forward(feed_dict) 
    scale_val = self._arguments['scale'].forward(feed_dict)
    standard_x_val = self._get_standard_x(x_val, 
                                          mean_val, 
                                          variance_val)
    val = standard_x_val * scale_val + offset_val
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
    x_val = self._arguments['x'].forward(feed_dict)
    mean_val, variance_val = self._get_batch_stats(x_val)[:2]
    offset_val = self._arguments['offset'].forward(feed_dict)  
    scale_val = self._arguments['scale'].forward(feed_dict)
    d_standard_x_val = grad_val * scale_val 
    
    d_mean_val = d_standard_x_val * np.power(variance_val + self._epsilon, -0.5)
    d_mean_val = -d_mean_val.sum(axis=self._dims)
    d_variance_val = d_standard_x_val * (x_val - mean_val) * np.power(
        variance_val + self._epsilon, -1.5)
    d_variance_val = -0.5 * d_variance_val.sum(axis=self._dims)
    dx_val = d_standard_x_val * np.power(variance_val + self._epsilon, -0.5)

    d_variance_val = d_variance_val * 2 * (x_val - mean_val)
#    dx_val = dx_val + (d_mean_val + d_variance_val) / np.prod(
#        [x_val.shape[_] for _ in self._dims]) 

    dx_val = dx_val + (d_mean_val + d_variance_val) / np.prod(
        np.array(x_val.shape)[np.array(self._dims)])

    d_offset_val = grad_val.sum(axis=self._dims)

    d_scale_val = self._get_standard_x(x_val, mean_val, variance_val) * grad_val
    d_scale_val = d_scale_val.sum(axis=self._dims)
    grad_dict = {self._arguments['x']: dx_val,
                 self._arguments['offset']: d_offset_val,
                 self._arguments['scale']: d_scale_val}
    return grad_dict

  def _get_standard_x(self, x_val, mean_val, variance_val):
    """Center and scale the input tensor `x` to zero-mean and unit variance.

    Args:
      x_val: numpy array, the forward pass value of input tensor `x`.
      mean_val: 1-D numpy array, the per-dimension mean of `x_val`. 
      variance_val: 1-D numpy array, the per-dimension variance of `x_val`.

    Returns:
      numpy array of same shape as input, the standardized input tensor. 
    """
    if 'standard_x' not in self._graph.get_runtime()._cache_data[self.name]:
      self._graph.get_runtime()._cache_data[self.name]['standard_x'] = (
          x_val - mean_val) / np.sqrt(variance_val + self._epsilon) 
    return self._graph.get_runtime()._cache_data[self.name]['standard_x'] 

  def _get_batch_stats(self, x_val):
    """Compute the batch mean and batch variance of input tensor x.

    Args:
      x_val: numpy array, the forward pass value of input tensor `x`.

    Returns:
      mean_val: 1-D numpy array, the per-dimension mean of `x_val`.
      variance_val: 1-D numpy array, the per-dimension variance of `x_val`.
      variance_val_ddof1: 1-D numpy array, same as `variance_val`, but computed
        with degree-of-freedom 1.
    """
    if 'batch_stats' not in self._graph.get_runtime()._cache_data[self.name]:
      mean_val = x_val.mean(axis=self._dims)
      variance_val = x_val.var(axis=self._dims)
      variance_val_ddof1 = x_val.var(axis=self._dims, ddof=1)
      self._graph.get_runtime()._cache_data[self.name]['batch_stats'] = (
          mean_val, variance_val, variance_val_ddof1)
    return self._graph.get_runtime()._cache_data[self.name]['batch_stats']

  def _update_moving_stats(self, mean_val, variance_val):
    """Update the moving mean and moving variance.

    Args:
      mean_val: 1-D numpy array, the per-dimension mean of `x_val`.
      variance_val: 1-D numpy array, the per-dimension variance of `x_val`.
    """
    self._arguments['moving_variance'].set_val(
        self._arguments['moving_variance'].val * self._decay +
        variance_val * (1 - self._decay))
    self._arguments['moving_mean'].set_val(
        self._arguments['moving_mean'].val * self._decay +
        mean_val * (1 - self._decay))
    

class _Kernel2D(base_node.Node):
  """Base class of all 2D kernel operations (Conv2D, MaxPool2D, AvgPool2D).
  """
  __metaclass__ = ABCMeta

  def __init__(self, shape, strides, padding, graph=None):
    """Constructor.

    Args:
      shape: a list (or tuple) of 4 integers (or None), the shape of the 
        resulting tensor.
      strides: a list (or tuple) of two ints, the stride in the height and width
        dimension. 
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      graph: a Graph instance.
    """
    super(_Kernel2D, self).__init__(shape, graph)
    self._strides = strides
    self._padding = padding

  def _get_shapes(self, x_val_shape, kernel_height, kernel_width):
    """Compute the spatial dimensions of the output tensor, and the padding 
    sizes according to the padding scheme.

    Padding sizes are computed according to

    https://www.tensorflow.org/versions/r1.3/api_guides/python/nn#Convolution

    Args:
      x_val_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depth] of the input tensor. 
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      out_height: int scalar, height of the output tensor.
      out_width: int scalar, width of the output tensor.
      padding: a tuple of 4 ints, the num of scalars padded to the beginning of
        height, end of height, begining of width, end of width dimension of the
        input tensor `x`. 
      x_pad_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depths] of the output tensor.
    """
    if 'padding' not in self._graph.get_runtime()._cache_data[self.name]:
      strides = self._strides
      if self._padding == 'SAME':
        out_height = int(np.ceil(float(x_val_shape[1]) / strides[0]))
        out_width = int(np.ceil(float(x_val_shape[2]) / strides[1]))

        pad_along_height = (max(kernel_height - strides[0], 0) 
            if x_val_shape[1] % strides[0] == 0
            else max(kernel_height - x_val_shape[1] % strides[0], 0))

        pad_along_width = (max(kernel_width - strides[1], 0) 
            if x_val_shape[2] % strides[1] == 0
            else max(kernel_width - x_val_shape[2] % strides[1], 0))

        padding = (pad_along_height // 2, 
                  pad_along_height - pad_along_height // 2,
                  pad_along_width // 2, 
                  pad_along_width - pad_along_width // 2)

      else: # padding == 'VALID'
        out_height = int(
            np.ceil(float(x_val_shape[1] - kernel_height + 1) / strides[0]))
        out_width = int(
            np.ceil(float(x_val_shape[2] - kernel_width + 1) / strides[1]))
        padding = 0, 0, 0, 0

      x_pad_shape = (x_val_shape[0], x_val_shape[1] + padding[0] + padding[1], 
                     x_val_shape[2] + padding[2] + padding[3], x_val_shape[3]) 
      self._graph.get_runtime()._cache_data[self.name]['padding'] = (
          out_height, out_width, padding, x_pad_shape)

    return self._graph.get_runtime()._cache_data[self.name]['padding']

  def _get_img_col_index(self, x_val_shape, kernel_height, kernel_width):
    """Compute the height and width coordinates of the upper left pixel of all
    image patches that match the size of the kernel. Example:

    Given the 4-by-4 image below, and a 2-by-2 kernel with strides [2, 2]
    0,0  0,1  0,2  0,3
    1,0  1,1  1,2  1,3
    2,0  2,1  2,2  2,3
    3,0  3,1  3,2  3,3

    [(h, w, h_index w_index)] = 
      [(0,  0,  0,  0),
       (0,  2,  0,  1),
       (2,  0,  1,  0),
       (2,  2,  1,  1)]

    Args:
      x_val_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depth] of the input tensor.
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      numpy array of shape [num_patches, 4], where each row holds a tuple of 4
        ints [h, w, h_index, w_index]. `h` and `w` correspond to the height and 
        width coordinates of the upper left pixel of each patch that match the 
        size of the kernel; `h_index` and `w_index` correspond to the height and
        width indices of each path.
    """
    if 'img_col_index' not in self._graph.get_runtime()._cache_data[self.name]:
      stride_height, stride_width = self._strides
      _, _, _, x_pad_shape = self._get_shapes(
          x_val_shape, kernel_height, kernel_width)
      _, in_height_pad, in_width_pad, _ = x_pad_shape

#      img_col_index = np.array([(h, w, h_index, w_index)
#          for h_index, h in enumerate(
#              np.arange(0, in_height_pad - kernel_height + 1, stride_height))
#          for w_index, w in enumerate(
#              np.arange(0, in_width_pad - kernel_width + 1, stride_width))])

      h_col_indices = np.arange(
          0, in_height_pad - kernel_height + 1, stride_height)
      w_col_indices = np.arange(
          0, in_width_pad - kernel_width + 1, stride_width)

      w_grid, h_grid = np.meshgrid(w_col_indices, h_col_indices)
      w_index_grid, h_index_grid = np.meshgrid(
          np.arange(w_col_indices.shape[0]), np.arange(h_col_indices.shape[0]))

      img_col_index = np.vstack([h_grid.ravel(), 
                                 w_grid.ravel(), 
                                 h_index_grid.ravel(), 
                                 w_index_grid.ravel()]).T

      self._graph.get_runtime()._cache_data[self.name]['img_col_index'] = img_col_index
    return self._graph.get_runtime()._cache_data[self.name]['img_col_index']

  def _pad_x(self, x_val, kernel_height, kernel_width):
    """Pad the input tensor with constant values along the height and width 
    dimensions.

    Args:
      x_val: 4-D numpy array, the forward pass value of the input tensor.
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      x_pad_val: 4-D numpy array, the padded input tensor.
    """
    _, _, padding, x_pad_shape = self._get_shapes(
        x_val.shape, kernel_height, kernel_width)
    _, in_height_pad, in_width_pad, _ = x_pad_shape
    pad_top, pad_bot, pad_left, pad_right = padding

    x_pad_val = (np.ones(x_pad_shape) * self._pad_value 
        if hasattr(self, '_pad_value') 
        else np.zeros(x_pad_shape, dtype=np.float32))

    x_pad_val[:,
        pad_top: in_height_pad - pad_bot,
        pad_left: in_width_pad - pad_right, :] = x_val
    return x_pad_val

  def _unpad_dx(self, dx_val, x_val_shape, kernel_height, kernel_width):
    """Unpad the padded tensor to its original shape.

    The padded gradient value `dx_val` will be unpadded to the original shape of
    the input tensor `x`.

    Args:
      dx_val: 4-D numpy array, the gradient w.r.t to the padded input tensor. 
      x_val_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depth] of the input tensor.
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      dx_val: 4-D numpy array, the unpadded gradient w.r.t. the input tensor.
    """
    _, _, padding, x_pad_shape = self._get_shapes(
        x_val_shape, kernel_height, kernel_width)
    _, in_height_pad, in_width_pad, _ = x_pad_shape
    pad_top, pad_bot, pad_left, pad_right = padding

    dx_val = dx_val[:,
        pad_top: in_height_pad - pad_bot,
        pad_left: in_width_pad - pad_right, :]
    return dx_val       
 
  def _get_x_tensor2patchmat(self, 
                             x_val, 
                             kernel_height, 
                             kernel_width, 
                             flat_batch=False, 
                             in_channels=None):
    """A preprocessing step that converts the input 4-D tensor into a 2-D tensor
    of the "patch matrix" layout. Example:

    Suppose the input 4-D tensor `x` has shape 
      [batch, in_height, in_width, in_channels] = [2, 3, 3, 2]
    and its element layout is as follows:

    0,0,0,0  0,0,1,0  0,0,2,0      0,0,0,1  0,0,1,1  0,0,2,1
    0,1,0,0  0,1,1,0  0,1,2,0      0,1,0,1  0,1,1,1  0,1,2,1
    0,2,0,0  0,2,1,0  0,2,2,0      0,2,0,1  0,2,1,1  0,2,2,1
                                                              batch dimension 
    1,0,0,0  1,0,1,0  1,0,2,0      1,0,0,1  1,0,1,1  1,0,2,1
    1,1,0,0  1,1,1,0  1,1,2,0      1,1,0,1  1,1,1,1  1,1,2,1
    1,2,0,0  1,2,1,0  1,2,2,0      1,2,0,1  1,2,1,1  1,2,2,1

                       in_channels dimension

    The four dimensions are [batch, height, width, in_channels], e.g. the 
    element 0,1,2,0 corresponds the pixel at coordinate [1, 2] in the 0th color
    channel of the 0th image in a batch.

    Given a 2x2 kernel, the *original* layout above is converted to the 
    *patch matrix* layout, i.e. a 2-D matrix of shape
    [out_height * out_width * batch, in_channels * kernel_height * kernel_width]

    where
    out_height = 2, out_width = 2, batch = 2,
    in_channels = 2, kernel_height = 2, kernel_width = 2.


    0,0,0,0  0,0,1,0  0,1,0,0  0,1,1,0  0,0,0,1  0,0,1,1  0,1,0,1  0,1,1,1
    1,0,0,0  1,0,1,0  1,1,0,0  1,1,1,0  1,0,0,1  1,0,1,1  1,1,0,1  1,1,1,1

    0,0,1,0  0,0,2,0  0,1,1,0  0,1,2,0  0,0,1,1  0,0,2,1  0,1,1,1  0,1,2,1
    1,0,1,0  1,0,2,0  1,1,1,0  1,1,2,0  1,0,1,1  1,0,2,1  1,1,1,1  1,1,2,1

    0,1,0,0  0,1,1,0  0,2,0,0  0,2,1,0  0,1,0,1  0,1,1,1  0,2,0,1  0,2,1,1
    1,1,0,0  1,1,1,0  1,2,0,0  1,2,1,0  1,1,0,1  1,1,1,1  1,2,0,1  1,2,1,1

    0,1,1,0  0,1,2,0  0,2,1,0  0,2,2,0  0,1,1,1  0,1,2,1  0,2,1,1  0,2,2,1
    1,1,1,0  1,1,2,0  1,2,1,0  1,2,2,0  1,1,1,1  1,1,2,1  1,2,1,1  1,2,2,1


    Args:
      x_val: 4-D numpy array, the forward pass value of the input tensor.
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.
      flat_batch: bool scalar, whether to flat the `batch`, `out_height`, 
        `out_width`, `in_channels` dimensions into one dimension. If True, the 
        patch matrix layout will be converted to a 2-D matrix of shape
        [out_height * out_width * batch * in_channels,
         kernel_height * kernel_width].
      in_channels: int scalar, num of input channels. Ignored if `flat_batch`
        is False.
          
    Returns:
      numpy array of shape [out_height * out_width * batch, 
        in_channels * kernel_height * kernel_width], the input tensor with 
        elements in the patch matrix format.
    """
    if 'x_val_mat' not in self._graph.get_runtime()._cache_data[self.name]:
      batch = x_val.shape[0]
      img_col_index_val = self._get_img_col_index(
          x_val.shape, kernel_height, kernel_width)
      x_val = self._pad_x(x_val, kernel_height, kernel_width)

#      tmp = [x_val[:, h:h+kernel_height, w:w+kernel_width, :].transpose(
#          0, 3, 1, 2).reshape((batch, -1)) for h, w, _, _ in img_col_index_val]

      def func(indices):
        h, w = indices
        return x_val[:, h:h+kernel_height, w:w+kernel_width, :
            ].transpose(0, 3, 1, 2).reshape((batch, -1))

      tmp = np.apply_along_axis(func, 1, img_col_index_val[:, :2])

      x_val_mat = np.vstack(tmp)
      if flat_batch and in_channels is not None:
        out_height, out_width, _, _ = self._get_shapes(
            x_val.shape, kernel_height, kernel_width)    
        x_val_mat = x_val_mat.reshape((
            out_height, out_width, -1, in_channels, kernel_height, kernel_width
            )).reshape((-1, kernel_height * kernel_width)) 
      self._graph.get_runtime()._cache_data[self.name]['x_val_mat'] = x_val_mat
    return self._graph.get_runtime()._cache_data[self.name]['x_val_mat']  

  def _dx_patchmat2tensor(self, dx_val_mat, x_val_shape, kernel_height, kernel_width):
    """A post-processing step that converts the gradient w.r.t. the "patch 
    matrix" back to a 4-D numpy array of the same shape as input tensor.

    Args:
      dx_val_mat: a 2-D numpy array, holding the gradient w.r.t. the "patch 
        matrix".
      x_val_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depth] of the input tensor. 
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      dx_val: 4-D numpy array of shape [batch, in_height, in_width, in_channels]
        , the gradient w.r.t. the input tensor.
    """
    out_height, out_width, _, x_pad_shape = self._get_shapes(
        x_val_shape, kernel_height, kernel_width)
    in_channels = x_val_shape[3]
    img_col_index_val = self._get_img_col_index(
        x_val_shape, kernel_height, kernel_width)

    dx_val_tmp = dx_val_mat.reshape((
        out_height, out_width, -1, in_channels, kernel_height, kernel_width
        )).transpose(0, 1, 2, 4, 5, 3)
    '''
    dx_val = np.zeros(x_pad_shape, dtype=np.float32)

    for h, w, h_index, w_index in img_col_index_val:
      dx_val[:, h:h+kernel_height, w:w+kernel_width, :] += dx_val_tmp[
          h_index, w_index]
    '''
    def func(indices):
      h, w, h_index, w_index = indices
      dx_val = np.zeros(x_pad_shape, dtype=np.float32)
      dx_val[:, h:h+kernel_height, w:w+kernel_width, :] = dx_val_tmp[h_index, w_index]
      return dx_val

    dx_val = np.apply_along_axis(func, 1, img_col_index_val)
    dx_val = dx_val.sum(axis=0)

    dx_val = self._unpad_dx(dx_val, x_val_shape, kernel_height, kernel_width)
    return dx_val

  def _compute_static_spatial_dim_size(
      self, input_size, kernel_size, stride_size, padding):
    """Compute the static size of height and width dimension of the output
    tensor.

    Args:
      input_size: int scalar or None, the height or width of the input tensor.
      kernel_size: int scalar, the height or width of the kernel. 
      stride_size: int scalar, the stride along the height or width dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').

    Returns:
      out_size: int scalar or NOne, the height or width of the output tensor. 
    """
    if input_size is None:
      out_size = None
    else:
      if padding == 'SAME':
        out_size = int(np.ceil(float(input_size) / stride_size))
      else: # padding == 'VALID'
        if kernel_size is None:
          out_size = None
        else:
          out_size = int(
              np.ceil(float(input_size - kernel_size + 1) / stride_size))
    return out_size

  def _check_arguments(self, x, kernel, strides, padding):
    """Checks if input arguments are valid.

    Args:
      x: 4-D node of shape [batch, height, width, in_channels], input tensor.
      kernel: 4-D node of shape [kernel_height, kernel_width, in_channels, 
        out_channels], the kernel.
      strides: a tuple of 2 ints, the stride along the height and width 
        dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID'). 
    """
    if x._shape._ndims != 4:
      raise ValueError('x in conv2d must be a 4-D tensor.')
    if kernel is not None:
      if kernel._shape._ndims != 4:
        raise ValueError('kernel in conv2d must be a 4-D tensor.')
      dim = self.input_channels_dim
      if not x._shape[3:].compatible_with(kernel._shape[dim:dim+1]):
        raise ValueError('input channels must be compatible: x.shape[3] = %s, '
            'kernel.shape[%d] = %s' % (x._shape[3], dim, kernel._shape[dim]))
    if not isinstance(strides, (list, tuple)):
      raise TypeError('strides must be of instance list or tuple.')
    if not (len(strides) == 2 and isint(strides[0]) and
                                  isint(strides[1]) and
                                  strides[0] > 0 and strides[1] > 0):
      raise ValueError('strides must be a list of two positive integers.')
    if padding not in ('SAME', 'VALID'):
      raise ValueError('Padding scheme should be "SAME" or "VALID".')


class Conv2D(_Kernel2D):
  """Compute 2D Convolution."""
  def __init__(self, x, kernel, strides, padding, graph=None):
    """Constructor.

    Args:
      x: 4-D node of shape [batch, height, width, in_channels], input tensor.
      kernel: 4-D node of shape [kernel_height, kernel_width, in_channels, 
        out_channels], the kernel.
      strides: a tuple of 2 ints, the stride along the height and width 
        dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      graph: a Graph instance.
    """
    self._check_arguments(x, kernel, strides, padding)

    out_height = self._compute_static_spatial_dim_size(x._shape._raw_shape[1], 
        kernel._shape._raw_shape[0], strides[0], padding)
    out_width = self._compute_static_spatial_dim_size(x._shape._raw_shape[2], 
        kernel._shape._raw_shape[1], strides[1], padding)

    if self.input_channels_dim == 2:
      shape = (x._shape._raw_shape[0], out_height, 
          out_width, kernel._shape._raw_shape[3])
    elif self.input_channels_dim == 3:
      shape = (x._shape._raw_shape[0], out_height,
          out_width, kernel._shape._raw_shape[2])
    else:
      raise ValueError('input channels dim must be 2 or 3.')

    super(Conv2D, self).__init__(shape, strides, padding, graph)
    self._arguments['x'] = x
    self._arguments['kernel'] = kernel

  @property
  def input_channels_dim(self):
    return 2

  def _get_kernel_tensor2patchmat(self, kernel_val):
    """A pre-processing step that converts the 4-D kernel tensor into a 2-D 
    tensor of the "patch matrix" layout. Example:

    Suppose the 4-D kernel tensor `kernel` has shape [kernel_height, 
    kernel_width, in_channels, out_channels] = [2, 2, 2, 2]

    0,0,0,0  0,1,0,0    0,0,1,0  0,1,1,0
    1,0,0,0  1,1,0,0    1,0,1,0  1,1,1,0
                                          out_channels dimension
    0,0,0,1  0,1,0,1    0,0,1,1  0,1,1,1
    1,0,0,1  1,1,0,1    1,0,1,1  1,1,1,1

            in_channels dimension
  
    `kernel` is converted into the "patch matrix" layout, i.e. a 2-D matrix
    of shape [in_channels * kernel_height * kernel_width, out_channels]

    0,0,0,0  0,0,0,1
    0,1,0,0  0,1,0,1
    1,0,0,0  1,0,0,1
    1,1,0,0  1,1,0,1
    0,0,1,0  0,0,1,1
    0,1,1,0  0,1,1,1
    1,0,1,0  1,0,1,1
    1,1,1,0  1,1,1,1

    The kernel in "patch matrix" layout will be left-multiplied with the
    input tensor in "patch matrix" layout to carry out the 2D Convolution
    operation.

    Args:
      kernel_val: 4-D numpy array of shape [kernel_height, kernel_width, 
        in_channels, out_channels], the forward pass value of the kernel.

    Returns:
      2-D numpy array of shape [in_channels * kernel_height * kernel_width, 
        out_channels], the kernel in "patch matrix" layout.
    """
    if 'kernel_val_mat' not in self._graph.get_runtime()._cache_data[self.name]:
      out_channels = kernel_val.shape[3]
      self._graph.get_runtime()._cache_data[self.name]['kernel_val_mat'] = (
          kernel_val.transpose(2, 0, 1, 3).reshape((-1, out_channels)))

    return self._graph.get_runtime()._cache_data[self.name]['kernel_val_mat'] 

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """ 
    x_val = self._arguments['x'].forward(feed_dict)
    kernel_val = self._arguments['kernel'].forward(feed_dict)

    out_height, out_width, _, _ = self._get_shapes(
        x_val.shape, kernel_val.shape[0], kernel_val.shape[1])
    batch, out_channels = x_val.shape[0], kernel_val.shape[3] 
    x_val_mat = self._get_x_tensor2patchmat(
        x_val, kernel_val.shape[0], kernel_val.shape[1])
    kernel_val_mat = self._get_kernel_tensor2patchmat(kernel_val)
    conv_val_mat = np.dot(x_val_mat, kernel_val_mat)
    return conv_val_mat.reshape(out_height, out_width, batch, out_channels
        ).transpose(2, 0, 1, 3)

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
    kernel_val = self._arguments['kernel'].forward(feed_dict)
    kernel_height, kernel_width, in_channels, out_channels = kernel_val.shape   

    grad_val = self._graph.get_runtime()._bwval[self.name]
    grad_val_mat = grad_val.transpose(1, 2, 0, 3).reshape((-1, out_channels))
    x_val_mat = self._get_x_tensor2patchmat(x_val, kernel_height, kernel_width)
    kernel_val_mat = self._get_kernel_tensor2patchmat(kernel_val)

    dkernel_val_mat = np.dot(x_val_mat.T, grad_val_mat)
    dkernel_val = dkernel_val_mat.reshape((
        in_channels, kernel_height, kernel_width, out_channels
        )).transpose(1, 2, 0, 3)

    dx_val_mat = np.dot(grad_val_mat, kernel_val_mat.T)
    dx_val = self._dx_patchmat2tensor(
        dx_val_mat, x_val.shape, kernel_height, kernel_width)

    grad_dict = {self._arguments['x']: dx_val,
                 self._arguments['kernel']: dkernel_val}
    return grad_dict    


class Conv2DTranspose(Conv2D):
  """Compute Transposed 2D Convolution. Also known as fractionally strided 
  convolution.
  """
  def __init__(self, x, kernel, strides, padding, graph=None):
    """Constructor.

    Args:
      x: 4-D node of shape [batch, height, width, in_channels], input tensor.
      kernel: 4-D node of shape [kernel_height, kernel_width, in_channels, 
        out_channels], the kernel.
      strides: a tuple of 2 ints, the stride along the height and width 
        dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      graph: a Graph instance.
    """
    super(Conv2DTranspose, self).__init__(x, kernel, strides, padding, graph)

  @property
  def input_channels_dim(self):
    return 3

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    kernel_val = self._arguments['kernel'].forward(feed_dict)

    kernel_height, kernel_width, in_channels, out_channels = kernel_val.shape
    batch = x_val.shape[0]
    in_height, in_width = self._get_input_spatial_size(
        x_val.shape, kernel_height, kernel_width)

    x_val_mat = x_val.transpose(1, 2, 0, 3).reshape((-1, out_channels))
    kernel_val_mat = self._get_kernel_tensor2patchmat(kernel_val)
    trans_conv_val_mat = np.dot(x_val_mat, kernel_val_mat.T)
    trans_conv_val = self._dx_patchmat2tensor(
        trans_conv_val_mat, 
        (batch, in_height, in_width, in_channels), 
        kernel_height, 
        kernel_width)

    return trans_conv_val

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
    kernel_val = self._arguments['kernel'].forward(feed_dict)
    grad_val = self._graph.get_runtime()._bwval[self.name]

    kernel_height, kernel_width, in_channels, out_channels = kernel_val.shape
    batch, out_height, out_width = x_val.shape[:-1]

    grad_val_mat = self._get_x_tensor2patchmat(
        grad_val, kernel_val.shape[0], kernel_val.shape[1])

    x_val_mat = x_val.transpose(1, 2, 0, 3).reshape((-1, out_channels))
    dkernel_val_mat = np.dot(x_val_mat.T, grad_val_mat)
    dkernel_val = dkernel_val_mat.reshape(
        (out_channels, in_channels, kernel_height, kernel_width)
        ).transpose((2, 3, 1, 0))

    kernel_val_mat = self._get_kernel_tensor2patchmat(kernel_val)
    dx_val_mat = np.dot(grad_val_mat, kernel_val_mat)
    dx_val = dx_val_mat.reshape(
        (out_height, out_width, batch, out_channels)).transpose((2, 0, 1, 3))

    grad_dict = {self._arguments['x']: dx_val,
                 self._arguments['kernel']: dkernel_val}
    return grad_dict


  def _get_input_spatial_size(self, x_val_shape, kernel_height, kernel_width):
    """Compute the height and with of the output tensor of Conv2dTranspose.

    Args:
      x_val_shape: a tuple of 4 ints, holding the dynamic shape 
        [batch, height, width, depth] of the input tensor.
      kernel_height: int scalar, height of the kernel.
      kernel_width: int scalar, width of the kernel.

    Returns:
      in_height: int scalar, height of input tensor. 
      in_width: int scalar, width of input tensor.
    """
    if self._padding == 'SAME':
      in_height = x_val_shape[1] * self._strides[0]
      in_width = x_val_shape[2] * self._strides[1]
    else:
      in_height = x_val_shape[1] * self._strides[0] + max(
          kernel_height - self._strides[0], 0)
      in_width = x_val_shape[2] * self._strides[1] + max(
          kernel_width - self._strides[1], 0)
  
    return in_height, in_width

  def _compute_static_spatial_dim_size(
      self, input_size, kernel_size, stride_size, padding):
    """Overrides the implementation of base class `_Kernel2D`. Compute the 
    static size of height and width dimension of the output tensor.

    Args:
      input_size: int scalar or None, the height or width of the input tensor.
      kernel_size: int scalar, the height or width of the kernel. 
      stride_size: int scalar, the stride along the height or width dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      
    Returns:
      out_size: int scalar or None, the height or width of the output tensor. 
    """
    if input_size is None:
      out_size = None
    else:
      if padding == 'SAME':
        out_size = input_size * stride_size
      else: # padding == 'VALID'
        if kernel_size is None:
          out_size = None
        else:
          out_size = input_size * stride_size + max(
              kernel_size - stride_size, 0)
    return out_size


class MaxPool2D(_Kernel2D):
  """Compute 2D Max Pooling."""
  def __init__(self, x, kernel_size, strides, padding, graph=None):
    """Constructor.

    Args:
      x: 4-D node of shape [batch, height, width, in_channels], input tensor.
      kernel_size: a tuple of 2 ints, the height and width of kernel.
      strides: a tuple of 2 ints, the stride along the height and width 
        dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      graph: a Graph instance.
    """
    self._check_arguments(x, None, strides, padding)

    out_height = self._compute_static_spatial_dim_size(
        x._shape._raw_shape[1], kernel_size[0], strides[0], padding)
    out_width = self._compute_static_spatial_dim_size(
        x._shape._raw_shape[2], kernel_size[1], strides[1], padding)

    shape = (x._shape._raw_shape[0], out_height, 
        out_width, x._shape._raw_shape[3])
    super(MaxPool2D, self).__init__(shape, strides, padding, graph)
    self._kernel_size = kernel_size
    self._pad_value = np.nan
    self._arguments['x'] = x

  @property
  def input_channels_dim(self):
    return 2

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)

    out_height, out_width, _, _ = self._get_shapes(
        x_val.shape, self._kernel_size[0], self._kernel_size[1])
    batch, in_channels = x_val.shape[0], x_val.shape[3]

    x_val_mat = self._get_x_tensor2patchmat(x_val, 
                                            self._kernel_size[0], 
                                            self._kernel_size[1], 
                                            flat_batch=True, 
                                            in_channels=in_channels)
    argmax = self._get_argmax(x_val_mat)

    maxpool_val = x_val_mat[np.arange(x_val_mat.shape[0]), argmax].reshape((
        out_height, out_width, batch, in_channels)).transpose(2, 0, 1, 3)
    return maxpool_val

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

    out_height, out_width, _, _ = self._get_shapes(
        x_val.shape, self._kernel_size[0], self._kernel_size[1])
    batch, in_channels = x_val.shape[0], x_val.shape[3]

    x_val_mat = self._get_x_tensor2patchmat(x_val, 
                                            self._kernel_size[0], 
                                            self._kernel_size[1], 
                                            flat_batch=True, 
                                            in_channels=in_channels)
    argmax = self._get_argmax(x_val_mat)
    ind_mat = np.zeros_like(x_val_mat, dtype=np.float32)
    ind_mat[np.arange(ind_mat.shape[0]), argmax] = 1
    
    grad_val = self._graph.get_runtime()._bwval[self.name]  
    grad_val_mat = np.tile(grad_val.transpose(1, 2, 0, 3).reshape((-1, 1)), 
        (1, ind_mat.shape[1]))

    dx_val_mat = ind_mat * grad_val_mat
    dx_val_mat = dx_val_mat.reshape((out_height * out_width * batch, 
        in_channels * self._kernel_size[0] * self._kernel_size[1]))
    dx_val = self._dx_patchmat2tensor(
        dx_val_mat, x_val.shape, self._kernel_size[0], self._kernel_size[1])
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict


  def _get_argmax(self, x_val_mat):
    """Get the argmax index of each patch of the input tensor from which the
    maximum value is taken.

    Args:
      x_val_mat: 2-D numpy array of shape [out_height * out_width * batch 
        * in_channels, kernel_height * kernel_width], the input tensor in
        "patch matrix" format. See the `_get_x_tensor2patchmat()` method of
        class `_Kernel2D`.

    Returns:
      1-D numpy array of shape [out_height * out_width * batch * in_channels],
        holding the flattend indices of the element with maximum value in each
        patch. 
    """
    if 'argmax' not in self._graph.get_runtime()._cache_data[self.name]:
      self._graph.get_runtime()._cache_data[self.name]['argmax'] = np.nanargmax(
          x_val_mat, axis=1)
    return self._graph.get_runtime()._cache_data[self.name]['argmax']


class AvgPool2D(_Kernel2D):
  """Compute 2D Average Pooling."""
  def __init__(self, x, kernel_size, strides, padding, graph=None):
    """Constructor.

    Args:
      x: 4-D node of shape [batch, height, width, in_channels], input tensor.
      kernel_size: a tuple of 2 ints, the height and width of kernel.
      strides: a tuple of 2 ints, the stride along the height and width 
        dimension.
      padding: string scalar, the padding scheme ('SAME' or 'VALID').
      graph: a Graph instance.
    """
    self._check_arguments(x, None, strides, padding)
    out_height = self._compute_static_spatial_dim_size(
        x._shape._raw_shape[1], kernel_size[0], strides[0], padding)
    out_width = self._compute_static_spatial_dim_size(
        x._shape._raw_shape[2], kernel_size[1], strides[1], padding)

    shape = (x._shape._raw_shape[0], out_height, 
        out_width, x._shape._raw_shape[3])
    super(AvgPool2D, self).__init__(shape, strides, padding, graph)
    self._kernel_size = kernel_size    
    self._pad_value = np.nan 
    self._arguments['x'] = x

  @property
  def input_channels_dim(self):
    return 2

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    
    out_height, out_width, _, _ = self._get_shapes(
        x_val.shape, self._kernel_size[0], self._kernel_size[1])
    batch, in_channels = x_val.shape[0], x_val.shape[3]

    x_val_mat = self._get_x_tensor2patchmat(x_val, 
                                            self._kernel_size[0], 
                                            self._kernel_size[1], 
                                            flat_batch=True, 
                                            in_channels=in_channels)
    x_val_mat = np.nanmean(x_val_mat, axis=1, dtype=np.float32)
    avgpool_val = x_val_mat.reshape((out_height, out_width, batch, in_channels
        )).transpose(2, 0, 1, 3)
    return avgpool_val

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

    out_height, out_width, _, _ = self._get_shapes(
        x_val.shape, self._kernel_size[0], self._kernel_size[1])
    batch, in_channels = x_val.shape[0], x_val.shape[3]

    x_val_mat = self._get_x_tensor2patchmat(x_val, 
                                            self._kernel_size[0], 
                                            self._kernel_size[1], 
                                            flat_batch=True, 
                                            in_channels=in_channels)
    divisor = np.logical_not(np.isnan(x_val_mat)).astype(np.float32).sum(
        axis=1, keepdims=True)
    ind_mat = np.ones_like(x_val_mat, dtype=np.float32) / divisor

    grad_val = self._graph.get_runtime()._bwval[self.name]
    grad_val_mat = np.tile(grad_val.transpose(1, 2, 0, 3).reshape((-1, 1)), 
        (1, ind_mat.shape[1]))

    dx_val_mat = ind_mat * grad_val_mat
    dx_val_mat = dx_val_mat.reshape((
        out_height * out_width * batch, 
        in_channels * self._kernel_size[0] * self._kernel_size[1]))
    dx_val = self._dx_patchmat2tensor(
        dx_val_mat, x_val.shape, self._kernel_size[0], self._kernel_size[1])
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict


class L2Norm(base_node.Node):
  """Compute the L2 norm of a tensor."""
  def __init__(self, x, scalar, graph=None):
    """Constructor.

    Args:
      x: a Node instance, the input tensor whose L2 norm is to be computed.
      scalar: int scalar, the integer to scale the L2 norm.
      graph: a Graph instance.
    """
    super(L2Norm, self).__init__((), graph)
    self._scalar = scalar
    self._arguments['x'] = x

  def _forward(self, feed_dict):
    """Compute the forward pass value of the node.

    Args: 
      feed_dict: a dict mapping from a `Node` instance to a numpy array.

    Returns:
      the forward pass value of the node.
    """
    x_val = self._arguments['x'].forward(feed_dict)
    return self._scalar * np.sum(x_val * x_val) * .5

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
    dx_val = self._scalar * grad_val * x_val
    grad_dict = {self._arguments['x']: dx_val}
    return grad_dict
