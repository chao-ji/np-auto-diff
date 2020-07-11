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
from autodiff.nodes import neural_network
from autodiff.nodes import origin_nodes
from autodiff.nodes import arithmetic
from autodiff.nodes import array_ops
from autodiff.nodes import losses

from autodiff.core.environments import get_default_graph


def placeholder(shape):
  """Create a placeholder node.

  Args:
    shape: raw shape, i.e. a list (or tuple) of integers (or None), or a 
      `TensorShape` instance.

  Returns:
    a Placeholder node of shape `shape`.
  """
  graph = get_default_graph()
  return origin_nodes.Placeholder(shape=shape, graph=graph)


def variable(shape, initializer, trainable=True):
  """Creates a variable node.

  Args:
    shape: raw shape, i.e. a list (or tuple) of integers (or None), or a 
      `TensorShape` instance.
    initializer: an Initializer instance.
    trainable: bool scalar, whether the variable is trainable.

  Returns:
    a Variable node.
  """
  graph = get_default_graph()
  return origin_nodes.Variable(shape=shape, 
                               initializer=initializer, 
                               trainable=trainable, 
                               graph=graph)


def constant(val):
  """Creates a constant node.

  Args:
    val: any numeric type convertable to numpy array.

  Returns:
    a Constant node.
  """
  graph = get_default_graph()
  return origin_nodes.Constant(val=val, graph=graph)


def add(x, y):
  """Performs elementwise addition operation.

  Args:
    x: a Node instance, the first operand.
    y: a Node instance, the second operand.

  Returns:
    an Add node of shape (x + y).shape, holding the result of addtion.
  """
  graph = get_default_graph()
  return arithmetic.Add(x=x, y=y, graph=graph)


def subtract(x, y):
  """Performs elementwise subtraction operation.

  Args:
    x: a Node instance, the first operand.
    y: a Node instance, the second operand.

  Returns:
    an Add node of shape (x - y).shape, holding the result of subtraction.
  """
  graph = get_default_graph()
  return arithmetic.Subtract(x=x, y=y, graph=graph)


def exp(x):
  """Computes elementwise exp(x) functions.

  Args:
    x: a Node instance, the input tensor.

  Returns:
    an Exp node of shape x.shape, holding the result of exp(x).
  """
  graph = get_default_graph()
  return arithmetic.Exp(x=x, graph=graph)


def multiply(x, y):
  """Performs multiply operation.

  Args:
    x: a Node instance, the first operand.
    y: a Node instance, the second operand.

  Returns:
    an Add node of shape (x * y).shape, holding the result of multiply.
  """
  graph = get_default_graph()
  return arithmetic.Multiply(x=x, y=y, graph=graph)


def matmul(x, y):
  """Performs dot product between two 2D arrays (i.e. matrix). 

  Args:
    x: a 2-D Node instance, the first operand.
    y: a 2-D Node instance, the second operand.

  Returns:
    a MatMul node of shape (x.shape[0], y.shape[1]), holding the result of 
      matrix multiplication.
  """
  graph = get_default_graph()
  return array_ops.MatMul(x=x, y=y, graph=graph)


def reshape(x, target_shape):
  """Performs reshaping of input node.

  Args:
    x: a Node instance, the tensor to be reshaped.
    target_shape: a list (or tuple) of integers >= -1, the target shape that
      the input tensor will be reshaped to. Up to one entry can be -1, and 
      will be inferred based on the shape of input tensor.

  Returns:
    a Reshape Node of shape `target_shape`, holding the result of reshaping.
  """
  graph = get_default_graph()
  return array_ops.Reshape(x=x, target_shape=target_shape, graph=graph)


def reduce_mean(x, axis=None):
  """Compute the mean of input Node across axes.

  Args:
    x: a Node instance, the input tensor on which the Reduction operation is 
      applied.
    axis: a list (or tuple) of ints, or an int scalar. Defaults to all 
      dimensions, i.e. `range(x.shape.ndims)`.

  Returns:
    a ReduceMean node of shape `reduced_shape`, where `reduced_shape` holds 
      axis indices not present in `axis`.
  """
  graph = get_default_graph()
  return array_ops.ReduceMean(x=x, axis=axis, graph=graph)


def reduce_sum(x, axis=None):
  """Compute the sum of input Node across axes.

  Args:
    x: a Node instance, the input tensor on which the Reduction operation is 
      applied.
    axis: a list (or tuple) of ints, or an int scalar. Defaults to all 
      dimensions, i.e. `range(x.shape.ndims)`.

  Returns:
    a ReduceSum Node of shape `reduced_shape`, where `reduced_shape` holds 
      axis indices not present in `axis`.
  """
  graph = get_default_graph()
  return array_ops.ReduceSum(x=x, axis=axis, graph=graph)


def pad(x, paddings, constant_value=0):
  """Performs padding on input Node.

  Args:
    x: a Node instance, the input tensor to be padded. 
    paddings: a 2-D numpy array of shape, [x.shape.ndims, 2]. `paddings[i, 0]`
      indicates the num of values *prepended* to the contents of `x`, while
      `paddings[i, 1]` indicates the num of values *appended* to the contents
      of `x`.
    constant_value: int scalar, the value to padded to input tensor. Defaults
      to 0.    

  Returns:
    a Pad Node of shape `x.shape + np.sum(paddings, axis=1)`.
  """
  graph = get_default_graph()
  return array_ops.Pad(
      x=x, paddings=paddings, constant_value=constant_value, graph=graph)


def concat(tensors, axis=0):
  """Concatenate tensors of the same rank along one of the dimensions. 

  Suppose `tensors[i]` has shape [s_0, s_1, ..., s_axis_i, ...]

  The output Node has shape [s_0, s_1, ..., sum(s_axis_i), ...].

  Args:
    tensors: a list (>= 2) of Node instances, holding tensors of the same rank
      (i.e. num of dimension). They must have the same size in all dimension 
      except for the axis `axis`.
    axis: int scalar, the axis that `tensors` will be concatenated along. 

  Returns:
    a Concat Node.
  """
  graph = get_default_graph()
  return array_ops.Concat(tensors=tensors, axis=axis, graph=graph)


def slice(x, begin, sizes):
  """Slice out a sub-tensor from input tensor.

  Equivalent to Numpy syntax if `x` were Numpy array:

  x[begin[0]: begin[0] + sizes[0], ..., begin[-1]: begin[-1] + sizes[-1]]

  For each dimension `i`, we must have

  0 <= begin[i] <= begin[i] + sizes[i] < x.shape[i]

  Args:
    x: a Node instance, the input tensor to be sliced.
    begin: a Node instance of rank 1, or a list or tuple of ints of length
      x.shape.ndims (i.e. rank of `x`).
    sizes: a Node instance of rank 1, or a list or tuple of ints of length
      x.shape.ndims (i.e. rank of `x`).    

  Returns:
    a Slice Node of shape `sizes`.
  """
  graph = get_default_graph()
  return array_ops.Slice(x, begin, sizes, graph=graph)


def sigmoid(x):
  """Computes elementwise sigmoid of input Node.

  Args:
    x: a Node instance, the input tensor whose elementwise sigmiod is to be 
      computed.

  Returns:
    a Sigmoid Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.Sigmoid(x=x, graph=graph)


def tanh(x):
  """Computes elementwise tanh of input Node.

  Args:
    x: a Node instance, the input tensor whose elementwise tanh is to be 
      computed.

  Returns:
    a Tanh Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.Tanh(x=x, graph=graph)


def relu(x):
  """Computes elementwise ReLU of input Node.

  Args:
    x: a Node instance, the input tensor whose elementwise ReLU is to be 
      computed.

  Returns:
    a ReLU Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.ReLU(x=x, graph=graph)


def leaky_relu(x, alpha):
  """Computes elementwise ReLU of input Node.

  Args:
    x: a Node instance, the input tensor whose elementwise ReLU is to be 
      computed.
    alpha: int scalar, slope of the input in the negative region.

  Returns:
    a LeakyReLU Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.LeakyReLU(x=x, alpha=alpha, graph=graph)


def dropout(x, rate, is_training=True):
  """Computes elementwise dropout of input Node.

  In training mode, some random set of elements are zeroed with probability 
  `rate`, while those that are not zeroed are scaled up by `1 / (1 - rate)`. 
  In test mode, all elements are output as is.

  Args:
    x: a Node instance, the input tensor.
    rate: float scalar, the probability that an element is dropped.
    is_training: bool scalar, whether dropout is in training mode.

  Returns:
    a Dropout Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.Dropout(
      x=x, rate=rate, is_training=is_training, graph=graph)


def fused_batch_norm(x,
                     scale,
                     offset,
                     moving_mean,
                     moving_variance,
                     epsilon=0.001,
                     decay=0.999,
                     is_training=True):
  """Fused Batch Normalization where the batch statistics is computed internally
  as opposed to being computed externally and then passed in.

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

  Returns:
    a FusedBatchNorm Node of shape `x.shape`.
  """
  graph = get_default_graph()
  return neural_network.FusedBatchNorm(x=x,
                                       scale=scale,
                                       offset=offset,
                                       moving_mean=moving_mean,
                                       moving_variance=moving_variance,
                                       epsilon=epsilon,
                                       decay=decay,
                                       is_training=is_training,
                                       graph=graph)


def conv2d(x, kernel, strides, padding):
  """Compute 2D Convolution.

  Args:
    x: 4-D node of shape [batch, height, width, in_channels], input tensor.
    kernel: 4-D node of shape [kernel_height, kernel_width, in_channels, 
      out_channels], the kernel.
    strides: a tuple of 2 ints, the stride along the height and width 
      dimension.
    padding: string scalar, the padding scheme ('SAME' or 'VALID').

  Returns:
    a Conv2D node of shape [batch, height_out, width_out, out_channels].
  """
  graph = get_default_graph()
  return neural_network.Conv2D(
      x=x, kernel=kernel, strides=strides, padding=padding, graph=graph)


def conv2d_transpose(x, kernel, strides, padding):
  """Compute 2D Transposed Convolution.

  Note: the `in_channels` and `out_channels` corresponds to the input and
  output channels of the UN-TRANSPOSED version of Conv2D. So the last dim of
  `x` has size `out_channels`.

  Args:
    x: 4-D node of shape [batch, height, width, out_channels], input tensor.
    kernel: 4-D node of shape [kernel_height, kernel_width, in_channels, 
      out_channels], the kernel.
    strides: a tuple of 2 ints, the stride along the height and width 
      dimension.
    padding: string scalar, the padding scheme ('SAME' or 'VALID').
 
  Returns:
    a Conv2DTranspose node of shape 
      [batch, height_out, width_out, out_channels].
  """
  graph = get_default_graph()
  return neural_network.Conv2DTranspose(
      x=x, kernel=kernel, strides=strides, padding=padding, graph=graph)


def maxpool2d(x, kernel_size, strides, padding):
  """Compute 2D Max Pooling.

  Args:
    x: 4-D node of shape [batch, height, width, in_channels], input tensor.
    kernel_size: a tuple of 2 ints, the height and width of kernel.
    strides: a tuple of 2 ints, the stride along the height and width 
      dimension.
    padding: string scalar, the padding scheme ('SAME' or 'VALID').

  Returns:
    a MaxPool2D node of shape [batch, height_out, width_out, in_channels].
  """
  graph = get_default_graph()
  return neural_network.MaxPool2D(x=x, 
                                  kernel_size=kernel_size, 
                                  strides=strides, 
                                  padding=padding, 
                                  graph=graph)


def avgpool2d(x, kernel_size, strides, padding):
  """Compute 2D Average Pooling.

  Args:
    x: 4-D node of shape [batch, height, width, in_channels], input tensor.
    kernel_size: a tuple of 2 ints, the height and width of kernel.
    strides: a tuple of 2 ints, the stride along the height and width 
      dimension.
    padding: string scalar, the padding scheme ('SAME' or 'VALID').

  Returns:
    a AvgPool2D node of shape [batch, height_out, width_out, in_channels].    
  """
  graph = get_default_graph()
  return neural_network.AvgPool2D(x=x, 
                                  kernel_size=kernel_size, 
                                  strides=strides, 
                                  padding=padding, 
                                  graph=graph)


def l2norm(x, scalar):
  """Compute the L2 norm of a tensor.

  Args:
    x: a Node instance, the input tensor whose L2 norm is to be computed.
    scalar: int scalar, the integer to scale the L2 norm.

  Returns:
    a L2Norm node holding a scalar tensor.
  """
  graph = get_default_graph()
  return neural_network.L2Norm(x=x, scalar=scalar, graph=graph)


def softmax_cross_entropy_loss(labels, logits):
  """Computes the elementwise softmax cross entropy.

  Note: this implementation does not compute gradient w.r.t `lables`.
  
  Args:
    labels: a Node instance, the tensor holding the groundtruth class labels.
      The last dimension is treated as the class dimension.
    logits: a Node instance of same shape as `labels`, the tensor holding the
      logits. The last dimension is treated as the class dimension.

  Returns:
    a SoftmaxCrossEntropyLoss node of shape `labels.shape[:-1]`.
  """
  graph = get_default_graph()
  return losses.SoftmaxCrossEntropyLoss(
      labels=labels, logits=logits, graph=graph)


def sigmoid_cross_entropy_loss(labels, logits):
  """Computes the elementwise sigmoid cross entropy.

  Note: this implementation does not compute gradient w.r.t `lables`.

  Args:
    labels: a Node instance, the tensor holding the groundtruth class labels.
    logits: a Node instance of same shape as `labels`, the tensor holding the
      logits. 

  Returns:
    a SigmoidCrossEntropyLoss node of shape `labels.shape`.
  """
  graph = get_default_graph()
  return losses.SigmoidCrossEntropyLoss(
      labels=labels, logits=logits, graph=graph)
