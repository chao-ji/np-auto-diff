import numpy as np

class _BaseOp(object):
  """Base class of all Op classes. An Op emits a tensor of a specific shape. Initializes the 
     following attributes of an Op:

    sess: `Session`;
      The session in which the Op is defined
    shape: `numpy.ndarray`;
      1D array, e.g. [2, 3, 3, 2], specifying the shape of the emitted tensor
    parent_total: integer;
      Total number of Op's for wihch the current Op is an argument; this is determined when
      the data flow graph was defined in the beginning
    parent_acc: integer;
      Initialized to zero; it keeps track of the number of parent Op's that have backpropped
      gradients to the current Op in an iteration
    is_terminal: bool;
      Initialized to False; indicates if the Op is terminal node (i.e. has no child node)

    Parameters
    ----------
    sess: `Session`;
      The session in which the Op is defined
  """  
  def __init__(self, sess):
    self.sess = sess
    self.shape = () 
    self.parent_total = 0
    self.parent_acc = 0
    self.is_terminal = False 
    self.sess.variables.append(self)

  def eval(self, feed_dict):
    """Evaluate the current Op.

    `_eval_func` is implemented by each Op separately to compute the tensor value.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of the tensor
    """
    if id(self) not in self.sess.values:
      self.sess.values[id(self)] = self._eval_func(feed_dict)
    return self.sess.values[id(self)]

  def grad(self, feed_dict, backprop):
    """Update the gradient w.r.t. the current Op (`backprop`), and propagate gradient down to 
    child Op's. `_grad_func` is implemented by each Op separately to propagate gradient down to 
    child Op's. 

    NOTE: `grad` is invoked when a parent Op propagates gradient (`backprop`) back to the current
    Op. When `grad` is invoked, the gradient is accumulated and `parent_acc` is incremented, which
    maintains the number of parent Op's that have already backpropped gradients. The computation of
    the gradient w.r.t. the current Op is finished when `parent_acc` == `parent_total`, and this 
    gradient is further propagated down to child Op's of the current Op by invoking 
    `self._grad_func(feed_dict)`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    backprop: `numpy.ndarray`;
      Gradient backpropped from a parent Op. Has the SAME shape as the shape of the current Op
      (i.e. `self.shape`)
    """
    if self.is_terminal and not self.is_variable:
      return

    if id(self) not in self.sess.gradients:
      self.sess.gradients[id(self)] = np.zeros(self.shape)
    self.sess.gradients[id(self)] += backprop
    self.parent_acc += 1

    if self.parent_acc == self.parent_total and not self.is_terminal:
        self._grad_func(feed_dict)


class PlaceholderOp(_BaseOp):
  """Creates placeholder for input or parameters.
  NOTE: placeholders must be terminal nodes of the data flow graph, so it does not implement
  the `_grad_func()` method.

  Parameters
  ----------
  shape: `numpy.array`;
    1D array specifying shape of tensor
  sess: `Session`;
    The session in which the Op is defined
  is_variable: bool;
    Indicates if the placeholder holds trainable parameters or not
  """ 
  def __init__(self, shape, sess, is_variable=True):
    super(PlaceholderOp, self).__init__(sess)
    self.is_terminal = True
    self.is_variable = is_variable
    self.shape = shape

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of the tensor
    """
    return feed_dict[self]
    

class SigmoidOp(_BaseOp):
  """Op that applies sigmoid function componentwise to the input `X`.

  Parameters
  ----------
  X: `_BaseOp`;
    The input tensor
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X, sess):
    super(SigmoidOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of the tensor
    """
    X_val = self.X.eval(feed_dict)
    return 1. / (1. + np.exp(-X_val))

  def _grad_func(self, feed_dict):
    """Propagate gradient down to child `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    val = self.eval(feed_dict)
    val = val * (1. - val)
    grad_val = self.sess.gradients[id(self)]
    dX_val = grad_val * val
    self.X.grad(feed_dict, dX_val)


class ReluOp(_BaseOp):
  """Op that appliesi Rectifier Linear Unit (i.e. ReLU) componentwise to the input `X`.

  Parameters
  ----------
  X: `_BaseOp`;
    The input tensor 
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X, sess):
    super(ReluOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of the tensor
    """
    X_val = self.X.eval(feed_dict)
    return np.maximum(0, X_val)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to child `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    X_val = self.X.eval(feed_dict)
    grad_val = self.sess.gradients[id(self)]
    grad_val[X_val <= 0] = 0
    dX_val = grad_val
    self.X.grad(feed_dict, dX_val)


class ParamRegOp(_BaseOp):
  """Op that computes the regularization term over the parameter tensor `W`. Specifically, it 
  computes `0.5 * reg * sum(|W|^2)`.
  
  Parameters
  ----------
  W: `_BaseOp`;
    The input tensor containing parameters
  reg: float;
    Float between 0.0 and 1.0, controls the strength of regularization
  sess: `Session`;
    The session in which the Op is defined
  """  
  def __init__(self, W, reg, sess):
    super(ParamRegOp, self).__init__(sess)
    self.shape = 1, 1
    self.W = W
    self.W.parent_total += 1
    self.reg = reg

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`: value of tensor
    """
    W_val = self.W.eval(feed_dict)
    return .5 * self.reg * np.sum(W_val * W_val)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `W`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    W_val = self.W.eval(feed_dict)
    grad_val = self.sess.gradients[id(self)]
    dW_val = self.reg * grad_val * W_val
    self.W.grad(feed_dict, dW_val)


class AddOp(_BaseOp):
  """Op that computes the sum of two input tensors `X1` and `X2`.
  NOTE: `X1` and `X2` must have the same shape.
  
  Parameters
  ----------
  X1: `_BaseOp`;
    Input tensor
  X2: `_BaseOp`;
    Input tensor
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X1, X2, sess):
    super(AddOp, self).__init__(sess)
    self.shape = X1.shape
    self.X1 = X1
    self.X2 = X2
    self.X1.parent_total += 1
    self.X2.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`: value of tensor
    """  
    return self.X1.eval(feed_dict) + self.X2.eval(feed_dict)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X1` and `X2`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    self.X1.grad(feed_dict, grad_val)
    self.X2.grad(feed_dict, grad_val)


class MatMulOp(_BaseOp):
  """Op that computes the matrix multiplication of input tensors `X1` and `X2`.
  NOTE: the shapes of `X1` and `X2` must be compatible.

  Parameters
  ----------
  X1: `_BaseOp`;
    Input 2D tensor, matrix
  X2: `_BaseOp`;
    Input 2D tensor, matrix
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X1, X2, sess):
    super(MatMulOp, self).__init__(sess)
    self.shape = X1.shape[0], X2.shape[1]
    self.X1 = X1
    self.X2 = X2
    self.X1.parent_total += 1
    self.X2.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of tensor
    """
    return np.dot(self.X1.eval(feed_dict), self.X2.eval(feed_dict))

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X1` and `X2`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    X1_val = self.X1.eval(feed_dict)
    X2_val = self.X2.eval(feed_dict)
    dX1_val = np.dot(grad_val, X2_val.T)
    dX2_val = np.dot(X1_val.T, grad_val)
    self.X1.grad(feed_dict, dX1_val)
    self.X2.grad(feed_dict, dX2_val)


class _2dKernelOp(_BaseOp):
  """Base class of 2D filter Op's, i.e. `Conv2dOp` and `MaxPool2dOp`.

  It provides methods for 
    1. Padding zeros (`_pad_X()`) for input 4D tensor;
    2. Removing padded zeros (`_depad_dX()`) for input 4D tensor;
    3. Computing the height and width coordinates of the upper left pixel of all image patches 
      (`_img_col_index()`);
    4. Converting input 4D tensor into the format of "patch matrix" (`_X_tensor2patchmat()`);
    5. Converting the gradient w.r.t. input tensor `X` in the format of "patch matrix" to a 4D 
      tensor with the same shape as `X` (`_dX_patchmat2tensor()`).

  Parameters
  ----------
  X: `_BaseOp`;
    Input 4D tensor, of dimensions in [batch, in_height, in_width, in_channels]
  fsize: `numpy.ndarray`;
    1D array of length 2, specifying filter sizes along height and width axes, e.g. [3, 3]
  strides: `numpy.ndarray`;
    1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
  padding: `numpy.ndarray`;
    1D array of length 2, specifying total zero padding along height and width axes, e.g. [2, 2]
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X, fsize, strides, padding, sess):
    if (X.shape[1] - fsize[0] + padding[0]) % strides[0] != 0 or (X.shape[2] - fsize[1] + \
        padding[1]) % strides[1] != 0:
      error_string = "Mismatch between geometric hyperparameters!"
      error_string += "\nxsize[0] - fsize[0] + padding[0] = %d" % (X.shape[1] - fsize[0] + \
        padding[0])
      error_string += "\nstrides[0] = %d" % strides[0]
      error_string += "\nxsize[1] - fsize[1] + padding[1] = %d" % (X.shape[2] - fsize[1] + \
        padding[1])
      error_string += "\nstrides[1] = %d" % strides[1]
      raise ValueError(error_string)

    super(_2dKernelOp, self).__init__(sess)
    self.X = X
    self.X.parent_total += 1
    self.fsize = fsize
    self.strides = strides
    self.padding = padding

    self._X_shape_pad = X.shape[0], X.shape[1] + padding[0], X.shape[2] + padding[1], X.shape[3]
    self._padding = self.padding[0] // 2, self.padding[0] - self.padding[0] // 2, \
                    self.padding[1] // 2, self.padding[1] - self.padding[1] // 2
    self._out_height = (X.shape[1] - fsize[0] + padding[0]) / strides[0] + 1
    self._out_width = (X.shape[2] - fsize[1] + padding[1]) / strides[1] + 1
    self._img_col_index_val = self._img_col_index()

  def _pad_X(self, feed_dict):
    """Pad a 4D tensor with zeros (or specified value) along `in_height` and `in_width` axes.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; the input 4D array with zero padded
    """
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad
    pad_top, pad_bot, pad_left, pad_right = self._padding

    X_pad = np.ones(self._X_shape_pad) * self._pv if hasattr(self, "_pv") \
            else np.zeros(self._X_shape_pad)
    X_pad[:,
        pad_top : in_height_pad - pad_bot,
        pad_left : in_width_pad - pad_right, :] = self.X.eval(feed_dict)
    return X_pad

  def _depad_dX(self, dX):
    """Remove padded zeros in a 4D tensor.

    Parameters
    ----------
    dX: `numpy.ndarray`;
      4D array containing gradients w.r.t. zero-padded `X`

    Returns
    -------
    `numpy.ndarray`; 4D array containing the gradient w.r.t. the input array `X` (`dX`) where padded
     zeros are removed
    """
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad
    pad_top, pad_bot, pad_left, pad_right = self._padding

    dX = dX[:,
          pad_top : in_height_pad - pad_bot,
          pad_left : in_width_pad - pad_right, :]
    return dX

  def _img_col_index(self):
    """Compute the height and width coordinates of the upper left pixel of all image patches.

    Returns
    -------
    `numpy.ndarray`; 2D array containing 4-tuples (h, w, h_idx, w_idx), where `h` and `w` 
    correspond to the height and width indexes of the upper left pixel of each patch within the 
    input 2D image, and `h_idx` and `w_idx` the height and width indexes of each patch within the
    output 2D image.

    Given 4px-by-4px image and 2px-by-2px filter with 2px strides along height and width
    X[0,0]  X[0,1]  X[0,2]  X[0,3]
    X[1,0]  X[1,1]  X[1,2]  X[1,3]
    X[2,0]  X[2,1]  X[2,2]  X[2,3]
    X[3,0]  X[3,1]  X[3,2]  X[3,3]

    [(h, w, h_idx, w_idx)] = 
      [(0,  0,  0,  0),
       (0,  2,  0,  1),
       (2,  0,  1,  0),
       (2,  2,  1,  1)]
    """
    filter_height, filter_width = self.fsize
    stride_height, stride_width = self.strides
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad

    img_col_index = np.array([(h, w, h_idx, w_idx)
                      for h_idx, h in enumerate(np.arange(0, in_height_pad - filter_height + 1,
                        stride_height))
                      for w_idx, w in enumerate(np.arange(0, in_width_pad - filter_width + 1,
                        stride_width))])
    return img_col_index

  def _X_tensor2patchmat(self, feed_dict):
    """Convert input 4D tensor into 2D tensor in the "patch matrix" format.

    Input 4D tensor `X` has dimensions [batch, in_height, in_width, in_channels] = [2, 3, 3, 2]

    X[0,0,0,0]  X[0,0,1,0]  X[0,0,2,0]      X[0,0,0,1]  X[0,0,1,1]  X[0,0,2,1]
    X[0,1,0,0]  X[0,1,1,0]  X[0,1,2,0]      X[0,1,0,1]  X[0,1,1,1]  X[0,1,2,1]
    X[0,2,0,0]  X[0,2,1,0]  X[0,2,2,0]      X[0,2,0,1]  X[0,2,1,1]  X[0,2,2,1]
    
    X[1,0,0,0]  X[1,0,1,0]  X[1,0,2,0]      X[1,0,0,1]  X[1,0,1,1]  X[1,0,2,1]
    X[1,1,0,0]  X[1,1,1,0]  X[1,1,2,0]      X[1,1,0,1]  X[1,1,1,1]  X[1,1,2,1]
    X[1,2,0,0]  X[1,2,1,0]  X[1,2,2,0]      X[1,2,0,1]  X[1,2,1,1]  X[1,2,2,1]

    Each 3px-by-3px submatrix corresponds to an image of dimensions [in_height, in_width], and the 
    four smaller submatrixes form a 2-by-2 "matrix" where the rows corresponds to `batch` and 
    columns to `in_channels`.

    Given geometric hyperparameters `filter_height`=2 and `filter_width`=2, `X` is converted 
    into 2D array in the "patch matrix" format of dimensions [out_height * out_width * batch, 
    in_channels * filter_height * filter_width] where `out_height`=2 and `out_width`=2.

    X[0,0,0,0]  X[0,0,1,0]  X[0,1,0,0]  X[0,1,1,0]  X[0,0,0,1]  X[0,0,1,1]  X[0,1,0,1]  X[0,1,1,1]
    X[1,0,0,0]  X[1,0,1,0]  X[1,1,0,0]  X[1,1,1,0]  X[1,0,0,1]  X[1,0,1,1]  X[1,1,0,1]  X[1,1,1,1]

    X[0,0,1,0]  X[0,0,2,0]  X[0,1,1,0]  X[0,1,2,0]  X[0,0,1,1]  X[0,0,2,1]  X[0,1,1,1]  X[0,1,2,1]
    X[1,0,1,0]  X[1,0,2,0]  X[1,1,1,0]  X[1,1,2,0]  X[1,0,1,1]  X[1,0,2,1]  X[1,1,1,1]  X[1,1,2,1]

    X[0,1,0,0]  X[0,1,1,0]  X[0,2,0,0]  X[0,2,1,0]  X[0,1,0,1]  X[0,1,1,1]  X[0,2,0,1]  X[0,2,1,1]
    X[1,1,0,0]  X[1,1,1,0]  X[1,2,0,0]  X[1,2,1,0]  X[1,1,0,1]  X[1,1,1,1]  X[1,2,0,1]  X[1,2,1,1]

    X[0,1,1,0]  X[0,1,2,0]  X[0,2,1,0]  X[0,2,2,0]  X[0,1,1,1]  X[0,1,2,1]  X[0,2,1,1]  X[0,2,2,1]
    X[1,1,1,0]  X[1,1,2,0]  X[1,2,1,0]  X[1,2,2,0]  X[1,1,1,1]  X[1,1,2,1]  X[1,2,1,1]  X[1,2,2,1]

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray` 2D array in the "patch matrix" format
    """
    X_val = self._pad_X(feed_dict)
    filter_height, filter_width = self.fsize
    batch = self.X.shape[0]

    X_val_mat = np.vstack([X_val[:, h:h+filter_height, w:w+filter_width, :]
              .transpose(0, 3, 1, 2).reshape((batch, -1))
                for h, w, _, _ in self._img_col_index_val])
    return X_val_mat

  def _dX_patchmat2tensor(self, dX_val_mat):
    """Convert the gradient w.r.t. the "patch matrix" into 4D tensor with the same shape as `X`.

    1. `dX_val_mat` of dimensions [out_height * out_width * batch, in_channels * filter_height *
      filter_width] is reshaped into 6D array of dimensions [out_height, out_width, batch, 
      in_channels, filter_height, filter_width];
    2. The dimensions in the 6D array is reordered as [out_height, out_width, batch, filter_height,
      filter_width, in_channels], as in `dX_val_tmp`;
    3. Initialize a zero tensor of shape `self._X_shape_pad`, as in `dX_val`;
    4. The gradient in each patch of `dX_val_tmp` (of dimensions [batch, filter_height, 
    filter_width, in_channels]) is added to the corresponding subarray in `dX_val`;
    5. Remove padded zeros in `dX_val`.

    Parameters
    ----------
    dX_val_mat: `numpy.ndarray`
      2D array, the patch matrix, of dimensions [out_height * out_width * batch, 
      in_channels * filter_height * filter_width]

    Returns
    -------
    `numpy.ndarray`; 4D tensor, of dimensions [batch, in_height, in_width, in_channels]
    """
    filter_height, filter_width = self.fsize
    batch, out_height, out_width, _ = self.shape
    in_channels = self.X.shape[3]

    dX_val_tmp = dX_val_mat.reshape((out_height, out_width, batch, in_channels, filter_height,\
                  filter_width)).transpose(0, 1, 2, 4, 5, 3)
    dX_val = np.zeros(self._X_shape_pad)
    for h, w, h_idx, w_idx in self._img_col_index_val:
      dX_val[:, h:h+filter_height, w:w+filter_width, :] += dX_val_tmp[h_idx, w_idx]
    dX_val = self._depad_dX(dX_val)

    return dX_val


class Conv2dOp(_2dKernelOp):
  def __init__(self, X, W, strides, padding, sess):
    """Op that performs 2D convolution between a 4D input tensor `X` and a 4D filter tensor `W`.

    Parameters
    ----------
      X: `_BaseOp`;
        Input 4D tensor, of dimensions in [batch, in_height, in_width, in_channels]
      W: `_BaseOp`;
        Filter 4D tensor, of dimensions in [filter_height, filter_width, in_channels, out_channels]
      strides: `numpy.ndarray`;
        1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
      padding: `numpy.ndarray`;
        1D array of length 2, specifying total zero padding along height and width axes, e.g. [2, 2]
      sess: `Session`;
        The session in which the Op is defined
    """
    fsize = W.shape[0], W.shape[1]
    super(Conv2dOp, self).__init__(X, fsize, strides, padding, sess)
    self.shape = X.shape[0], self._out_height, self._out_width, W.shape[3]
    self.W = W
    self.W.parent_total += 1

  def _W_tensor2patchmat(self, feed_dict):
    """Convert 4D filter tensor into "patch matrix" format.

    Filter 4D tensor `W` has dimensions [filter_height, filter_width, in_channels, out_channels] 
    = [2, 3, 3, 2]

    W[0,0,0,0]  W[0,1,0,0]    W[0,0,1,0]  W[0,1,1,0]
    W[1,0,0,0]  W[1,1,0,0]    W[1,0,1,0]  W[1,1,1,0]

    W[0,0,0,1]  W[0,1,0,1]    W[0,0,1,1]  W[0,1,1,1]
    W[1,0,0,1]  W[1,1,0,1]    W[1,0,1,1]  W[1,1,1,1]
    
    Each 2px-by-2px submatrix corresponds to a 2D array of dimensions [filter_height, filter_width],
    and the four smaller submatrixes form a 2-by-2 "matrix" where the rows corresponds to `out_channels`
    and columns to `in_channels`.

    `W` is converted to 2D array in the "patch matrix" format of dimensions [in_channels * 
    filter_height * filter_width, out_channels]

    W[0,0,0,0]  W[0,0,0,1]
    W[0,1,0,0]  W[0,1,0,1]
    W[1,0,0,0]  W[1,0,0,1]
    W[1,1,0,0]  W[1,1,0,1]
    W[0,0,1,0]  W[0,0,1,1]
    W[0,1,1,0]  W[0,1,1,1]
    W[1,0,1,0]  W[1,0,1,1]
    W[1,1,1,0]  W[1,1,1,1]

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 2D array, matrix
    """
    W_val = self.W.eval(feed_dict)
    out_channels = self.W.shape[3]
    W_val_mat = W_val.transpose(2, 0, 1, 3).reshape((-1, out_channels))
    return W_val_mat

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Given `X` and `W` in "patch matrix" format `X_val_mat` and `W_val_mat`, right-multiplying 
    `W_val_mat` with `X_val_mat` produces 2D array of dimensions [out_height * out_width * batch, 
    out_channels] `C_val_mat`. `C_val_mat` is then reshaped into 4D tensor of dimensions [out_height,
    out_width, batch, out_channels], which is then reordered as [batch, out_height, out_width, 
    out_channels].

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
      `numpy.ndarray`; 4D tensor of dimensions [batch, out_height, out_width, out_channels]
    """
    batch, out_height, out_width, out_channels = self.shape
    X_val_mat = self._X_tensor2patchmat(feed_dict)
    W_val_mat = self._W_tensor2patchmat(feed_dict)
    C_val_mat = np.dot(X_val_mat, W_val_mat)
    return C_val_mat.reshape((out_height, out_width, batch, out_channels)).transpose(2, 0, 1, 3)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X` and `W`.

    Given the 4D tensor `grad_val` (i.e. gradient w.r.t. Op's output) of dimensions [batch, out_height,
    out_width, out_channels], it is reordered into [out_height, out_width, batch, out_channels]
    and then reshaped into 2D array `grad_val_mat` of dimensions [out_height * out_width * batch,
    out_channels].

    The gradient w.r.t `W` in the "patch matrix" format is computed by right-multiplying `grad_val_mat`
    with `X_val_mat.T` (of dimensions [in_channels * filter_height * filter_width, out_height * 
    out_width * batch]), producing 2D array of dimensions [in_channels * filter_height * filter_width,
    out_channels].

    The gradient w.r.t. `X` in the "path matrix" format is computed by right-multiplying `W_val_mat.T`
    (of dimensions [out_channels, in_channels * filter_height * filter_width]) with `grad_val_mat`, 
    procuding 2D array of dimensions [out_height * out_width * batch, in_channels * filter_height * 
    filter_width].

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    filter_height, filter_width, in_channels, out_channels = self.W.shape

    grad_val = self.sess.gradients[id(self)]
    grad_val_mat = grad_val.transpose(1, 2, 0, 3).reshape((-1, out_channels))
    X_val_mat = self._X_tensor2patchmat(feed_dict)
    W_val_mat = self._W_tensor2patchmat(feed_dict)

    dW_val_mat = np.dot(X_val_mat.T, grad_val_mat)
    dW_val = dW_val_mat.reshape((in_channels, filter_height, filter_width, out_channels)).\
              transpose(1, 2, 0, 3)

    dX_val_mat = np.dot(grad_val_mat, W_val_mat.T)
    dX_val = self._dX_patchmat2tensor(dX_val_mat)

    self.X.grad(feed_dict, dX_val)
    self.W.grad(feed_dict, dW_val)


class MaxPool2dOp(_2dKernelOp):
  def __init__(self, X, fsize, strides, padding, sess):
    """Op that performs 2D max-pooling on a 4D tensor.

    Parameters
    ----------
      X: `_BaseOp`;
        Input 4D tensor, of dimensions in [batch, in_height, in_width, in_channels]
      fsize: `numpy.ndarray`
        1D array of length 2, specifying filter size along height and width axes, e.g. [2, 2]
      strides: `numpy.ndarray`;
        1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
      padding: `numpy.ndarray`;
        1D array of length 2, specifying total zero pading along height and width axes, e.g. [2, 2]
      sess: `Session`;
        The session that the Op is associated with
    """
    super(MaxPool2dOp, self).__init__(X, fsize, strides, padding, sess)
    self.shape = X.shape[0], self._out_height, self._out_width, X.shape[3]

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 4D tensor, the output of MaxPool(X)
    """  
    batch, out_height, out_width, in_channels = self.shape
    X_val_mat = self._X_tensor2patchmat(feed_dict)
    P_val = X_val_mat.max(axis=1).reshape((out_height, out_width, batch, in_channels)).\
            transpose(2, 0, 1, 3)
    return P_val

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    `X_val_mat` is a 2D array of dimensions [out_height * out_width * batch * in_channels, 
    filter_height * filter_width], where each row is an indicator vector that indicates the index 
    of the maximum-intensity pixel.

    Given the 4D tensor `grad_val` (i.e. gradient w.r.t. Op's output) of dimensions [batch, 
    out_height, out_width, out_channels], it is reordered into [out_height, out_width, batch, 
    out_channels] and then reshaped into 2D array of dimensions [out_height * out_width * batch 
    * out_channels, 1]. The 2D array is duplicated (`np.tile`) along the width axis by a factor of
    `X_val_mat.shape[1]` (i.e. `filter_height` * `filter_width`), producing `grad_val_mat`.

    `X_val_mat` is component-wise multiplied by `grad_val_mat`, which contains the gradient w.r.t.
    `X` in the "patch matrix" format.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    filter_height, filter_width = self.fsize
    batch, out_height, out_width, in_channels = self.shape

    X_val_mat = self._X_tensor2patchmat(feed_dict)
    X_val_mat = np.apply_along_axis(self.__ind_vec, 1, X_val_mat)

    grad_val = self.sess.gradients[id(self)]
    grad_val_mat = np.tile(grad_val.transpose(1, 2, 0, 3).reshape((-1, 1)), (1, X_val_mat.shape[1]))

    dX_val_mat = X_val_mat * grad_val_mat
    dX_val_mat = dX_val_mat.reshape((out_height * out_width * batch, in_channels * filter_height * 
                  filter_width))
    dX_val = self._dX_patchmat2tensor(dX_val_mat)

    self.X.grad(feed_dict, dX_val)

  def __ind_vec(self, row):
    """Helper that computes indicator vector that marks that maximum of input 1D vector

    For example, row = [-1, 3, 2, -3]
    output = [0, 1, 0, 0] 

    Parameters
    ----------
    row: `numpy.ndarray`;
      1D array
    """
    index = np.argmax(row)
    ind_vec = np.zeros_like(row)
    ind_vec[index] = 1.0
    return ind_vec

  def _X_tensor2patchmat(self, feed_dict):
    """Convert input 4D tensor into 2D tensor in the "patch matrix" format.

    The patch matrix is 2D array of dimensions in 
      [out_height * out_width * batch * in_channels, filter_height * filter_width]

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    X_val_mat: `numpy.ndarray`
      The output 2D tensor
    """
    batch, out_height, out_width, in_channels = self.shape
    filter_height, filter_width = self.fsize
    self._pv = np.finfo(np.float32).min # self._pv = np.finfo(feed_dict[self.X].dtype).min
    X_val_mat = super(MaxPool2dOp, self)._X_tensor2patchmat(feed_dict).\
                reshape((out_height, out_width, batch, in_channels, filter_height, filter_width)).\
                reshape((-1, filter_height * filter_width))
    return X_val_mat


class BiasBroadcastOp(_BaseOp):
  """Op that broadcast the bias.

  For example, given `X` = array([0.1, 0.2, 0.4]) and `shape` = [2, 3, 3]
  The emitted tensor would be
  array([[[ 0.1,  0.2,  0.4],
          [ 0.1,  0.2,  0.4],
          [ 0.1,  0.2,  0.4]],

         [[ 0.1,  0.2,  0.4],
          [ 0.1,  0.2,  0.4],
          [ 0.1,  0.2,  0.4]]])

  Parameters
  ----------
  X: `_BaseOp`;
    Input 1D tensor containing bias
  shape: `np.ndarray`;
    1D array of length >= 1, the last dim must matches dim[0] of X
  sess: `Session`;
    The session in which the Op is defined 
  """
  def __init__(self, X, shape, sess):
    if shape[-1] != X.shape[0]:
      raise ValueError("Last dim of new shape (%d) and dim[0] of X (%d) do not match!" \
              % (shape[-1], X.shape[0]))
    super(BiasBroadcastOp, self).__init__(sess)
    self.shape = shape
    self.X = X
    self.X.parent_total += 1
    self.ones = np.ones((np.prod(self.shape[:-1]), 1))

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of tensor containing bias broadcast across dimensions
    """
    X_val = self.X.eval(feed_dict).reshape((1, -1))
    X_tnsr_val = np.dot(self.ones, X_val).reshape(self.shape)
    return X_tnsr_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)].reshape((-1, self.X.shape[0]))
    dX_val = np.dot(self.ones.T, grad_val).reshape(self.X.shape[0])
    self.X.grad(feed_dict, dX_val)


class ReshapeOp(_BaseOp):
  """Op that reshapes input tensor.

  Paramters
  ---------
  X: `_BaseOp`;
    Input tensor to be reshaped
  shape: `np.ndarray`;
    The shape that `X` will be reshaped to
  sess: `Session`;
    The session that the Op is associated with  
  """
  def __init__(self, X, shape, sess):
    super(ReshapeOp, self).__init__(sess)
    self.shape = shape
    self.X = X
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; the n-D array containing the values of the reshaped tensor
    """
    X_val = self.X.eval(feed_dict).reshape(self.shape)
    return X_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    dX_val = grad_val.reshape(self.X.shape)
    self.X.grad(feed_dict, dX_val)


class DropoutOp(_BaseOp):
  """Op that implements dropout.

  In input tensor `X`, a random subset of units are forced to output zero with probability 
  `1 - KEEP_PROB`, while the output of the remaining units is scaled up by `1 / self.KEEP_PROB`.
  
  Paramters
  ---------
  X: `_BaseOp`;
    Input tensor in which a random subset of units is forced to output zero
  KEEP_PROB: `_BaseOp`;
    The probability with which each unit in `X` is kept (i.e. not forced to output zero) 
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, X, KEEP_PROB, sess):
    super(DropoutOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.KEEP_PROB = KEEP_PROB
    self.X.parent_total += 1
    self.KEEP_PROB.parent_total += 1
    self._mask = None

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; the n-D array containing the values of `self.X` after dropout
    """
    X_val = self.X.eval(feed_dict)
    X_dropout_val = X_val / self.KEEP_PROB.eval(feed_dict) 
    self._set_mask(feed_dict)
    X_dropout_val[self._mask] = 0.
    return X_dropout_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    dX_val = grad_val / self.KEEP_PROB.eval(feed_dict)
    self._set_mask(feed_dict)
    dX_val[self._mask] = 0.
    self.X.grad(feed_dict, dX_val)

  def _get_mask(self, feed_dict):
    """Compute a boolean-valued tensor the same shape as `X` containing indicator variables
    (True if the component is to be dropped, False if the component is to be kept).

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    return np.random.rand(*self.shape) >= self.KEEP_PROB.eval(feed_dict)

  def _set_mask(self, feed_dict):
    """Set the masking tensor containing indicator variables if None.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    if self._mask is None:
      self._mask = self._get_mask(feed_dict)

  def _reset_mask(self):
    """Reset `self._mask` to None"""
    self._mask = None


class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
  """Op that computes the cross-entropy

  Logits are first transformed into probabilitis by applying softmax function on each row of 
  `logits`, then cross-entropy  is computed based on probabilities and true class labels. 
  Emits tensor of dimension [batch].
  

  Parameters
  ----------
  labels: `_BaseOp`;
    2D tensor of dimensions [batch, num_of_classes] 
  logits: `_BaseOp`;
    2D tensor of dimensions [batch, num_of_classes]
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, labels, logits, sess):
    super(SoftmaxCrossEntropyWithLogitsOp, self).__init__(sess)
    self.shape = (logits.shape[0],)
    self.labels = labels
    self.logits = logits
    self.labels.parent_total += 1
    self.logits.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 1D array of dimension [batch]
    """
    logits_probs_val = self._eval_softmax(feed_dict)
    labels_val = self.labels.eval(feed_dict)
    cross_entropy = np.sum(-np.log(logits_probs_val) * labels_val, axis=1)
    return cross_entropy

  def _eval_softmax(self, feed_dict):
    """Transform `logits` into probabilities by applying softmax function

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 2D array of dimensions [batch, num_of_classes]
    """
    logits_val = self.logits.eval(feed_dict)
    logits_probs_val = np.exp(logits_val)
    logits_probs_val = logits_probs_val / logits_probs_val.sum(axis=1, keepdims=True)
    return logits_probs_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `logits`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    logits_probs_val = self._eval_softmax(feed_dict)
    labels_val = self.labels.eval(feed_dict)
    dlogits_val = logits_probs_val - labels_val
    dlogits_val = grad_val.reshape((-1, 1)) * dlogits_val
    self.logits.grad(feed_dict, dlogits_val)


class ReduceMeanOp(_BaseOp):
  """Op that reduce a tensor to its mean (average) along an axis

  Parameters
  ----------
  X: `_BaseOp`;
    Input tensor
  axis: integer;
    The axis along which `X` is averaged
  sess: `Session`;
    The session that the Op is associated with 
  """
  def __init__(self, X, axis, sess):
    super(ReduceMeanOp, self).__init__(sess)
    self.shape = X.shape[:axis] + X.shape[axis + 1:]
    self.X = X
    self.X.parent_total = 1
    self.axis = axis

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; N-D array with one less dimension than `X`
    """    
    X_reduce_mean_val = self.X.eval(feed_dict).mean(axis=self.axis)
    return X_reduce_mean_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """    
    axis, shape = self.axis, self.shape
    grad_val = self.sess.gradients[id(self)] / self.X.shape[axis]
    grad_val = grad_val.reshape((int(np.prod(shape[:axis])), int(np.prod(shape[axis:]))))
    grad_val = np.tile(grad_val, (1, self.X.shape[axis]))
    dX_val = grad_val.reshape(self.X.shape)
    self.X.grad(feed_dict, dX_val)


class Session(object):
  """ Session that keeps track of the following info of all the Operations (Op) in a data flow 
  graph across iterations of backpropagation:
    `variables`: Op's
    `values`: Value of each node (i.e. tensor emitted by `Op`) in the graph  
    `gradients`: Gradients w.r.t. each node (i.e. tensor emitted by `Op`) in the graph
  """
  def __init__(self):
    self.variables = []
    self.values = {}
    self.gradients = {}

  def _start(self, obj_tensor, feed_dict):
    """Set the objective tensor and kicks off the gradient computation. 

    This function sets `parent_total` of `obj_tensor` to 1 and the gradient w.r.t to 
    `obj_tensor` is set to 1., and this gradient is backpropped THROUGHOUT THE ENTIRE DATA FLOW 
    GRAPH by invoking the `grad()` and `_grad_func()` method of each Op recursively. In the end the
    gradient w.r.t each Op (except for tensors containing constant input data) is computed and 
    stored in the dict `sess.gradients`.
    
    Parameters
    ----------
    obj_tensor: `_BaseOp`;
      The objective function to be optimized (e.g. loss), with shape (1, 1) or ()
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; N-D array with one less dimension than `X`
    """
    obj_tensor.parent_total = 1
    obj_tensor.grad(feed_dict, 1.)

  def sgd_update(self, params, obj_tensor, feed_dict):
    """Update the tunable parameters using SGD algorithm.

    Parameters
    ----------
    params: `params`;
      dict: containing hyperparameters
    obj_tensor: `_BaseOp`;
      The objective function to be optimized (e.g. loss), with shape (1, 1) or ()
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    self._start(obj_tensor, feed_dict)

    alpha = params["alpha"]

    tensor_value_list = [(tensor, value) for tensor, value in zip(feed_dict.keys(), feed_dict.values())
                          if tensor.is_variable]
    updated_value_list = [value - alpha * self.gradients[id(tensor)]
                            for tensor, value in tensor_value_list]

    tensor_value_list = zip(*tensor_value_list)
    tensor_value_list[1] = updated_value_list
    tensor_value_list = zip(*tensor_value_list)

    feed_dict.update(dict(tensor_value_list))
    self._reset()

  def adam_update(self, params, obj_tensor, feed_dict):
    """Update the tunable parameters using ADAM algorithm.

    http://arxiv.org/abs/1412.6980

    Parameters
    ----------
    params: `params`;
      dict: containing hyperparameters
    obj_tensor: `_BaseOp`;
      The objective function to be optimized (e.g. loss), with shape (1, 1) or ()
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    self._start(obj_tensor, feed_dict)

    alpha = params["alpha"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    epsilon = params["epsilon"]
    t = params["t"]
    m = params["m"]
    v = params["v"]   

    alpha_t = alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t)) if t >= 1 else alpha

    for tensor in feed_dict.keys():
      if not tensor.is_variable:
        continue
      g = self.gradients[id(tensor)]
      m[tensor] = beta1 * m[tensor] + (1 - beta1) * g
      v[tensor] = beta2 * v[tensor] + (1 - beta2) * g * g
      feed_dict[tensor] += -alpha_t * m[tensor] / (np.sqrt(v[tensor]) + epsilon) 

    params["m"] = m
    params["v"] = v
    params["t"] += 1
    self._reset()   

  def eval_tensor(self, tensor, feed_dict):
    """Evaluate a tensor.

    Parameters
    ----------
    tensor: `_BaseOp`;
      The tensor whose value is to be computed
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    tensor_val = tensor.eval(feed_dict)
    self._reset()
    return tensor_val

  def _reset(self):
    """Reset data associated with Op's in each iteration"""
    self.values = {}
    self.gradients = {}
    for op in self.variables:
      op.parent_acc = 0
      if op.__class__ == DropoutOp:
        op._reset_mask()
