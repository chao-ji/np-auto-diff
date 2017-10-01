import numpy as np

class _BaseOp(object):
  """Base class of all Op classes. An Op emits a tensor of a specific shape, and contains the
    following attributes:

    sess: `Session`;
      The session in which the Op is defined.
    shape: `numpy.ndarray`;
      1D array, e.g. [2, 3, 3, 2], specifying the shape of the emitted tensor.
    parent_total: integer;
      Total number of Op's for which the current Op is an argument; this is determined when
      the data flow graph was defined in the beginning.
    parent_acc: integer;
      Initialized to zero; it keeps track of the number of parent Op's that have backpropped
      gradients to the current Op in an iteration.
    is_terminal: bool;
      Initialized to False; indicates if the Op is terminal node (i.e. has no child node).
    _cache_data: dict;
      Caches data that are needed in both forward and backward pass to avoid recomputing.

    Parameters
    ----------
    sess: `Session`;
      The session in which the Op is defined.
  """  
  def __init__(self, sess):
    self.sess = sess
    self.shape = ()
    self.parent_total = 0
    self.parent_acc = 0
    self.is_terminal = False
    self._cache_data = {}
    self.sess.variables.append(self)

  def eval(self, feed_dict):
    """Forward pass. Evaluate the current Op.

    `_eval_func` is implemented by each Op separately to compute the tensor value.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}.

    Returns
    -------
    `numpy.ndarray`; value of the tensor.
    """
    if id(self) not in self.sess.values:
      self.sess.values[id(self)] = self._eval_func(feed_dict)
    return self.sess.values[id(self)]

  def grad(self, feed_dict, backprop):
    """Backward pass. Update the gradient w.r.t. the current Op (`backprop`), and propagate 
    gradient down to child Op's. `_grad_func` is implemented by each Op separately to propagate
    gradient down to child Op's. 

    NOTE: `grad` is invoked when a parent Op propagates gradient (`backprop`) back to the current
    Op. When `grad` is invoked, the gradient is accumulated and `parent_acc` is incremented, which
    maintains the number of parent Op's that have already backpropped gradients. The computation of
    the gradient w.r.t. the current Op is finished when `parent_acc` == `parent_total`, and this 
    gradient is further propagated down to child Op's of the current Op by invoking 
    `self._grad_func(feed_dict)`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}.

    backprop: `numpy.ndarray`;
      Gradient backpropped from a parent Op. Has the SAME shape as the shape of the current Op
      (i.e. `self.shape`).
    """
    if self.is_terminal and not self.is_variable:
      return

    if id(self) not in self.sess.gradients:
      self.sess.gradients[id(self)] = np.zeros(self.shape)
    self.sess.gradients[id(self)] += backprop
    self.parent_acc += 1

    if self.parent_acc == self.parent_total and not self.is_terminal:
        self._grad_func(feed_dict)

  def __repr__(self):
    """Display the representation with tensor shape."""
    return super(_BaseOp, self).__repr__()[:-1] + ", shape=" + str(self.shape) + ">"


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


class L2LossOp(_BaseOp):
  """Op that performs l2-regularization over the parameter tensor `W`. Specifically, it computes 
  `0.5 * reg * sum(|W|^2)`.
  
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
    super(L2LossOp, self).__init__(sess)
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
  """Base class of 2D filter Op's (`Conv2dOp` and `MaxPool2dOp`).

  It provides methods for
    1. Set the padding hyperparameters (`_set_padding()`) 
    2. Padding zeros (`_pad_X()`) for input 4D tensor;
    3. Removing padded zeros (`_depad_dX()`) for input 4D tensor;
    4. Computing the height and width coordinates of the upper left pixel of all image patches 
      (`_img_col_index()`);
    5. Converting input 4D tensor into the format of "patch matrix" (`_X_tensor2patchmat()`);
    6. Converting the gradient w.r.t. input tensor `X` in the format of "patch matrix" to a 4D 
      tensor with the same shape as `X` (`_dX_patchmat2tensor()`).

  Parameters
  ----------
  X: `_BaseOp`;
    Input 4D tensor, of dimensions in [batch, in_height, in_width, in_channels]
  fsize: `numpy.ndarray`;
    1D array of length 2, specifying filter sizes along height and width axes, e.g. [3, 3]
  strides: `numpy.ndarray`;
    1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
  padding: string;
    Either "SAME" or "VALID", specifying the padding algorithm 
  sess: `Session`;
    The session in which the Op is defined
  """
  def __init__(self, X, fsize, strides, padding, sess):
    super(_2dKernelOp, self).__init__(sess)
    self.X = X
    self.X.parent_total += 1
    self.fsize = fsize
    self.strides = strides
    self.padding = padding
    self._out_height, self._out_width, self._padding, self._X_shape_pad = self._set_padding()
    self._img_col_index_val = self._img_col_index()

  def _set_padding(self):
    """Set the padding hyperparameters according to the algorithm:

    https://www.tensorflow.org/versions/r1.3/api_guides/python/nn#Convolution

    Returns
    -------
    `tuple`; Tuple containing the 2D output dimensions (`_out_height` and `out_width`), padding
      hyperparameters (`_padding`), and the shape of the padded `X` (`_X_shape_pad`)
    """
    X, strides, fsize = self.X, self.strides, self.fsize
    if self.padding == "SAME":
      _out_height = int(np.ceil(float(X.shape[1]) / strides[0]))
      _out_width = int(np.ceil(float(X.shape[2]) / strides[1]))
      pad_along_height = max(fsize[0] - strides[0], 0) if X.shape[1] % strides[0] == 0 \
        else max(fsize[0] - X.shape[1] % strides[0], 0)
      pad_along_width = max(fsize[1] - strides[1], 0) if X.shape[2] % strides[1] == 0 \
        else max(fsize[1] - X.shape[2] % strides[1], 0)
      _padding = pad_along_height // 2, pad_along_height - pad_along_height // 2, \
                      pad_along_width // 2, pad_along_width - pad_along_width // 2
    elif self.padding == "VALID":
      _out_height = int(np.ceil(float(X.shape[1] - fsize[0] + 1) / strides[0]))
      _out_width = int(np.ceil(float(X.shape[2] - fsize[1] + 1) / strides[1]))
      _padding = 0, 0, 0, 0
    else:
      raise ValueError("Padding scheme should be 'SAME' or 'VALID'.")
    _X_shape_pad = X.shape[0], X.shape[1] + _padding[0] + _padding[1], \
                    X.shape[2] + _padding[2] + _padding[3], X.shape[3]
    return _out_height, _out_width, _padding, _X_shape_pad

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
    """PRE-PROCESSING step that converts input 4D tensor into 2D tensor in the "patch matrix" 
    format.

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
    if "X_val_mat" in self._cache_data:
      return self._cache_data["X_val_mat"] 

    X_val = self._pad_X(feed_dict)
    filter_height, filter_width = self.fsize
    batch = self.X.shape[0]

    self._cache_data["X_val_mat"] = np.vstack([X_val[:, h:h+filter_height, w:w+filter_width, :]
                                      .transpose(0, 3, 1, 2).reshape((batch, -1))
                                      for h, w, _, _ in self._img_col_index_val])
    return self._cache_data["X_val_mat"]

  def _dX_patchmat2tensor(self, dX_val_mat):
    """POST-PROCESSING step that convert the gradient w.r.t. the "patch matrix" into 4D tensor 
    with the same shape as `X`.

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
    """PRE-PROCESSING step that convert 4D filter tensor into "patch matrix" format.

    Filter 4D tensor `W` has dimensions [filter_height, filter_width, in_channels, out_channels] 
    = [2, 3, 3, 2]

    W[0,0,0,0]  W[0,1,0,0]    W[0,0,1,0]  W[0,1,1,0]
    W[1,0,0,0]  W[1,1,0,0]    W[1,0,1,0]  W[1,1,1,0]

    W[0,0,0,1]  W[0,1,0,1]    W[0,0,1,1]  W[0,1,1,1]
    W[1,0,0,1]  W[1,1,0,1]    W[1,0,1,1]  W[1,1,1,1]
    
    Each 2px-by-2px submatrix corresponds to a 2D array of dimensions [filter_height, filter_width],
    and the four smaller submatrixes form a 2-by-2 "matrix" where the rows corresponds to 
    `out_channels` and columns to `in_channels`.

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
    if "W_val_mat" in self._cache_data:
      return self._cache_data["W_val_mat"]

    W_val = self.W.eval(feed_dict)
    out_channels = self.W.shape[3]
    self._cache_data["W_val_mat"] = W_val.transpose(2, 0, 1, 3).reshape((-1, out_channels))
    return self._cache_data["W_val_mat"]

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Given `X` and `W` in "patch matrix" format `X_val_mat` and `W_val_mat`, right-multiplying 
    `W_val_mat` with `X_val_mat` produces 2D array of dimensions [out_height * out_width * batch, 
    out_channels] `C_val_mat`. `C_val_mat` is then reshaped into 4D tensor of dimensions [out_height
    , out_width, batch, out_channels], which is then reordered as [batch, out_height, out_width, 
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

    Given the 4D tensor `grad_val` (i.e. gradient w.r.t. Op's output) of dimensions [batch, 
    out_height, out_width, out_channels], it is reordered into [out_height, out_width, batch, 
    out_channels] and then reshaped into 2D array `grad_val_mat` of dimensions [out_height * 
    out_width * batch, out_channels].

    The gradient w.r.t `W` in the "patch matrix" format is computed by right-multiplying 
    `grad_val_mat` with `X_val_mat.T` (of dimensions [in_channels * filter_height * filter_width, 
    out_height * out_width * batch]), producing 2D array of dimensions [in_channels * filter_height
    * filter_width, out_channels].

    The gradient w.r.t. `X` in the "path matrix" format is computed by right-multiplying 
    `W_val_mat.T` (of dimensions [out_channels, in_channels * filter_height * filter_width]) with 
    `grad_val_mat`, procuding 2D array of dimensions [out_height * out_width * batch, in_channels *
     filter_height * filter_width].

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
    _argmax = self._argmax(X_val_mat)
    P_val = X_val_mat[np.arange(X_val_mat.shape[0]), _argmax].\
              reshape((out_height, out_width, batch, in_channels)).\
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
    _argmax = self._argmax(X_val_mat)
    ind_mat = np.zeros_like(X_val_mat)
    ind_mat[np.arange(ind_mat.shape[0]), _argmax] = 1

    grad_val = self.sess.gradients[id(self)]
    grad_val_mat = np.tile(grad_val.transpose(1, 2, 0, 3).reshape((-1, 1)), (1, ind_mat.shape[1]))

    dX_val_mat = ind_mat * grad_val_mat
    dX_val_mat = dX_val_mat.reshape((out_height * out_width * batch, in_channels * filter_height * 
                  filter_width))
    dX_val = self._dX_patchmat2tensor(dX_val_mat)
    self.X.grad(feed_dict, dX_val)

  def _X_tensor2patchmat(self, feed_dict):
    """PRE-PROCESSING step that converts input 4D tensor into 2D tensor in "patch matrix" format.

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
    self._pv = np.finfo(self.sess.dtype).min # self._pv = np.finfo(feed_dict[self.X].dtype).min
    X_val_mat = super(MaxPool2dOp, self)._X_tensor2patchmat(feed_dict).\
                reshape((out_height, out_width, batch, in_channels, filter_height, filter_width)).\
                reshape((-1, filter_height * filter_width))
    return X_val_mat

  def _argmax(self, X_val_mat):
    """Compute the indexes of the largest element of each flattened image patch per depth (channel) 
    of length `filter_height` * `filter_width`.

    Parameters
    ----------
    `X_val_mat`: `numpy.array`;
      2-D array of dimensions [out_height * out_width * batch * in_channels, 
                                filter_height * filter_width]
    """
    if "argmax" not in self._cache_data:
      self._cache_data["argmax"] = X_val_mat.argmax(axis=1)
    return self._cache_data["argmax"]


class BiasAddOp(_BaseOp):
  """Op that adds bias `B` to a tensor `X`.

  `X` may have arbitrary shape. `B` must be 1D tensor with length matching the last dim of `X`. For 
  example, given `B` = array([0.1, 0.2, 0.4]) and `X` = 
  array([[[0, 4, 0],
          [3, 9, 9],
          [6, 6, 6]],

         [[7, 0, 4],
          [7, 1, 7],
          [7, 1, 1]]])

  `X`+'B' would be 
  array([[[ 0.1,  4.2,  0.4],
          [ 3.1,  9.2,  9.4],
          [ 6.1,  6.2,  6.4]],

         [[ 7.1,  0.2,  4.4],
          [ 7.1,  1.2,  7.4],
          [ 7.1,  1.2,  1.4]]])
  Gradients are backpropped to both `X` and `B`.

  Parameters
  ----------
  X: `_BaseOp`;
    Input tensor with arbitrary shape
  B: `_BaseOp`;
    Input 1D tensor containing bias
  sess: `Session`;
    The session in which the Op is defined 
  """
  def __init__(self, X, B, sess):
    if len(B.shape) != 1 or X.shape[-1] != B.shape[0]:
      raise ValueError("`B` must be a 1D array with length matching the last dim of `X`.\
                        B.shape[0]: %d, X.shape[-1]: %d" % (B.shape[0], X.shape[-1]))
    super(BiasAddOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.B = B
    self.X.parent_total += 1
    self.B.parent_total += 1
    self._ones = np.ones((np.prod(self.shape[:-1]), 1))

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of broadcast `B` added to `X`
    """
    return self.X.eval(feed_dict) + self.B.eval(feed_dict)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X` and `B`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    dX_val = grad_val
    dB_val = np.dot(self._ones.T, grad_val.reshape((-1, self.B.shape[0]))).reshape(self.B.shape[0])
    self.X.grad(feed_dict, dX_val)
    self.B.grad(feed_dict, dB_val)


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
    return self.X.eval(feed_dict).reshape(self.shape)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

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

  http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

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
    _mask = self._mask(feed_dict)
    X_dropout_val[_mask] = 0.
    return X_dropout_val

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    dX_val = grad_val / self.KEEP_PROB.eval(feed_dict)
    _mask = self._mask(feed_dict)
    dX_val[_mask] = 0.
    self.X.grad(feed_dict, dX_val)

  def _mask(self, feed_dict):
    """Compute a boolean-valued tensor the same shape as `X` containing indicator variables
    (True if the component is to be dropped, False if the component is to be kept).

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; the n-D array with the same size as `X` containing indicator variables
    """
    if "mask" not in self._cache_data:
      self._cache_data["mask"] = np.random.binomial(1, 1. - self.KEEP_PROB.eval(feed_dict), 
                                  size=self.shape).astype(np.bool)
    return self._cache_data["mask"]


class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
  """Op that computes the cross-entropy.

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
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 1D array of dimension [batch]
    """
    logits_probs_val = self._softmax(feed_dict)
    labels_val = self.labels.eval(feed_dict)
    cross_entropy = np.sum(-np.log(logits_probs_val) * labels_val, axis=1)
    return cross_entropy

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `logits`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    logits_probs_val = self._softmax(feed_dict)
    labels_val = self.labels.eval(feed_dict)
    dlogits_val = logits_probs_val - labels_val
    dlogits_val = grad_val.reshape((-1, 1)) * dlogits_val
    self.logits.grad(feed_dict, dlogits_val)

  def _softmax(self, feed_dict):
    """Transform `logits` into probabilities by applying softmax function.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 2D array of dimensions [batch, num_of_classes]
    """
    if "softmax" in self._cache_data:
      return self._cache_data["softmax"]
    logits_val = self.logits.eval(feed_dict)
    logits_probs_val = np.exp(logits_val)
    self._cache_data["softmax"] = logits_probs_val / logits_probs_val.sum(axis=1, keepdims=True)
    return self._cache_data["softmax"]


class ReduceMeanOp(_BaseOp):
  """Op that reduce a tensor to its mean (average) along an axis.

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
    self.X.parent_total += 1
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
    return self.X.eval(feed_dict).mean(axis=self.axis)
    
  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

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


class ReduceSumOp(_BaseOp):
  """Op that reduce a tensor to its sum along an axis.

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
    super(ReduceSumOp, self).__init__(sess)
    self.shape = X.shape[:axis] + X.shape[axis + 1:]
    self.X = X
    self.X.parent_total += 1
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
    return self.X.eval(feed_dict).sum(axis=self.axis)

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    axis, shape = self.axis, self.shape
    grad_val = self.sess.gradients[id(self)]
    grad_val = grad_val.reshape((int(np.prod(shape[:axis])), int(np.prod(shape[axis:]))))
    grad_val = np.tile(grad_val, (1, self.X.shape[axis]))
    dX_val = grad_val.reshape(self.X.shape)
    self.X.grad(feed_dict, dX_val)


class LRNOp(_BaseOp):
  """Op that performs Local Response Normalization.

  Parameters
  ----------
  X: `_BaseOp`;
    Input 4D tensor
  sess: `Session`;
    The session that the Op is associated with 
  depth_radius: integer, defaults to 5;
    Half-width of the 1D normalization window 
  bias: float, defaults to 1.;
    Offset
  alpha: float, defaults to 1.;
    Scale factor
  beta: float, defaults to .5;
    Exponent
  """
  def __init__(self, X, depth_radius, bias, alpha, beta, sess):
    super(LRNOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.X.parent_total += 1
    self.depth_radius = depth_radius
    self.bias = bias
    self.alpha = alpha
    self.beta = beta

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of local response normalization performed on `X` 
    """
    X_val = self.X.eval(feed_dict)
    weighted_sqr_sum = self._weighted_sqr_sum(feed_dict)
    X_val = X_val / weighted_sqr_sum ** self.beta
    return X_val

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """  
    radius = self.depth_radius
    scale = -2 * self.alpha * self.beta 
    _matrix_shape = -1, self.shape[-1]

    X_val = self.X.eval(feed_dict).reshape(_matrix_shape)
    _WSS = self._weighted_sqr_sum(feed_dict).reshape(_matrix_shape)
    M = X_val * np.power(_WSS, -self.beta - 1)
    WSS_p = np.power(_WSS, -self.beta) 
    grad_val = self.sess.gradients[id(self)].reshape(_matrix_shape)

    def _func(i):
      X_val_row, M_row, WSS_p_row, grad_val_row = X_val[i], M[i], WSS_p[i], grad_val[i]

      def _func1(j):
        vec_k = np.zeros(grad_val.shape[1])
        def _func2(k):
          update = scale * M_row[j] * X_val_row[k]
          update = update + WSS_p_row[j] if k == j else update
          return update * grad_val_row[j]
        indexes = range(max(0, j - radius), min(j + radius + 1, grad_val.shape[1]))
        vec_k[indexes] += np.array(map(_func2, indexes))
        return vec_k
      return np.array(map(_func1, range(grad_val.shape[1]))).sum(axis=0)

    dX_val = np.array(map(_func, range(grad_val.shape[0]))).reshape(self.shape)
    self.X.grad(feed_dict, dX_val)

  def _weighted_sqr_sum(self, feed_dict):
    """Computes the weighted squared sum of local response normalization.

    The output raise to power of `beta` is used to normalized input tensor `X`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; 2-D array the same size as `X` containing the sum of weighted squares 
    """
    if "WSS" in self._cache_data:
      return self._cache_data["WSS"]
    X_val = self.X.eval(feed_dict)
    radius = self.depth_radius
    X_val = X_val.reshape((-1, self.shape[-1]))
    def _func(x):
      return map(lambda i: np.sum(np.square(x[max(0, i - radius):
              min(i + radius + 1, self.shape[-1])])), xrange(len(x)))
    sqr_sum = np.apply_along_axis(_func, 1, X_val)
    weighted_sqr_sum = self.bias + self.alpha * sqr_sum 
    self._cache_data["WSS"] = weighted_sqr_sum.reshape(self.shape)
    return self._cache_data["WSS"]


class MomentsOp(_BaseOp):
  """Op that emits the mean and variance of input tensor `X` over axes `axes`.

  The dimensions of `X` are kept in mean and variance. For example, `X` has shape
  [2, 3, 5, 4], and `axes_drop` = [0, 3], then the resulting mean and variance have shape
  [1, 3, 5, 1], and the output of `_eval_func` has shape [2, 1, 3, 5, 1].

  Parameters
  ----------
  X: `_BaseOp`;
    Input tensor
  axes_drop: `list`;
    The axes that are normalized over
  sess: `Session`;
    The session that the Op is associated with 
  """
  def __init__(self, X, axes_drop, sess):
    super(MomentsOp, self).__init__(sess)
    self._axes_keep = [i for i in xrange(len(X.shape)) if i not in axes_drop]
    self._shape_keep = [X.shape[i] for i in self._axes_keep]
    self._shape_drop = [X.shape[i] for i in axes_drop]
    self._average_over = int(np.prod(self._shape_drop))
    self._shape = np.array(X.shape)
    self._shape[axes_drop] = 1
    self.shape = tuple([2] + list(self._shape))
    self.X = X
    self.X.parent_total += 1
    self.axes_drop = axes_drop

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of the ndarray containing mean and variance of `X` 
    """
    X_val = self.X.eval(feed_dict)
    X_val = X_val.transpose(self.axes_drop + self._axes_keep).\
              reshape((-1, int(np.prod(self._shape_keep))))
    mean_val, var_val = X_val.mean(axis=0), X_val.var(axis=0)
    mean_val, var_val = mean_val.reshape(self._shape), var_val.reshape(self._shape)
    return np.array([mean_val, var_val])

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)]
    dX_m_val = grad_val[0] / self._average_over
    X_val = self.X.eval(feed_dict)
    mean_val = self.eval(feed_dict)[0]
    dX_v_val = grad_val[1] * (2 * (X_val - mean_val) / self._average_over)
    self.X.grad(feed_dict, dX_m_val + dX_v_val)


class BatchNormOp(_BaseOp):
  """Op that performs batch normalization.

  http://arxiv.org/abs/1502.03167 

  Parameters
  ----------
  X: `_BaseOp`;
    Input tensor to be batch-normalized
  mean_var: `_BaseOp`;
    Tensor containing mean and variance of `X`. Emitted by `MomentsOp`.
  offet: `_BaseOp`;
    Offset tensor (i.e. `beta` in the paper). Has the same shape as mean.
  scale: `_BaseOp`;
    Scale tensor (i.e. `gamma` in the paper). Has the same shape as var.
  epsilon: float;
    Small float to avoid dividing by zero
  sess: `Session`;
    The session that the Op is associated with 
  """
  def __init__(self, X, mean_var, offset, scale, epsilon, sess):
    super(BatchNormOp, self).__init__(sess)
    _indexes = np.arange(len(X.shape))
    self.axes_drop = list(_indexes[np.array(X.shape) != np.array(offset.shape)])
    self._axes_keep = list(_indexes[np.array(X.shape) == np.array(offset.shape)])
    self._shape_keep = [X.shape[i] for i in self._axes_keep]

    self.shape = X.shape 
    self.X = X
    self.X.parent_total += 1
    self.mean_var = mean_var
    self.mean_var.parent_total += 1
    self.offset = offset
    self.offset.parent_total += 1
    self.scale = scale
    self.scale.parent_total += 1
    self.epsilon = epsilon
     
  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    `numpy.ndarray`; value of local response normalization performed on `X` 
    """
    X_val, epsilon = self.X.eval(feed_dict), self.epsilon
    mean_val, var_val = self.mean_var.eval(feed_dict)
    offset_val, scale_val = self.offset.eval(feed_dict), self.scale.eval(feed_dict)
    standard_X_val = self._standard_X(X_val, mean_val, var_val, epsilon)
    val = scale_val * standard_X_val + offset_val
    return val

  def _grad_func(self, feed_dict):
    """Propagate gradient down to `X`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.gradients[id(self)] 
    X_val, epsilon = self.X.eval(feed_dict), self.epsilon
    mean_val, var_val = self.mean_var.eval(feed_dict)
    offset_val, scale_val = self.offset.eval(feed_dict), self.scale.eval(feed_dict)
    dStandard_X = grad_val * scale_val

    d_mean_val = dStandard_X * np.power(var_val + epsilon, -0.5)
    d_mean_val = -self._postprocess(d_mean_val, self.mean_var.shape[1:])
    d_var_val = dStandard_X * (X_val - mean_val) * np.power(var_val + epsilon, -1.5)
    d_var_val = -0.5 * self._postprocess(d_var_val, self.mean_var.shape[1:])
    self.mean_var.grad(feed_dict, np.array([d_mean_val, d_var_val]))

    dX_val = dStandard_X * np.power(var_val + epsilon, -0.5)
    self.X.grad(feed_dict, dX_val)

    d_offset_val = self._postprocess(grad_val, self.offset.shape)
    self.offset.grad(feed_dict, d_offset_val)

    d_scale_val = self._standard_X(X_val, mean_val, var_val, epsilon) * grad_val
    d_scale_val = self._postprocess(d_scale_val, self.scale.shape)
    self.scale.grad(feed_dict, d_scale_val)

  def _standard_X(self, X_val, mean_val, var_val, epsilon):
    """Computes the standardized `X` (i.e. Zero mean and unit variance).

    Parameters
    ----------
    X_val: `numpy.ndarray`;
      Input array
    mean_val: `numpy.ndarray`;
      Mean of `X_val`
    var_val: `numpy.ndarray`;
      Variance of `X_val`

    Returns
    -------
    `numpy.ndarray`; standardized `X`
    """
    if "standard_X" not in self._cache_data:
      self._cache_data["standard_X"] = (X_val - mean_val) / np.sqrt(var_val + epsilon)
    return self._cache_data["standard_X"]

  def _postprocess(self, array, shape):
    """Postprocess input `array` for computing gradients.
    
    Parameters
    ----------
    array: `numpy.ndarray`;
      2D input array to be processed
    shape: `list`;
      The desired shape

    Returns
    -------
    `numpy.ndarray`; the post-processed array
    """
    array = array.transpose(self.axes_drop + self._axes_keep).\
            reshape((-1, int(np.prod(self._shape_keep)))).sum(axis=0).\
            reshape(shape)
    return array
    

class Session(object):
  """ Session that keeps track of the following info of all the Operations (Op) in a data flow 
  graph across iterations of backpropagation:
    `variables`: Op's
    `values`: Value of each node (i.e. tensor emitted by `Op`) in the graph  
    `gradients`: Gradients w.r.t. each node (i.e. tensor emitted by `Op`) in the graph
    `dtype`: Data type of arrays
  """
  def __init__(self, dtype=np.float32):
    self.variables = []
    self.values = {}
    self.gradients = {}
    self.dtype = dtype

  def _start(self, obj_tensor, feed_dict):
    """Set the objective tensor and kicks off the gradient computation. 

    This function sets `parent_total` of `obj_tensor` to 1 and the gradient w.r.t each component of 
    `obj_tensor` is set to 1., and this gradient is backpropped THROUGHOUT THE ENTIRE DATA FLOW 
    GRAPH by invoking the `grad()` and `_grad_func()` method of each Op recursively. In the end the
    gradient w.r.t each Op (except for tensors containing constant input data) is computed and 
    stored in the dict `sess.gradients`.
    
    Parameters
    ----------
    obj_tensor: `_BaseOp`;
      The objective function to be optimized (e.g. loss)
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    obj_tensor.parent_total = 1
    obj_tensor.grad(feed_dict, np.ones(obj_tensor.shape))

  def sgd_update(self, params, obj_tensor, feed_dict):
    """Update the tunable parameters using Stochastic Gradient Descent (SGD) algorithm.

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
    tensor_value_list = [(tensor, value) for tensor, value in zip(feed_dict.keys(), 
                          feed_dict.values()) if tensor.is_variable]
    updated_value_list = [(tensor, value - alpha * self.gradients[id(tensor)])
                            for tensor, value in tensor_value_list if id(tensor) in self.gradients]
    for tensor, value in updated_value_list:
      feed_dict[tensor] = value
    self._reset()

  def adam_update(self, params, obj_tensor, feed_dict):
    """Update the tunable parameters using Adaptive Moment Estimation (Adam) algorithm.

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

    alpha, beta1, beta2 = params["alpha"], params["beta1"], params["beta2"]
    epsilon, t, m, v = params["epsilon"], params["t"], params["m"], params["v"]
    alpha_t = alpha if t < 1 else alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t))

    for tensor in feed_dict.keys():
      if not tensor.is_variable and id(tensor) not in self.gradients:
        continue
      g = self.gradients[id(tensor)]
      m[tensor] = beta1 * m[tensor] + (1 - beta1) * g
      v[tensor] = beta2 * v[tensor] + (1 - beta2) * g * g
      feed_dict[tensor] += -alpha_t * m[tensor] / (np.sqrt(v[tensor]) + epsilon) 

    params["m"], params["v"] = m, v
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

    Returns
    -------
    `numpy.ndarray`; n-D array containing the value of `tensor`
    """
    tensor_val = tensor.eval(feed_dict)
    self._reset()
    return tensor_val

  def _reset(self):
    """Reset data associated with Op's in each iteration."""
    self.values = {}
    self.gradients = {}
    for op in self.variables:
      op.parent_acc = 0
      op._cache_data = {}
