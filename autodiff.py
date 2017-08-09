import numpy as np

class _BaseOp(object):
  """Base class of all Op classes. An Op emits a tensor of a specific shape.

    Initializes the following attributes of an Op:

    sess: `Session`;
    shape: `numpy.ndarray`;
      1D array, e.g. [2, 3, 3, 2], specifying shape of the emitted tensor
    parent_total: integer;
      Total number of Op's that take the current Op as argument, which is determined when
      computational graph is defined initially 
    parent_acc: integer;
      Initialized to zero, it keeps track of number of parent Op's that have backpropped
      gradients to the current Op in an iteration

    Parameters
    ----------
    sess: `Session`;
      The session that the Op is associated with
  """  
  def __init__(self, sess):
    self.sess = sess
    self.shape = None
    self.parent_total = 0
    self.parent_acc = 0
    self.sess.vars.append(self)
    

  def grad(self, feed_dict, backprop):
    """Update the gradient with respect to the current Op, and propagate gradient to child Op's.
  
    `_grad_func` is implemented by each Op to propagate gradient downstream to child Op's.

    Graident backpropped from parent Op's are first accumulated, and then are propagated to child
    Op's when `parent_acc` == `parent_total`.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {`Op`: `numpy.ndarray`}

    backprop: `numpy.ndarray`;
      Gradient backpropped from a parent Op. Has the same shape as `self.shape`
    """
    if id(self) not in self.sess.grads:
      self.sess.grads[id(self)] = np.zeros(self.shape)
    self.sess.grads[id(self)] += backprop
    self.parent_acc += 1

    if self.parent_acc == self.parent_total and hasattr(self, "_grad_func"):
      self._grad_func(feed_dict)

  def eval(self, feed_dict):
    """Evaluate the current Op.

    `_eval_func` is implemented by each Op to compute the tensor value.

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    self.sess.vals[id(self)]: `numpy.ndarray`;
      Value of the tensor
    """
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = self._eval_func(feed_dict)
    return self.sess.vals[id(self)]


class PlaceholderOp(_BaseOp):
  """Op that holds place for input and parameters, which are the leaf nodes of the computational
  graph

  %Emits: X

  Parameters
  ----------
  shape: `numpy.array`;
    1D array specifying shape of tensor
  sess: `Session`;
    The session that the Op is associated with
  """ 
  def __init__(self, shape, sess):
    super(PlaceholderOp, self).__init__(sess)
    self.shape = shape

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    feed_dict[self]: `numpy.ndarray`;
      Value of the tensor
    """
    return feed_dict[self]
    

class SigmoidOp(_BaseOp):
  """Op that applies sigmoid function to the input `X` componentwise

  %Emits: sigmoid(X)

  Parameters
  ----------
  X: `numpy.ndarray`;
    The input tensor
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, X, sess):
    super(SigmoidOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    1. / (1. + np.exp(-self.X.eval(feed_dict))): `numpy.ndarray`
      Value of tensor
    """
    return 1. / (1. + np.exp(-self.X.eval(feed_dict)))

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to child `X`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    sig = self.eval(feed_dict)
    sig = sig * (1. - sig)
    grad_val = self.sess.grads[id(self)]
    self.X.grad(feed_dict, grad_val * sig)


class ReluOp(_BaseOp):
  """Op that applies ReLU to the input `X` componentwise

  %Emits: ReLU(X)

  Parameters
  ----------
  X: `numpy.ndarray`;
    The input tensor 
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, X, sess):
    super(ReluOp, self).__init__(sess)
    self.shape = X.shape
    self.X = X
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    np.maximum(0, self.X.eval(feed_dict)): `numpy.ndarray`
      Value of tensor
    """
    return np.maximum(0, self.X.eval(feed_dict))

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    val = self.X.eval(feed_dict)
    grad_val = self.sess.grads[id(self)]
    grad_val[val <= 0] = 0
    self.X.grad(feed_dict, grad_val)


class ParamRegOp(_BaseOp):
  """Op that computes the regularization term from input params `X`

  %Emits: 0.5 * `reg` * `X.ravel().square().sum()`

  Parameters
  ----------
  X: `numpy.ndarray`;
    The input parameters
  reg: float;
    Float between 0.0 and 1, controls strength of regularization
  sess: `Session`;
    The session that the Op is associated with
  """  
  def __init__(self, X, reg, sess):
    super(ParamRegOp, self).__init__(sess)
    self.shape = 1, 1
    self.X = X
    self.reg = reg
    self.X.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    .5 * self.reg * np.sum(val * val): `numpy.ndarray`
      Value of tensor (scalar)
    """
    val = self.X.eval(feed_dict)
    return .5 * self.reg * np.sum(val * val)

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    val = self.X.eval(feed_dict)
    grad_val = self.sess.grads[id(self)]
    self.X.grad(feed_dict, grad_val * self.reg * val)


class AddOp(_BaseOp):
  """Op that computes the summation of two input tensors

  %Emits: X1 + X2

  Parameters
  ----------
  X1: `numpy.ndarray`;
    Input tensor
  X2: `numpy.ndarray`;
    Input tensor
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, X1, X2, sess):
    super(AddOp, self).__init__(sess)
    self.shape = X1.shape
    self.X1 = X1
    self.X2 = X2
    self.X1.parent_total += 1
    self.X2.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    self.X1.eval(feed_dict) + self.X2.eval(feed_dict): `numpy.ndarray`
      Value of tensor
    """  
    return self.X1.eval(feed_dict) + self.X2.eval(feed_dict)

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X1` and `X2`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.grads[id(self)]
    self.X1.grad(feed_dict, grad_val)
    self.X2.grad(feed_dict, grad_val)


class MatMulOp(_BaseOp):
  """Op that computes the matrix multiplication

  %Emits: np.dot(X1, X2)

  Parameters
  ----------
  X1: `numpy.ndarray`;
    Input 2D tensor, matrix
  X2: `numpy.ndarray`;
    Input 2D tensor, matrix
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, X1, X2, sess):
    super(MatMulOp, self).__init__(sess)
    self.shape = X1.shape[0], X2.shape[1]
    self.X1 = X1
    self.X2 = X2
    self.X1.parent_total += 1
    self.X2.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    np.dot(self.X1.eval(feed_dict), self.X2.eval(feed_dict)): `numpy.ndarray`
      Value of tensor
    """  
    return np.dot(self.X1.eval(feed_dict), self.X2.eval(feed_dict))

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X1` and `X2`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.grads[id(self)]
    X1_val = self.X1.eval(feed_dict)
    X2_val = self.X2.eval(feed_dict)
    self.X1.grad(feed_dict, np.dot(grad_val, X2_val.T))
    self.X2.grad(feed_dict, np.dot(X1_val.T, grad_val))


class _2dKernelOp(_BaseOp):
  """Base class of 2D filter Op's, i.e. `Conv2dOp` and `MaxPool2dOp`.

  Provides methods for padding zeros (`_pad_X`) and removing padded zeros (`_depad_dX`) for
  a 4D tensor, method for computing the height and width coordinates of the upper left pixel
  of all image patches (`_img_col_index`), and method for converting the gradient w.r.t. `X`
  in the format of "patch matrix" to a 4D tensor with the same shape as `X` (`_dX_patchmat2tensor`)

  Parameters
  ----------
  X: `_BaseOp`;
    Op that emits a 4D tensor, with dimensions in [batch, in_height, in_width, in_channels]
  fsize: `numpy.ndarray`;
    1D array of length 2, specifying filter sizes along height and width axes, e.g. [3, 3]
  strides: `numpy.ndarray`;
    1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
  padding: `numpy.ndarray`;
    1D array of length 2, specifying total zero pading along height and width axes, e.g. [2, 2]
  sess: `Session`;
    The session that the Op is associated with
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
    """Pad a 4D tensor with zeros (or specified value).

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    X_pad: `numpy.ndarray`;
      The input 4D array with zero padded
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
    """Remove padded zeros in a 4D tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    dX: `numpy.ndarray`;
      4D array, the gradient with respect to the input array `X` (`dX`) where padded zeros are removed
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
    img_col_index: `numpy.ndarray`
      2D array containing 4-tuples of `h`, `w`, `h_idx`, `w_idx`, where `h` and `w` have spacing
      greater than 1 depending on strides, and `h_idx` and `w_idx` have spaceing of 1. 
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
    """Convert image columns from input tensor into matrix form

    The matrix is 2D array with dimensions in 
      [batch * out_height * out_width, filter_height * filter_width * in_channels]

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    X_val_mat: `numpy.ndarray`
      2D array, matrix
    """
    X_val = self._pad_X(feed_dict)
    filter_height, filter_width = self.fsize
    batch = self.X.shape[0]

    X_val_mat = np.vstack([X_val[:, h:h+filter_height, w:w+filter_width, :]
              .transpose(0, 3, 1, 2).reshape((batch, -1))
                for h, w, _, _ in self._img_col_index_val])
    return X_val_mat

  def _dX_patchmat2tensor(self, dX_val_mat):
    """Convert the gradient stored in "patch matrix" into 4D tensor with the same shape as `X`

    The conversion is conducted in the following steps:

    1. `dX_val_mat` with dimensions [out_height * out_width * batch, 
      in_channels * filter_height * filter_width] is reshaped into 6D array with dimensions
      [out_height, out_width, batch, in_channels, filter_height, filter_width]

    2. The dimensions in the 6D array is reordered as [out_height, out_width, batch, filter_height,
      filter_width, in_channels], i.e. reshaped into `dX_val_tmp`

    3. Initialize a zero tensor of shape `self._X_shape_pad`, i.e. `dX_val`

    4. The gradient in each patch of `dX_val_tmp` is added to the corresponding subarray (i.e. patch)
      in `dX_val`

    5. Removed padded zeros in `dX_val`

    Parameters
    ----------
    dX_val_mat: `numpy.ndarray`
      2D array, the patch matrix, with dimensions [out_height * out_width * batch, 
      in_channels * filter_height * filter_width]

    Returns
    -------
    dX_val: `numpy.ndarray`
      4D tensor, with dimensions [batch, in_height, in_width, in_channels]
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
    """Op that performs 2D convolution on a 4D tensor

    Computes output tensor of dimensions [batch, out_height, out_width, out_channels]

    Parameters
    ----------
      X: `_BaseOp`;
        Input 4D tensor, with dimensions in [batch, in_height, in_width, in_channels]
      W: `_2dKernelOp`;
        Kernel 4D tensor, with dimensions in [filter_height, filter_width, in_channels, out_channels]
      strides: `numpy.ndarray`;
        1D array of length 2, specifying strides along height and width axes, e.g. [2, 2]
      padding: `numpy.ndarray`;
        1D array of length 2, specifying total zero pading along height and width axes, e.g. [2, 2]
      sess: `Session`;
        The session that the Op is associated with
    """
    fsize = W.shape[0], W.shape[1]
    super(Conv2dOp, self).__init__(X, fsize, strides, padding, sess)
    self.shape = X.shape[0], self._out_height, self._out_width, W.shape[3]
    self.W = W
    self.W.parent_total += 1

  def _W_tensor2patchmat(self, feed_dict):
    """Convert filter tensor into matrix form

    The matrix is 2D array with dimensions in
    [in_channels * filter_height * filter_width, out_channels]

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    W_val_mat: `numpy.ndarray`
      2D array, matrix
    """
    W_val = self.W.eval(feed_dict)
    out_channels = self.W.shape[3]
    W_val_mat = W_val.transpose(2, 0, 1, 3).reshape((-1, out_channels))
    return W_val_mat

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    C_val_mat.reshape((out_height, out_width, batch, out_channels)).transpose(2, 0, 1, 3): 
      `numpy.ndarray`
      4D tensor, the output of Conv2d(X, W)
    """
    batch, out_height, out_width, out_channels = self.shape
    X_val_mat = self._X_tensor2patchmat(feed_dict)
    W_val_mat = self._W_tensor2patchmat(feed_dict)
    C_val_mat = np.dot(X_val_mat, W_val_mat)
    return C_val_mat.reshape((out_height, out_width, batch, out_channels)).transpose(2, 0, 1, 3)

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X` and `W`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    filter_height, filter_width, in_channels, out_channels = self.W.shape

    grad_val = self.sess.grads[id(self)]
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
    """Op that performs 2D max pooling on a 4D tensor

    Computes output tensor of dimensions [batch, out_height, out_width, out_channels]

    Parameters
    ----------
      X: `_BaseOp`;
        Input 4D tensor, with dimensions in [batch, in_height, in_width, in_channels]
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
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    P_val: `numpy.ndarray`
      4D tensor, the output of MaxPool(X)
    """  
    batch, out_height, out_width, in_channels = self.shape
    X_val_mat = self._X_tensor2patchmat(feed_dict)
    P_val = X_val_mat.max(axis=1).reshape((out_height, out_width, batch, in_channels)).\
            transpose(2, 0, 1, 3)
    return P_val

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `X`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    filter_height, filter_width = self.fsize
    batch, out_height, out_width, in_channels = self.shape

    X_val_mat = self._X_tensor2patchmat(feed_dict)
    X_val_mat = np.apply_along_axis(self.__ind_vec, 1, X_val_mat)

    grad_val = self.sess.grads[id(self)]
    grad_val_mat = np.repeat(grad_val.transpose(1, 2, 0, 3).reshape((-1, 1)), X_val_mat.shape[1], axis=1)

    dX_val_mat = X_val_mat * grad_val_mat
    dX_val = self._dX_patchmat2tensor(dX_val_mat)

    self.X.grad(feed_dict, dX_val)

  def __ind_vec(self, row):
    """Helper that computes indicator vector that marks that maximum of input 1D vector

    For example, row = [-1, 3, 2, -3]
    output = [0, 1, 0, 0] 

    Parameters
    ----------
    row: `numpy.ndarray`
      1D array
    """
    index = np.argmax(row)
    ind_vec = np.zeros_like(row)
    ind_vec[index] = 1.0
    return ind_vec

  def _X_tensor2patchmat(self, feed_dict):
    """Convert image columns from input tensor into matrix form

    The matrix is 2D array with dimensions in 
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
    self._pv = np.finfo(feed_dict[self.X].dtype).min
    X_val_mat = super(MaxPool2dOp, self)._X_tensor2patchmat(feed_dict).\
                reshape((out_height, out_width, batch, in_channels, filter_height, filter_width)).\
                reshape((-1, filter_height * filter_width))
    return X_val_mat
  

class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
  """Op that computes the cross-entropy (averaged over all instances) 

  Logits are first transformed into probabilitis by applying softmax function followed by 
  normalization, then cross-entropy is computed based on probabilities and true class labels.

  Parameters
  ----------
  logits: `_BaseOp`;
    2D tensor of dimensions [batch, num_of_classes]
  labels: `numpy.ndarray`
    1D array containing ids of ground truth class labels 
  sess: `Session`;
    The session that the Op is associated with
  """
  def __init__(self, logits, labels, sess):   
    super(SoftmaxCrossEntropyWithLogitsOp, self).__init__(sess)
    self.shape = 1, 1
    self.logits = logits
    self.labels = labels
    self.logits.parent_total += 1

  def _eval_func(self, feed_dict):
    """Function that outputs the value of the tensor

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    cross_entropy: `numpy.ndarray`
      Value of tensor (scalar)
    """
    probs = self._eval_softmax(feed_dict)
    cross_entropy = -np.log(probs[range(probs.shape[0]), self.labels])
    cross_entropy = cross_entropy.mean()
    return cross_entropy
   
  def _eval_softmax(self, feed_dict):
    """Evaluate softmax function

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}

    Returns
    -------
    probs: `numpy.ndarray`
      2D array of dimensions [batch, num_of_classes]
    """
    logits = self.logits.eval(feed_dict)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

  def _grad_func(self, feed_dict):
    """Propagate gradient downstream to `logits`

    Parameters
    ----------
    feed_dict: `dict`;
      dict: {id(`Op`): `numpy.ndarray`}
    """
    grad_val = self.sess.grads[id(self)]
    probs = self._eval_softmax(feed_dict)
    dscores = probs
    dscores[range(dscores.shape[0]), self.labels] -= 1.
    dscores /= dscores.shape[0]
    self.logits.grad(feed_dict, grad_val * dscores)


class Session(object):
  """ Session that keeps track of the following info of all the Operations (Op) in a computational
  graph across iterations of backpropagation:
    `vars`: Op's
    `vals`: Value of tensor emitted from an Op, given a `feed_dict` to the `Placeholder`
    `grads`: Gradient with respect to tensor emitted from an Op, given a 'feed_dict' to `Placeholder`
  """
  def __init__(self):
    self.vals = {}
    self.grads = {}
    self.vars = []

  def reset(self):
    """Reset info associated with Op's in each iteration"""
    self.vals = {}
    self.grads = {}
    for tensor in self.vars:
      tensor.parent_acc = 0
