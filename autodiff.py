import numpy as np

class _BaseOp(object):
  def __init__(self, sess):
    self.sess = sess
    self.shape = None
    self.parent_total = 0
    self.parent_count = 0
    self.sess.vars.append(self)

  def grad(self, feed_dict, backprop):
    if id(self) not in self.sess.grads:
      self.sess.grads[id(self)] = np.zeros(self.shape)
    self.sess.grads[id(self)] += backprop


class _2dKernelOp(_BaseOp):
  def __init__(self, X, strides, ksize, padding, sess):
    super(_2dKernelOp, self).__init__(sess)
    self.X = X
    self.X.parent_total += 1
    self.strides = strides
    self.padding = padding
    self.ksize = ksize
    self._X_shape_pad = self.X.shape[0], self.X.shape[1] + 2 * self.padding[0], self.X.shape[2] + 2 * self.padding[1], self.X.shape[3]
    self._out_height = (X.shape[1] - ksize[0] + 2 * padding[0]) / strides[0] + 1 
    self._out_width = (X.shape[2] - ksize[1] + 2 * padding[1]) / strides[1] + 1

  def _pad_X(self, feed_dict):
    padding_height, padding_width = self.padding
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad

    X_pad = np.zeros(self._X_shape_pad)
    X_pad[:,
        padding_height : in_height_pad - padding_height,
        padding_width : in_width_pad - padding_width, :] += self.X.eval(feed_dict)

    return X_pad

  def _depad_dX(self, dX):
    padding_height, padding_width = self.padding
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad

    dX = dX[:,
          padding_height : in_height_pad - padding_height,
          padding_width : in_width_pad - padding_width, :]
    return dX

  def _img_col_index(self):
    filter_height, filter_width, _, _ = self.W.shape
    stride_height, stride_width = self.strides
    _, in_height_pad, in_width_pad, _ = self._X_shape_pad

    img_col_index = [(i, j)
                      for i in np.arange(0, in_height_pad - filter_height + 1, stride_height)
                      for j in np.arange(0, in_width_pad - filter_width + 1, stride_width)]
    return img_col_index


class PlaceholderOp(_BaseOp):
  def __init__(self, shape, sess):
    super(PlaceholderOp, self).__init__(sess)
    self.shape = shape

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = feed_dict[self]
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(PlaceholderOp, self).grad(feed_dict, backprop)
    self.parent_count += 1


class SigmoidOp(_BaseOp):
  def __init__(self, arg, sess):
    super(SigmoidOp, self).__init__(sess)
    self.shape = arg.shape
    self.arg = arg
    self.arg.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = 1. / (1. + np.exp(-self.arg.eval(feed_dict)))
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(SigmoidOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      sig = self.eval(feed_dict)
      sig = sig * (1. - sig)
      grad_val = self.sess.grads[id(self)]
      self.arg.grad(feed_dict, grad_val * sig)


class ReLuOp(_BaseOp):
  def __init__(self, arg, sess):
    super(ReLuOp, self).__init__(sess)
    self.shape = arg.shape
    self.arg = arg
    self.arg.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = np.maximum(0, self.arg.eval(feed_dict))
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(ReLuOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      val = self.arg.eval(feed_dict)
      grad_val = self.sess.grads[id(self)]
      grad_val[val <= 0] = 0
      self.arg.grad(feed_dict, grad_val)


class RegMatOp(_BaseOp):
  def __init__(self, arg, reg, sess):
    super(RegMatOp, self).__init__(sess)
    self.shape = 1, 1
    self.arg = arg
    self.reg = reg
    self.arg.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      val = self.arg.eval(feed_dict)
      self.sess.vals[id(self)] = .5 * self.reg * np.sum(val * val)
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(RegMatOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      val = self.arg.eval(feed_dict)
      grad_val = self.sess.grads[id(self)]
      self.arg.grad(feed_dict, grad_val * self.reg * val)


class AddOp(_BaseOp):
  def __init__(self, arg1, arg2, sess):
    super(AddOp, self).__init__(sess)
    self.shape = arg1.shape
    self.arg1 = arg1
    self.arg2 = arg2
    self.arg1.parent_total += 1
    self.arg2.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = self.arg1.eval(feed_dict) + self.arg2.eval(feed_dict)
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(AddOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      grad_val = self.sess.grads[id(self)]
      self.arg1.grad(feed_dict, grad_val)
      self.arg2.grad(feed_dict, grad_val)


class MulOp(_BaseOp):
  def __init__(self, arg1, arg2, sess):
    super(MulOp, self).__init__(sess)
    self.shape = arg1.shape[0], arg2.shape[1]
    self.arg1 = arg1
    self.arg2 = arg2
    self.arg1.parent_total += 1
    self.arg2.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      self.sess.vals[id(self)] = np.dot(self.arg1.eval(feed_dict), self.arg2.eval(feed_dict))
    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(MulOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      grad_val = self.sess.grads[id(self)]
      arg1_val = self.arg1.eval(feed_dict)
      arg2_val = self.arg2.eval(feed_dict)
      self.arg1.grad(feed_dict, np.dot(grad_val, arg2_val.T))
      self.arg2.grad(feed_dict, np.dot(arg1_val.T, grad_val))


class Conv2dOp(_2dKernelOp):
  def __init__(self, X, W, strides, padding, sess):
    """
      X: [batch, in_height, in_width, in_channels]
      W: [filter_height, filter_width, in_cnannels, out_channels]
      C: [batch, out_height, out_width, out_channels]
    """ 
    ksize = W.shape[0], W.shape[1]
    super(Conv2dOp, self).__init__(X, strides, ksize, padding, sess)
    self.shape = X.shape[0], self._out_height, self._out_width, W.shape[3]
    self.W = W
    self.W.parent_total += 1

  def _X_tensor2mat(self, feed_dict):
    X_val = self._pad_X(feed_dict) 

    filter_height, filter_width, _, _ = self.W.shape
    batch = self.X.shape[0]

    X_val_mat = np.vstack([X_val[:, i:i+filter_height, j:j+filter_width, :]
              .transpose(0, 3, 1, 2).reshape((batch, -1))
                for i, j in self._img_col_index()])

    return X_val_mat

  def _W_tensor2mat(self, feed_dict):
    W_val = self.W.eval(feed_dict) 
    out_channels = self.W.shape[3]
    W_val_mat = W_val.transpose(2, 0, 1, 3).reshape((-1, out_channels))
    return W_val_mat

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      batch, out_height, out_width, out_channels = self.shape
      X_val_mat = self._X_tensor2mat(feed_dict)
      W_val_mat = self._W_tensor2mat(feed_dict)
      C = np.dot(X_val_mat, W_val_mat)
      self.sess.vals[id(self)] = C.reshape((out_height, out_width, batch, out_channels)).transpose(2, 0, 1, 3)

    return self.sess.vals[id(self)]

  def grad(self, feed_dict, backprop):
    super(Conv2dOp, self).grad(feed_dict, backprop)
    self.parent_count += 1

    if self.parent_count == self.parent_total:
      filter_height, filter_width, in_channels, out_channels = self.W.shape
      batch, out_height, out_width, _ = self.shape
      padding_height, padding_width = self.padding

      grad_val = self.sess.grads[id(self)]
      grad_val = grad_val.transpose(1, 2, 0, 3).reshape((-1, out_channels))
      W_val_mat = self._W_tensor2mat(feed_dict)
      X_val_mat = self._X_tensor2mat(feed_dict)

      d_W = np.dot(X_val_mat.T, grad_val)
      d_W = d_W.reshape((in_channels, filter_height, filter_width, out_channels)).transpose(1, 2, 0, 3)

      d_X_mat = np.dot(grad_val, W_val_mat.T)           
      d_X_mat = d_X_mat.reshape((out_height, out_width, batch, in_channels, filter_height, filter_width))
      d_X_mat = d_X_mat.transpose(0, 1, 2, 4, 5, 3)

      d_X = np.zeros(self._X_shape_pad)
      for i, j in self._img_col_index():
        d_X[:, i : i + filter_height, j : j + filter_width, :] += d_X_mat[i, j] 
      d_X = self._depad_dX(d_X)

      self.X.grad(feed_dict, d_X)
      self.W.grad(feed_dict, d_W)
      

class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
  def __init__(self, logits, labels, sess):   
    super(SoftmaxCrossEntropyWithLogitsOp, self).__init__(sess)
    self.shape = 1, 1
    self.logits = logits
    self.labels = labels
    self.logits.parent_total += 1

  def eval(self, feed_dict):
    if id(self) not in self.sess.vals:
      probs = self._eval_softmax(feed_dict)
      cross_entropy = -np.log(probs[range(probs.shape[0]), self.labels])
      cross_entropy = cross_entropy.mean()
      self.sess.vals[id(self)] = cross_entropy
    return self.sess.vals[id(self)]
   
  def _eval_softmax(self, feed_dict):
    logits = self.logits.eval(feed_dict)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

  def grad(self, feed_dict, backprop):
    super(SoftmaxCrossEntropyWithLogitsOp, self).grad(feed_dict, backprop)
    self.parent_count += 1
    if self.parent_count == self.parent_total:
      grad_val = self.sess.grads[id(self)]
      probs = self._eval_softmax(feed_dict)
      dscores = probs
      dscores[range(dscores.shape[0]), self.labels] -= 1.
      dscores /= dscores.shape[0]
      self.logits.grad(feed_dict, grad_val * dscores)


class Session(object):
  def __init__(self):
    self.vals = {}
    self.grads = {}
    self.vars = []

  def reset(self):
    self.vals = {}
    self.grads = {}
    for tensor in self.vars:
      tensor.parent_count = 0
