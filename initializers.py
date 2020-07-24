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
import numpy as np


class Initializer(object):
  """Base class of all variable initializer.

  Must implement the `__call__` method that generate a numpy array holding 
  initialized values.
  """
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

  def _get_random_state(self):
    """Returns numpy's random state for generating random numbers."""
    seed = self._params['seed'] if 'seed' in self._params else None
    random_state = np.random.RandomState(seed=seed)
    return random_state 


class TruncatedNormalInitializer(Initializer):
  """Truncated Normal initializer.

  Values are drawn from the Normal distribution. Those that lie outside of 
  [-2*stddev, 2*stddev] are re-drawn, with maximum num of attempts.
  """
  def __call__(self, shape, max_attempts=100):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of integers, the shape of numpy array.
      max_attempts: int scalar, max num of attempts to re-drawn values.

    Returns:
      numpy array of shape `shape`.
    """
    random_state = self._get_random_state()

    mean, stddev = self._params['mean'], self._params['stddev']
    values = random_state.normal(loc=mean, scale=stddev, size=np.prod(shape))
    for i in range(max_attempts):
      invalid_indices = np.abs(values) > 2. * stddev
      if invalid_indices.sum() == 0:
        break
      new_drawn = random_state.normal(
          loc=mean, scale=stddev, size=invalid_indices.sum())
      values[invalid_indices] = new_drawn
    return values.reshape(shape).astype(np.float32)


class RandomUniformInitializer(Initializer):
  """Random Uniform initializer.

  Draw values from the interval [minval, maxval] uniformly.
  """
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    random_state = self._get_random_state()
    return random_state.uniform(
        low=self._params['minval'], high=self._params['maxval'], size=shape)


class ZerosInitializer(Initializer):
  """Initialize an all-zero numpy array."""
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    return np.zeros(shape, dtype=np.float32)


class OnesInitializer(Initializer):
  """Initialize an all-one numpy array."""
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    return np.ones(shape, dtype=np.float32)


class VarianceBasedInitializer(Initializer):
  """Base class for weight initialization strategies that are designed to
  stabilize the variance of activations and gradients:

    * Glorot Uniform
    * Glorot Normal
    * He Uniform
    * He Normal
  """
  def _get_limit(self, shape, numerator, fanin_only):
    """Returns the limit of uniform or normal initializer.

    Args:
      shape: list of 2 or 4 ints, shape of the kernel.
      numerator: int scalar, the numerator specific to uniform or normal
        initializer.
      fanin_only: bool scalar, whether to

    Returns:
      limit: float scalar, the limit of uniform or normal initializer.
    """
    if len(shape) == 2:
      fan_in, fan_out = shape
    elif len(shape) == 4:
      kernel_height, kernel_width, in_channels, out_channels = shape
      fan_in = in_channels * kernel_height * kernel_width
      fan_out = out_channels * kernel_height * kernel_width
    else:
      raise ValueError('kernel must be 2D or 4D array, got %dD.' % len(shape))

    if fanin_only:
      limit = np.sqrt(numerator / fan_in)
    else:
      limit = np.sqrt(numerator / (fan_in + fan_out))
    return limit


class GlorotUniformInitializer(VarianceBasedInitializer):
  """Glorot Uniform Initializer proposed in
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  """
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of 2 or 4 integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    limit = self._get_limit(shape, 6, False)
    random_state = self._get_random_state()
    return random_state.uniform(low=-limit, high=limit, size=shape)

 
class GlorotNormalInitializer(VarianceBasedInitializer):
  """Glorot Normal Initializer proposed in
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  """
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of 2 or 4 integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    limit = self._get_limit(shape, 2, False)
    random_state = self._get_random_state()
    return random_state.normal(loc=0, scale=limit, size=shape)


class HeUniformInitializer(VarianceBasedInitializer):
  """He Uniform Initializer proposed in https://arxiv.org/abs/1502.01852."""
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of 2 or 4 integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    limit = self._get_limit(shape, 6, True)
    random_state = self._get_random_state()
    return random_state.uniform(low=-limit, high=limit, size=shape)


class HeNormalInitializer(VarianceBasedInitializer):
  """He Uniform Initializer proposed in https://arxiv.org/abs/1502.01852."""
  def __call__(self, shape):
    """Generate the numpy array holding initialized values.

    Args:
      shape: a tuple or list of 2 or 4 integers, the shape of numpy array.

    Returns:
      numpy array of shape `shape`.
    """
    limit = self._get_limit(shape, 2, True)
    random_state = self._get_random_state()
    return random_state.normal(loc=0, scale=limit, size=shape)
