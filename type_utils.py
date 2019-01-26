import numpy as np


def isint(i):
  """Returns if input is of integer type."""
  return isinstance(i, (int, np.int8, np.int16, np.int32, np.int64))

def is_variable(v):
  """Returns if input is of variable type."""
  return type(v).__name__ == 'Variable'

def to_numpy(val):
  """Converts input to a numpy array."""
  if not isinstance(val, np.ndarray):
    val = np.array(val).astype(np.float32)
  return val

def is_numeric(val):
  """Returns if input is of numeric type."""
  return isinstance(val, (int, float,
                          np.int8, np.int16, np.int32, np.int16, np.int64,
                          np.float16, np.float32, np.float64))
