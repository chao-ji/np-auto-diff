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
import unittest

import tensorflow as tf
import numpy as np

import autodiff as ad
from test_utils import get_combinations

SHAPES = [(), 
          (1,), (5,), 
          (1, 1), (1, 2), (4, 1), (2, 3), 
          (1, 1, 1), (1, 1, 5), (1, 5, 1), (1, 2, 3),
          (5, 1, 2), (2, 4, 1), (2, 1, 1), (2, 2, 3), 
          (2, 2, 4, 1), (3, 1, 2, 3), (1, 1, 4, 3)]


def add_case(shape_a, shape_b):
  av = np.random.randint(-5, 5, size=shape_a).astype(np.float32)
  bv = np.random.randint(-5, 5, size=shape_b).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  bt = tf.convert_to_tensor(bv)

  try:
    ct = at + bt
  except ValueError:
    return 

  gv = np.random.randint(-5, 5, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)

  ct_val = ct.eval()
  
  grad = tf.gradients(ct, [at, bt], gt)
  dat_val = grad[0].eval()
  dbt_val = grad[1].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)
  with graph.as_default_graph():
    a = ad.placeholder(shape_a)
    b = ad.placeholder(shape_b)
    c = a + b 

  feed_dict = {a: av, b: bv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)
    c_val = rt._fwval[c.name]
    da_val = rt._bwval[a.name]
    db_val = rt._bwval[b.name]

  return ct_val, c_val, dat_val, da_val, dbt_val, db_val


def multiply_case(shape_a, shape_b):
  av = np.random.randint(-5, 5, size=shape_a).astype(np.float32)
  bv = np.random.randint(-5, 5, size=shape_b).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  bt = tf.convert_to_tensor(bv)

  try:
    ct = at * bt
  except ValueError:
    return

  gv = np.random.randint(-5, 5, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)

  ct_val = ct.eval()

  grad = tf.gradients(ct, [at, bt], gt)
  dat_val = grad[0].eval()
  dbt_val = grad[1].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)
  with graph.as_default_graph():
    a = ad.placeholder(shape_a)
    b = ad.placeholder(shape_b)
    c = a * b 

  feed_dict = {a: av, b: bv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)
    c_val = rt._fwval[c.name]
    da_val = rt._bwval[a.name]
    db_val = rt._bwval[b.name]

  return ct_val, c_val, dat_val, da_val, dbt_val, db_val


def subtract_case(shape_a, shape_b):
  av = np.random.randint(-5, 5, size=shape_a).astype(np.float32)
  bv = np.random.randint(-5, 5, size=shape_b).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  bt = tf.convert_to_tensor(bv)

  try:
    ct = at - bt
  except ValueError:
    return

  gv = np.random.randint(-5, 5, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)

  ct_val = ct.eval()

  grad = tf.gradients(ct, [at, bt], gt)
  dat_val = grad[0].eval()
  dbt_val = grad[1].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)
  with graph.as_default_graph():
    a = ad.placeholder(shape_a)
    b = ad.placeholder(shape_b)
    c = a - b

  feed_dict = {a: av, b: bv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)
    c_val = rt._fwval[c.name]
    da_val = rt._bwval[a.name]
    db_val = rt._bwval[b.name]

  return ct_val, c_val, dat_val, da_val, dbt_val, db_val


def matmul_case(shape_p, shape_q, shape_r):
  av = np.random.randint(-5, 5, size=(shape_p, shape_q)).astype(np.float32)
  bv = np.random.randint(-5, 5, size=(shape_q, shape_r)).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  bt = tf.convert_to_tensor(bv)

  ct = tf.matmul(at, bt)

  gv = np.random.randint(-5, 5, size=(shape_p, shape_r)).astype(np.float32)
  gt = tf.convert_to_tensor(gv)

  ct_val = ct.eval()

  grad = tf.gradients(ct, [at, bt], gt)
  dat_val = grad[0].eval()
  dbt_val = grad[1].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)
  with graph.as_default_graph():
    a = ad.placeholder((shape_p, shape_q))
    b = ad.placeholder((shape_q, shape_r))
    c = ad.matmul(a, b)

  feed_dict = {a: av, b: bv}
  with rt.forward_backward_cycle(): 
    c.backward(feed_dict, gv)
    c_val = rt._fwval[c.name]
    da_val = rt._bwval[a.name]
    db_val = rt._bwval[b.name]

  return ct_val, c_val, dat_val, da_val, dbt_val, db_val

  
class TestArithmeticOps(unittest.TestCase):

  def _get_binary_op_shapes(self):
    binary_op_shapes = [(i, j) for i in SHAPES for j in SHAPES]
    return binary_op_shapes

  def _get_matmul_shapes(self):
    matmul_shapes = get_combinations([range(1, 4), range(1, 4), range(1, 4)])
    return matmul_shapes

  def _print_parameters_binary_op(self, combo, i):
    print('parameter combination', i)
    print('shape_a', combo[0])
    print('shape_b', combo[1], '\n')
    
  def _print_parameters_matmul(self, combo, i):
    print('parameter combination', i)
    print('p', combo[0])
    print('q', combo[1])
    print('r', combo[2], '\n')

  def test_add(self):
    print('\nTesting Add...')
    sess = tf.InteractiveSession()
    binary_op_shapes = self._get_binary_op_shapes()
    c = 0
    for i, combo in enumerate(binary_op_shapes):
      self._print_parameters_binary_op(combo, i)
      results = add_case(*combo)
      if results is  None:
        continue
      ct_val, c_val, dat_val, da_val, dbt_val, db_val = results
      self.assertTrue((ct_val == c_val).all())
      self.assertTrue((dat_val == da_val).all())
      self.assertTrue((dbt_val == db_val).all())
      c += 1
    print('total num of cases:', c)
    sess.close()

  def test_multiply(self):
    print('\nTesting Multiply...')
    sess = tf.InteractiveSession()
    binary_op_shapes = self._get_binary_op_shapes()
    c = 0
    for i, combo in enumerate(binary_op_shapes):
      self._print_parameters_binary_op(combo, i)
      results = multiply_case(*combo)
      if results is  None:
        continue
      ct_val, c_val, dat_val, da_val, dbt_val, db_val = results
      self.assertTrue((ct_val == c_val).all())
      self.assertTrue((dat_val == da_val).all())
      self.assertTrue((dbt_val == db_val).all())
      c += 1
    print('total num of cases:', c)
    sess.close()

  def test_subtract(self):
    print('\nTesting Add...')
    sess = tf.InteractiveSession()
    binary_op_shapes = self._get_binary_op_shapes()
    c = 0
    for i, combo in enumerate(binary_op_shapes):
      self._print_parameters_binary_op(combo, i)
      results = subtract_case(*combo)
      if results is  None:
        continue
      ct_val, c_val, dat_val, da_val, dbt_val, db_val = results
      self.assertTrue((ct_val == c_val).all())
      self.assertTrue((dat_val == da_val).all())
      self.assertTrue((dbt_val == db_val).all())
      c += 1
    print('total num of cases:', c)
    sess.close()

  def test_matmul(self):
    print('\nTesting MatMul...')
    sess = tf.InteractiveSession()
    matmul_shapes = self._get_matmul_shapes()
    for i, combo in enumerate(matmul_shapes):
      self._print_parameters_matmul(combo, i)
      ct_val, c_val, dat_val, da_val, dbt_val, db_val = matmul_case(*combo)
      self.assertTrue((ct_val == c_val).all())
      self.assertTrue((dat_val == da_val).all())
      self.assertTrue((dbt_val == db_val).all())
    sess.close()    

if __name__ == '__main__':
  unittest.main()
