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
from test_utils import get_subsets
from test_utils import get_combinations


SHAPE_LIST = [(), (1,), (2,), (1, 3), (1, 1, 1), (5, 1, 2, 3), (7, 4, 2, 3, 2)]


def enum_subshapes(shape, unique=True):
  if len(shape) <= 1:
    return [tuple(shape)] 
  subshapes_list = []
  boundaries = get_subsets(list(range(1, len(shape))), 0)
  boundaries = [[0] + boundary + [len(shape) + 1] for boundary in boundaries]

  for boundary in boundaries:
    subshapes = tuple(int(np.prod(shape[l:h])) for l, h 
        in zip(boundary[:-1], boundary[1:]))
    subshapes_list.append(subshapes) 

  if unique:
    subshapes_list = list(set(subshapes_list))
  return subshapes_list


def reshape_case(old_shape, new_shape):
  av = np.random.randint(-5, 5, size=old_shape).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  rst = tf.reshape(at, new_shape)
  gv = np.random.randint(-5, 5, size=new_shape).astype(np.float32)

  rst_val = rst.eval()
  grad = tf.gradients(rst, [at], gv)
  dat_val = grad[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    a = ad.placeholder(old_shape)
    rs = ad.reshape(a, new_shape)

  feed_dict = {a: av}
  with rt.forward_backward_cycle():
    rs.backward(feed_dict, gv)
    rs_val = rt._fwval[rs.name]
    da_val = rt._bwval[a.name]

  return rst_val, rs_val, dat_val, da_val


def reduce_sum_case(shape, dims):
  av = np.random.randint(-5, 5, size=shape).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  rst = tf.reduce_sum(at, axis=dims)
  gv = np.random.randint(-5, 5, size=rst.shape).astype(np.float32)

  rst_val = rst.eval()
  grad = tf.gradients(rst, [at], gv)
  dat_val = grad[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    a = ad.placeholder(shape)
    rs = ad.reduce_sum(a, axis=dims)

  feed_dict = {a: av}
  with rt.forward_backward_cycle():
    rs.backward(feed_dict, gv)
    rs_val = rt._fwval[rs.name]
    da_val = rt._bwval[a.name]

  return rst_val, rs_val, dat_val, da_val


def reduce_mean_case(shape, dims):
  av = np.random.randint(-5, 5, size=shape).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  rst = tf.reduce_mean(at, axis=dims)
  gv = np.random.randint(-5, 5, size=rst.shape).astype(np.float32)

  rst_val = rst.eval()
  grad = tf.gradients(rst, [at], gv)
  dat_val = grad[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    a = ad.placeholder(shape)
    rs = ad.reduce_mean(a, axis=dims)

  feed_dict = {a: av}
  with rt.forward_backward_cycle():
    rs.backward(feed_dict, gv)
    rs_val = rt._fwval[rs.name]
    da_val = rt._bwval[a.name]

  return rst_val, rs_val, dat_val, da_val


def pad_case(shape, padding):
  av = np.random.randint(-5, 5, size=shape).astype(np.float32)
  # tf 
  at = tf.convert_to_tensor(av)
  pt = tf.pad(at, padding)
  gv = np.random.randint(-5, 5, size=pt.shape).astype(np.float32)

  pt_val = pt.eval()
  grad = tf.gradients(pt, [at], gv)
  dat_val = grad[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    a = ad.placeholder(shape)
    p = ad.pad(a, padding)

  feed_dict = {a: av}
  with rt.forward_backward_cycle():
    p.backward(feed_dict, gv)
    p_val = rt._fwval[p.name]
    da_val = rt._bwval[a.name]

  return pt_val, p_val, dat_val, da_val 


class TestArrayOps(unittest.TestCase):

  def _get_shapes_for_reshape(self):
    old_shapes = []
    new_shapes = []
    for old_shape in SHAPE_LIST:
      subshapes = enum_subshapes(old_shape)
      old_shapes.extend([old_shape] * len(subshapes))
      new_shapes.extend(subshapes)
    return old_shapes, new_shapes

  def _get_shapes_for_reduction_op(self):
    shape_list = []
    dims_list = []
    for shape in SHAPE_LIST:
      if len(shape) > 0:
        dims = get_subsets(list(range(len(shape))))
        shape_list.extend([shape] * len(dims))
        dims_list.extend(dims)

    # add cases where dims is an int scalar
    shape_list.extend(list(filter(lambda s: len(s) > 0, SHAPE_LIST)))
    dims_list.extend([0] * len(list(filter(lambda s: len(s) > 0, SHAPE_LIST))))
    return shape_list, dims_list

  def _get_shapes_for_pad(self, shape, max_pad_size=3):
    dims = len(shape)
    padding = np.array(get_combinations([range(0, max_pad_size)] * dims * 2))
    padding = padding.reshape((-1, dims, 2))
    return padding

  def _print_parameters_reshape(self, old_shape, new_shape):
    print('old_shape', old_shape)
    print('new_shape', new_shape, '\n')

  def _print_parameters_reduction_op(self, shape, dims):
    print('shape', shape)
    print('dims', dims, '\n')
 
  def _print_parameters_pad(self, shape, padding):
    print('shape', shape)
    print('padding', padding, '\n')

  def test_reshape(self):
    print('\nTesting Reshape...')    
    sess = tf.InteractiveSession()
    old_shapes, new_shapes = self._get_shapes_for_reshape()
    for old_shape, new_shape in zip(old_shapes, new_shapes):
      self._print_parameters_reshape(old_shape, new_shape)
      rst_val, rs_val, dat_val, da_val = reshape_case(old_shape, new_shape)
      self.assertTrue((rst_val == rs_val).all())
      self.assertTrue((dat_val == da_val).all())
    sess.close()

  def test_reduce_sum(self):
    print('\nTesting ReduceSum...')
    sess = tf.InteractiveSession()
    shape_list, dims_list = self._get_shapes_for_reduction_op()
    for shape, dims in zip(shape_list, dims_list):
      self._print_parameters_reduction_op(shape, dims)
      rst_val, rs_val, dat_val, da_val = reduce_sum_case(shape, dims)
      self.assertTrue((rst_val == rs_val).all())
      self.assertTrue((dat_val == da_val).all())
    sess.close() 

  def test_reduce_mean(self):
    print('\nTesting ReduceMean...')
    sess = tf.InteractiveSession()
    shape_list, dims_list = self._get_shapes_for_reduction_op()
    for shape, dims in zip(shape_list, dims_list):
      self._print_parameters_reduction_op(shape, dims)
      rst_val, rs_val, dat_val, da_val = reduce_mean_case(shape, dims)
      self.assertTrue((rst_val == rs_val).all())
      self.assertTrue((dat_val == da_val).all())
    sess.close()

  def test_pad(self):
    print('\nTessing Pad...')
    sess = tf.InteractiveSession()
    shape = (2, 3)
    paddings = self._get_shapes_for_pad(shape)
    for padding in paddings:
      self._print_parameters_pad(shape, padding)
      pt_val, p_val, dat_val, da_val = pad_case(shape, padding)
      self.assertTrue((pt_val == p_val).all())
      self.assertTrue((dat_val == da_val).all())
    sess.close()

if __name__ == '__main__':
  unittest.main()
