import unittest

import numpy as np
import tensorflow as tf

import autodiff as ad
from test_utils import get_combinations

BATCH = 1, 2
INPUT_SIZE = (1, 1), (3, 3), (16, 16), (16, 25) 
KERNEL_SIZE = (1, 1), (2, 2), (3, 3), (3, 6)
IN_CHANNELS = 1, 2
OUT_CHANNELS = 1, 3 
STRIDES = (1, 1), (2, 2)
PADDING = 'SAME', 'VALID'


def conv2d_case(batch, 
                input_size, 
                kernel_size, 
                in_channels, 
                out_channels, 
                strides, 
                padding):
  xv = np.random.randint(-3, 3, size=(batch,) + input_size + (in_channels,)
      ).astype(np.float32)
  kv = np.random.randint(-3, 3, 
      size=kernel_size + (in_channels,) + (out_channels,)).astype(np.float32)
  # tf
  xt = tf.convert_to_tensor(xv)
  kt = tf.convert_to_tensor(kv)
  ct = tf.nn.conv2d(xt, kt, (1,) + strides + (1,), padding)

  gv = np.random.randint(-3, 3, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)
  grads = tf.gradients(ct, [xt, kt], gt)

  ct_val = ct.eval()
  dxt_val = grads[0].eval()
  dkt_val = grads[1].eval()
  # autotiff 
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)
  
  with graph.as_default_graph():
    x = ad.placeholder((batch,) + input_size + (in_channels,))
    k = ad.placeholder(kernel_size + (in_channels,) + (out_channels,))
    c = ad.conv2d(x, k, strides, padding) 

  feed_dict = {x: xv, k: kv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)

    c_val = rt._fwval[c.name]
    dx_val = rt._bwval[x.name]
    dk_val = rt._bwval[k.name]
  
  return c_val, ct_val, dx_val, dxt_val, dk_val, dkt_val


def maxpool2d_case(batch, 
                   input_size, 
                   kernel_size, 
                   in_channels, 
                   out_channels, 
                   strides, 
                   padding):
  xv = np.random.randint(-3, 3, size=(batch,) + input_size + (in_channels,)
      ).astype(np.float32)
  # tf
  xt = tf.convert_to_tensor(xv)
  ct = tf.nn.max_pool(
      xt, (1,) + kernel_size + (1,), (1,) + strides + (1,), padding)

  gv = np.random.randint(-3, 3, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)
  grads = tf.gradients(ct, [xt], gt)

  ct_val = ct.eval()
  dxt_val = grads[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    x = ad.placeholder((batch,) + input_size + (in_channels,))
    c = ad.maxpool2d(x, kernel_size, strides, padding)

  feed_dict = {x: xv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)

    c_val = rt._fwval[c.name]
    dx_val = rt._bwval[x.name]

  return c_val, ct_val, dx_val, dxt_val


def avgpool2d_case(batch, 
                   input_size, 
                   kernel_size, 
                   in_channels, 
                   out_channels, 
                   strides, 
                   padding):
  xv = np.random.randint(-3, 3, size=(batch,) + input_size + (in_channels,)
      ).astype(np.float32)
  # tf
  xt = tf.convert_to_tensor(xv)
  ct = tf.nn.avg_pool(
      xt, (1,) + kernel_size + (1,), (1,) + strides + (1,), padding)

  gv = np.random.randint(-3, 3, size=ct.shape).astype(np.float32)
  gt = tf.convert_to_tensor(gv)
  grads = tf.gradients(ct, [xt], gt)

  ct_val = ct.eval()
  dxt_val = grads[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    x = ad.placeholder((batch,) + input_size + (in_channels,))
    c = ad.avgpool2d(x, kernel_size, strides, padding)

  feed_dict = {x: xv}
  with rt.forward_backward_cycle():
    c.backward(feed_dict, gv)

    c_val = rt._fwval[c.name]
    dx_val = rt._bwval[x.name]

  return c_val, ct_val, dx_val, dxt_val


def relu_case(shape):
  av = np.random.randint(-5, 5, size=shape).astype(np.float32)
  # tf
  at = tf.convert_to_tensor(av)
  rlt = tf.nn.relu(at)
  gv = np.random.randint(-5, 5, size=shape).astype(np.float32)
  grads = tf.gradients(rlt, [at], gv)
  rlt_val = rlt.eval()
  dat_val = grads[0].eval()
  # autodiff
  graph = ad.Graph()
  rt = ad.RunTime()
  graph.set_runtime(rt)

  with graph.as_default_graph():
    a = ad.placeholder(shape)
    rl = ad.relu(a)

  feed_dict = {a: av}
  with rt.forward_backward_cycle():
    rl.backward(feed_dict, gv)
    rl_val = rt._fwval[rl.name]
    da_val = rt._bwval[a.name]  

  return rl_val, rlt_val, da_val, dat_val


class TestKernel2DOps(unittest.TestCase):
  def _get_combos(self):
    lists = [BATCH, INPUT_SIZE, KERNEL_SIZE, IN_CHANNELS, 
             OUT_CHANNELS, STRIDES, PADDING]
    combos = get_combinations(lists)
    print('num total parameter combinations', len(combos))
    combos = list(filter(lambda combo: combo[1][0] >= combo[2][0] and 
        combo[1][1] >= combo[2][1], combos))
    print('num valid parameter combinations', len(combos))
    return combos

  def _print_parameters(self, combo, i):
    print('parameter combination', i)
    print('batch', combo[0])
    print('input_size', combo[1])
    print('kernel_size', combo[2])
    print('in_channels', combo[3])
    print('out_channels', combo[4])
    print('strides', combo[5])
    print('padding', combo[6], '\n')

  def test_conv2d(self):
    print('\nTesting Conv2d...')
    sess = tf.InteractiveSession()
    combos = self._get_combos()
    for i, combo in enumerate(combos):
      self._print_parameters(combo, i)
      c_val, ct_val, dx_val, dxt_val, dk_val, dkt_val = conv2d_case(*combo)
      self.assertTrue((c_val == ct_val).all())
      self.assertTrue((dx_val == dxt_val).all())
      self.assertTrue((dk_val == dkt_val).all())
    sess.close()

  def test_maxpool2d(self):
    print('\nTesting MaxPool2D...')
    sess = tf.InteractiveSession()
    combos = self._get_combos()
    for i, combo in enumerate(combos):
      self._print_parameters(combo, i)
      c_val, ct_val, dx_val, dxt_val = maxpool2d_case(*combo)
      self.assertTrue((c_val == ct_val).all())
      self.assertTrue((dx_val == dxt_val).all())
    sess.close()

  def test_avgpool2d(self):
    print('\nTesting AvgPool2D...')
    sess = tf.InteractiveSession()
    combos = self._get_combos()
    for i, combo in enumerate(combos):
      self._print_parameters(combo, i)
      c_val, ct_val, dx_val, dxt_val = avgpool2d_case(*combo)
      self.assertTrue((c_val == ct_val).all())
      self.assertTrue((dx_val == dxt_val).all())
    sess.close()


class TestReLU(unittest.TestCase):
  def _get_shapes(self):
    return [(), (1,), (2,), (1, 1), (2, 1), (2, 4, 9, 5)]

  def _print_parameters(self, shape):
    print('shape', shape, '\n')

  def test_relu(self):
    print('\nTesting ReLU...')
    sess = tf.InteractiveSession()
    shapes = self._get_shapes()
    for shape in shapes:
      self._print_parameters(shape)
      rl_val, rlt_val, da_val, dat_val = relu_case(shape)
      self.assertTrue((rl_val == rlt_val).all())  
      self.assertTrue((da_val == dat_val).all())
    sess.close()


if __name__ == '__main__':
  unittest.main()
