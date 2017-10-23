import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from autodiff import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
reg = 1e-3
iterations = 1000

w = np.random.normal(scale=0.01, size=(784, 10))
b = np.random.normal(scale=0.01, size=(1, 10))

sess = Session()

X = PlaceholderOp([batch_size, 784], sess, False)
Y = PlaceholderOp([batch_size, 10], sess, False)
W = PlaceholderOp([784, 10], sess)
B = PlaceholderOp([10], sess)
S = BiasAddOp(MatMulOp(X, W, sess), B, sess)
H = ReduceMeanOp(SoftmaxCrossEntropyWithLogitsOp(Y, S, sess), 0, sess)
F = AddOp(H, L2LossOp(W, reg, sess), sess)

feed_dict = {W: w, B: b}
params = {"alpha": 0.5}
for ii in xrange(iterations):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)

  feed_dict[X] = batch_xs
  feed_dict[Y] = batch_ys

  if ii % 100 == 0:
    F_val = sess.eval_tensor(F, feed_dict)
    S_val = sess.eval_tensor(S, feed_dict)
    print "iteration: %d, loss: %f, train accuracy: %f" % (ii, F_val, np.mean(np.argmax(S_val, axis=1) == np.argmax(batch_ys, axis=1)))

  sess.sgd_update(params, F, feed_dict)

print "Test stage:"

feed_dict[X] = mnist.test.images

S_val = sess.eval_tensor(S, feed_dict)
print "test set size: %d, test accuracy: %f" % (mnist.test.images.shape[0], np.mean(np.argmax(S_val, axis=1) == np.argmax(mnist.test.labels, axis=1)))
