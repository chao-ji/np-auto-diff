import numpy as np
from autodiff import *

# Generating synthetic data... 
np.random.seed(0)
features = 2 # dimensionality
classes = 3 # number of classes
batch = 100 # number of points per class
x = np.zeros((batch * classes, features))
y = np.zeros((batch * classes, classes))
for j in xrange(classes):
  ix = range(batch * j, batch * (j + 1))
  r = np.linspace(0.0, 1, batch) # radius
  t = np.linspace(j * 4, (j + 1) * 4, batch) + np.random.randn(batch) * 0.2 # theta
  x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
  y[ix, j] = 1.0

# Define a three-layer fully connected network...
hidden = 50
hidden2 = 50
reg = 1e-3

w1 = 0.1 * np.random.randn(features, hidden) 
w2 = 0.1 * np.random.randn(hidden, hidden2)
w3 = 0.1 * np.random.randn(hidden2, classes)
b1 = np.zeros((1, hidden))
b2 = np.zeros((1, hidden2))
b3 = np.zeros((1, classes))
    
sess = Session()

X = PlaceholderOp([batch * classes, features], sess, False)
Y = PlaceholderOp([batch * classes, classes], sess, False)
W1 = PlaceholderOp([features, hidden], sess)
B1 = PlaceholderOp([hidden], sess)
H_RAW = AddOp(MatMulOp(X, W1, sess), BiasBroadcastOp(B1, [batch * classes, hidden], sess), sess)
H = SigmoidOp(H_RAW, sess)

W2 = PlaceholderOp([hidden, hidden2], sess)
B2 = PlaceholderOp([hidden2], sess)
H2_RAW = AddOp(MatMulOp(H, W2, sess), BiasBroadcastOp(B2, [batch * classes, hidden2], sess), sess)
H2 = SigmoidOp(H2_RAW, sess)

W3 = PlaceholderOp([hidden2, classes], sess)
B3 = PlaceholderOp([classes], sess)
S = AddOp(MatMulOp(H2, W3, sess), BiasBroadcastOp(B3, [batch * classes, classes], sess), sess)

H = ReduceMeanOp(SoftmaxCrossEntropyWithLogitsOp(Y, S, sess), 0, sess)

R_W1 = ParamRegOp(W1, reg, sess)
R_W2 = ParamRegOp(W2, reg, sess)
R_W3 = ParamRegOp(W3, reg, sess)

LL = AddOp(AddOp(H, R_W1, sess), AddOp(R_W2, R_W3, sess), sess)

# Train...
feed_dict = {W1: w1, B1: b1, W2: w2, B2: b2, W3: w3, B3: b3}
params = {"alpha": 1e-1}

for ii in xrange(50000):

  feed_dict[X] = x
  feed_dict[Y] = y

  if ii % 1000 == 0:
    print "iteration %d: loss: %f" % (ii, LL.eval(feed_dict))

  sess.sgd_update(feed_dict, params, LL)

s_val = sess.eval_tensor(S, feed_dict)
print "accuracy: %f" % np.mean(np.argmax(s_val, axis=1) == np.argmax(y, axis=1))
