import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
num_train_examples = X.shape[0]
y = np.zeros(N*K, dtype='uint8')
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

def sigmoid(x):
    x = 1/(1+np.exp(-x))
    return x

def sigmoid_grad(x):
    return (x)*(1-x)

def relu(x):
    return np.maximum(0,x)


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
h=50
h2=50
num_train_examples = X.shape[0]


reg = 1e-3

model={}
model['h'] = h # size of hidden layer 1
model['h2']= h2# size of hidden layer 2
model['W1']= 0.1 * np.random.randn(D,h)
model['b1'] = np.zeros((1,h))
model['W2'] = 0.1 * np.random.randn(h,h2)
model['b2']= np.zeros((1,h2))
model['W3'] = 0.1 * np.random.randn(h2,K)
model['b3'] = np.zeros((1,K))

h= model['h']
h2= model['h2']
W1= model['W1']
W2= model['W2']
W3= model['W3']
b1= model['b1']
b2= model['b2']
b3= model['b3']
    
    
# some hyperparameters


# gradient descent loop
num_examples = X.shape[0]
plot_array_1=[]
plot_array_2=[]


hidden_layer = relu(np.dot(X, W1) + b1)
hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
scores = np.dot(hidden_layer2, W3) + b3

exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

# compute the loss: average cross-entropy loss and regularization
corect_logprobs = -np.log(probs[range(num_examples),y])
data_loss = np.sum(corect_logprobs)/num_examples
reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)+ 0.5*reg*np.sum(W3*W3)
loss = data_loss + reg_loss


dscores = probs
dscores[range(num_examples),y] -= 1
dscores /= num_examples

 
# BACKPROP HERE
dW3 = (hidden_layer2.T).dot(dscores)
db3 = np.sum(dscores, axis=0, keepdims=True)

#backprop ReLU nonlinearity here
dhidden2 = np.dot(dscores, W3.T)
dhidden2[hidden_layer2 <= 0] = 0
dW2 =  np.dot( hidden_layer.T, dhidden2)
plot_array_2.append(np.sum(np.abs(dW2))/np.sum(np.abs(dW2.shape)))
db2 = np.sum(dhidden2, axis=0)
dhidden = np.dot(dhidden2, W2.T)
dhidden[hidden_layer <= 0] = 0

dW1 =  np.dot(X.T, dhidden)
plot_array_1.append(np.sum(np.abs(dW1))/np.sum(np.abs(dW1.shape)))
db1 = np.sum(dhidden, axis=0)


dW3 += reg * W3
dW2 += reg * W2
dW1 += reg * W1






print loss

#############################################################

from autodiff import *

sess = Session()

x = PlaceholderOp([300, 2], sess)
w1 = PlaceholderOp([2, 50], sess)
i1 = PlaceholderOp([300, 1], sess)
bb1 = PlaceholderOp([1, 50], sess)
a1l = MulOp(x, w1, sess)
a1r = MulOp(i1, bb1, sess)
h_raw = AddOp(a1l, a1r, sess)
h = ReLuOp(h_raw, sess)

w2 = PlaceholderOp([50, 50], sess)
i2 = PlaceholderOp([300, 1], sess)
bb2 = PlaceholderOp([1, 50], sess)
a2l = MulOp(h, w2, sess)
a2r = MulOp(i2, bb2, sess)
h2_raw = AddOp(a2l, a2r, sess)
h2 = ReLuOp(h2_raw, sess)

w3 = PlaceholderOp([50, 3], sess)
i3 = PlaceholderOp([300, 1], sess)
bb3 = PlaceholderOp([1, 3], sess)
a3l = MulOp(h2, w3, sess)
a3r = MulOp(i3, bb3, sess)
s = AddOp(a3l, a3r, sess)

H = SoftmaxCrossEntropyWithLogitsOp(s, y, sess)
r_w1 = RegMatOp(w1, reg, sess)
r_w2 = RegMatOp(w2, reg, sess)
r_w3 = RegMatOp(w3, reg, sess)

ll = AddOp(AddOp(H, r_w1, sess), AddOp(r_w2, r_w3, sess), sess)

step_size = 1e-1
for ii in xrange(50000):

    feed_dict = {x: X,
        w1: W1,
        i1: np.ones((300, 1)),
        bb1: b1,
        w2: W2,
        i2: np.ones((300, 1)),
        bb2: b2,
        w3: W3,
        i3: np.ones((300, 1)),
        bb3: b3}

    if ii % 1000 == 0:
        print "iteration %d: loss: %f" % (ii, ll.eval(feed_dict))
    ll.parent_total = 1
    ll.deriv(feed_dict, 1.)

    W1 += -step_size * sess.derivs[id(w1)]
    b1 += -step_size * sess.derivs[id(bb1)]
    W2 += -step_size * sess.derivs[id(w2)]
    b2 += -step_size * sess.derivs[id(bb2)]
    W3 += -step_size * sess.derivs[id(w3)]
    b3 += -step_size * sess.derivs[id(bb3)]

    sess.reset()


s_val = s.eval(feed_dict)
print "accuracy: %f" % np.mean(np.argmax(s_val, axis=1) == y)
