from test import *
import numpy as np

"""
Inspired by
https://cs224d.stanford.edu/notebooks/vanishing_grad_example.html
"""

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

h=50
h2=50
num_train_examples = X.shape[0]

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

step_size=1e-1

x = PlaceholderOp([300, 2])
w1 = PlaceholderOp([2, 50])

i1 = PlaceholderOp([300, 1])
bb1 = PlaceholderOp([1, 50])

a1l = MulOp(x, w1)
a1r = MulOp(i1, bb1)

h_raw = AddOp(a1l, a1r)

h = SigmoidOp(h_raw)

#h = PlaceholderOp([300, 50])
w2 = PlaceholderOp([50, 50])

i2 = PlaceholderOp([300, 1])
bb2 = PlaceholderOp([1, 50])

a2l = MulOp(h, w2)
a2r = MulOp(i2, bb2)

h2_raw = AddOp(a2l, a2r) 

#h2_raw = PlaceholderOp([300, 50])
h2 = SigmoidOp(h2_raw) 




w3 = PlaceholderOp([50, 3])
i3 = PlaceholderOp([300, 1])
bb3 = PlaceholderOp([1, 3])

a3l = MulOp(h2, w3)
a3r = MulOp(i3, bb3)
s = AddOp(a3l, a3r)

H = SoftmaxCrossEntropyWithLogitsOp(s, y)

reg = 1e-3

r_w1 = RegMatOp(w1, reg)
r_w2 = RegMatOp(w2, reg)
r_w3 = RegMatOp(w3, reg)

ll = AddOp(AddOp(H, r_w1), AddOp(r_w2, r_w3))


for ii in range(1):
    feed_dict = {   w3: W3,
                i3: np.ones((300, 1)),
                bb3: b3,
                w2: W2,
                i2: np.ones((300, 1)),
                bb2: b2,
                w1: W1,
                i1: np.ones((300, 1)),
                bb1: b1,
                x: X }

    print ll.eval(feed_dict)
    ll.deriv(feed_dict, np.ones((1, 1)))


    W1 += -step_size * w1.adjoint
    b1 += -step_size * bb1.adjoint
    W2 += -step_size * w2.adjoint
    b2 += -step_size * bb2.adjoint
    W3 += -step_size * w3.adjoint
    b3 += -step_size * bb3.adjoint
    
#    w3.reset(); w2.reset(); w1.reset(); bb3.reset(); bb2.reset(); bb1.reset()
#    i3.reset(); i2.reset(); i1.reset(); x.reset(); 

