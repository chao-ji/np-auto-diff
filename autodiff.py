import numpy as np

def _sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


class PlaceholderOp():
    def __init__(self, shape):
        self.shape = np.array(shape)
        self.adjoint = np.zeros(self.shape)

    def eval(self, feed_dict):
        return feed_dict[self]

    def deriv(self, feed_dict, backprop, local_deriv=None):
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta.reshape(self.shape)
    def reset(self):
        self.adjoint = np.zeros(self.shape)

class SigmoidOp():
    def __init__(self, arg):
        self.arg = arg
        self.shape = arg.shape
        self.adjoint = np.zeros(self.shape)

    def eval(self, feed_dict):
        return _sigmoid(self.arg.eval(feed_dict))

    def local_deriv(self, feed_dict):
        val = self.eval(feed_dict)

        deriv_shape = list(self.shape)
        deriv_shape.extend(self.shape)

        deriv = np.zeros(deriv_shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                deriv[i, j, i, j] = val[i, j] * (1. - val[i, j]) 

        deriv = deriv.reshape((deriv_shape[0] * deriv_shape[1], deriv_shape[2] * deriv_shape[3]))

        return deriv

    def deriv(self, feed_dict, backprop, local_deriv=None):
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta.reshape(self.adjoint.shape)

#        print self.adjoint.reshape((1, -1)).shape, self.local_deriv(feed_dict).shape
        self.arg.deriv(feed_dict, self.adjoint.reshape((1, -1)), self.local_deriv(feed_dict))
    def reset(self):
        self.adjoint = np.zeros(self.shape)

class RegMatOp():
    def __init__(self, arg, reg):
        self.arg = arg
        self.shape = np.array([1, 1])
        self.adjoint = np.zeros(self.shape)
        self.reg = reg

    def eval(self, feed_dict):
        val = self.arg.eval(feed_dict)
        return .5 * self.reg * np.sum(val * val)

    def local_deriv(self, feed_dict):
        val = self.arg.eval(feed_dict)
        deriv = self.reg * val   
        deriv = deriv.reshape((1, -1))
        return deriv

    def deriv(self, feed_dict, backprop, local_deriv=None):
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta

#        return self.adjoint, self.local_deriv(feed_dict)
        self.arg.deriv(feed_dict, self.adjoint, self.local_deriv(feed_dict))
    def reset(self):
        self.adjoint = np.zeros(self.shape) 
        

class AddOp():
    """
        Matrix Addition

        M = X + Y
    """
    def __init__(self, arg_1, arg_2):
        self.arg_1 = arg_1
        self.arg_2 = arg_2
        self.shape = arg_1.shape
        self.adjoint = np.zeros(self.shape)

    def eval(self, feed_dict):
        return self.arg_1.eval(feed_dict) + self.arg_2.eval(feed_dict)

    def local_deriv(self, feed_dict):
        deriv_shape_l = list(self.shape)
        deriv_shape_l.extend(self.shape)
        deriv_l = np.zeros(deriv_shape_l)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                deriv_l[i, j, i, j] = 1.

        deriv_shape_r = list(self.shape)
        deriv_shape_r.extend(self.shape)
        deriv_r = np.zeros(deriv_shape_r)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                deriv_r[i, j, i, j] = 1.

        deriv_l = deriv_l.reshape((deriv_shape_l[0] * deriv_shape_l[1], deriv_shape_l[2] * deriv_shape_l[3]))
        deriv_r = deriv_r.reshape((deriv_shape_r[0] * deriv_shape_r[1], deriv_shape_r[2] * deriv_shape_r[3]))

        return deriv_l, deriv_r

    def deriv(self, feed_dict, backprop, local_deriv=None):
#        print backprop.shape, local_deriv.shape
#        print self.adjoint.shape
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta.reshape(self.adjoint.shape)

        dv = self.local_deriv(feed_dict)

#        print self.adjoint.shape, dv[0].shape
        self.arg_1.deriv(feed_dict, self.adjoint.reshape((1, -1)), dv[0])
        self.arg_2.deriv(feed_dict, self.adjoint.reshape((1, -1)), dv[1]) 

    def reset(self):
        self.adjoint = np.zeros(self.shape)


class MulOp():
    """
        Matrix Multiplication

        M = X * Y
    """
    def __init__(self, arg_l, arg_r):
        self.arg_l = arg_l
        self.arg_r = arg_r
        self.shape = np.array([arg_l.shape[0], arg_r.shape[1]])
        self.adjoint = np.zeros(self.shape)       
 
    def eval(self, feed_dict):
        return np.dot(self.arg_l.eval(feed_dict), self.arg_r.eval(feed_dict))

    def local_deriv(self, feed_dict):
        left = self.arg_l.eval(feed_dict)
        right = self.arg_r.eval(feed_dict)

        deriv_shape_l = list(self.shape)
        deriv_shape_l.extend(self.arg_l.shape)
        deriv_l = np.zeros(deriv_shape_l)

        for i in range(self.shape[0]):
            deriv_l[i, :, i, :] = right.T
        
        deriv_shape_r = list(self.shape)
        deriv_shape_r.extend(self.arg_r.shape)
        deriv_r = np.zeros(deriv_shape_r)

        for j in range(self.shape[1]):
            deriv_r[:, j, :, j] = left

        deriv_l = deriv_l.reshape((deriv_shape_l[0] * deriv_shape_l[1], deriv_shape_l[2] * deriv_shape_l[3]))
        deriv_r = deriv_r.reshape((deriv_shape_r[0] * deriv_shape_r[1], deriv_shape_r[2] * deriv_shape_r[3]))
    
        return deriv_l, deriv_r

    def deriv(self, feed_dict, backprop, local_deriv=None):
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta.reshape(self.shape)

        dv = self.local_deriv(feed_dict)
#        self.arg_l.deriv(feed_dict, self.adjoint, dv[0])

#        print self.adjoint.shape, dv[1].shape
        self.arg_l.deriv(feed_dict, self.adjoint.reshape((1, -1)), dv[0])
        self.arg_r.deriv(feed_dict, self.adjoint.reshape((1, -1)), dv[1])

    def reset(self):
        self.adjoint = np.zeros(self.shape)

class SoftmaxCrossEntropyWithLogitsOp():
    
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels
        self.shape = np.array([1, 1])
        self.adjoint = np.zeros(self.shape)

    def eval(self, feed_dict):
        probs = self._eval_softmax(feed_dict)

        cross_entropy = -np.log(probs[range(probs.shape[0]), self.labels])
        cross_entropy = cross_entropy.mean()
        return cross_entropy
   
    def _eval_softmax(self, feed_dict):
        logits = self.logits.eval(feed_dict)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def local_deriv(self, feed_dict):
        probs = self._eval_softmax(feed_dict)

        dscores = probs
        dscores[range(probs.shape[0]), self.labels] -= 1
        dscores /= probs.shape[0]

        dscores = dscores.reshape((1, -1))

        return dscores

    def deriv(self, feed_dict, backprop, local_deriv=None):
        # backprop = np.ones((1, 1))
        # local_deriv = None
        delta = np.dot(backprop, local_deriv) if local_deriv is not None else backprop
        self.adjoint += delta

#        return self.adjoint, self.local_deriv(feed_dict)
        self.logits.deriv(feed_dict, self.adjoint, self.local_deriv(feed_dict))

    def reset(self):
        self.adjoint = np.zeros(self.shape)
