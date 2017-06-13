import numpy as np

class _BaseOp(object):
    def __init__(self, sess):
        self.sess = sess
        self.shape = None

    def deriv(self, feed_dict, backprop):
        if id(self) not in self.sess.derivs:
            self.sess.derivs[id(self)] = np.zeros(self.shape)
        self.sess.derivs[id(self)] += backprop


class PlaceholderOp(_BaseOp):
    def __init__(self, shape, sess):
        super(PlaceholderOp, self).__init__(sess)
        self.shape = shape

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            self.sess.vars[id(self)] = feed_dict[self]
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(PlaceholderOp, self).deriv(feed_dict, backprop)


class SigmoidOp(_BaseOp):
    def __init__(self, arg, sess):
        super(SigmoidOp, self).__init__(sess)
        self.shape = arg.shape
        self.arg = arg

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            self.sess.vars[id(self)] = 1. / (1. + np.exp(-self.arg.eval(feed_dict)))
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(SigmoidOp, self).deriv(feed_dict, backprop)
        sig = self.eval(feed_dict)
        sig = sig * (1. - sig)
        self.arg.deriv(feed_dict, backprop * sig)


class ReLuOp(_BaseOp):
    def __init__(self, arg, sess):
        super(ReLuOp, self).__init__(sess)
        self.shape = arg.shape
        self.arg = arg

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            self.sess.vars[id(self)] = np.maximum(0, self.arg.eval(feed_dict))
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(ReLuOp, self).deriv(feed_dict, backprop)
        val = self.arg.eval(feed_dict)
        backprop[val <= 0] = 0
        self.arg.deriv(feed_dict, backprop)


class RegMatOp(_BaseOp):
    def __init__(self, arg, reg, sess):
        super(RegMatOp, self).__init__(sess)
        self.shape = 1, 1
        self.arg = arg
        self.reg = reg

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            val = self.arg.eval(feed_dict)
            self.sess.vars[id(self)] = .5 * self.reg * np.sum(val * val)
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(RegMatOp, self).deriv(feed_dict, backprop)
        val = self.arg.eval(feed_dict)
        self.arg.deriv(feed_dict, backprop * self.reg * val)


class AddOp(_BaseOp):
    def __init__(self, arg1, arg2, sess):
        super(AddOp, self).__init__(sess)
        self.shape = arg1.shape
        self.arg1 = arg1
        self.arg2 = arg2

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            self.sess.vars[id(self)] = self.arg1.eval(feed_dict) + self.arg2.eval(feed_dict)
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(AddOp, self).deriv(feed_dict, backprop)
        self.arg1.deriv(feed_dict, backprop)
        self.arg2.deriv(feed_dict, backprop) 


class MulOp(_BaseOp):
    def __init__(self, arg1, arg2, sess):
        super(MulOp, self).__init__(sess)
        self.shape = arg1.shape[0], arg2.shape[1]
        self.arg1 = arg1
        self.arg2 = arg2

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            self.sess.vars[id(self)] = np.dot(self.arg1.eval(feed_dict), self.arg2.eval(feed_dict))
        return self.sess.vars[id(self)]

    def deriv(self, feed_dict, backprop):
        super(MulOp, self).deriv(feed_dict, backprop)
        arg1_val = self.arg1.eval(feed_dict)
        arg2_val = self.arg2.eval(feed_dict)
        self.arg1.deriv(feed_dict, np.dot(backprop, arg2_val.T))
        self.arg2.deriv(feed_dict, np.dot(arg1_val.T, backprop))


class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
    def __init__(self, logits, labels, sess):
        super(SoftmaxCrossEntropyWithLogitsOp, self).__init__(sess)
        self.shape = 1, 1
        self.logits = logits
        self.labels = labels

    def eval(self, feed_dict):
        if id(self) not in self.sess.vars:
            probs = self._eval_softmax(feed_dict)
            cross_entropy = -np.log(probs[range(probs.shape[0]), self.labels])
            cross_entropy = cross_entropy.mean()
            self.sess.vars[id(self)] = cross_entropy
        return self.sess.vars[id(self)]
   
    def _eval_softmax(self, feed_dict):
        logits = self.logits.eval(feed_dict)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def deriv(self, feed_dict, backprop):
        super(SoftmaxCrossEntropyWithLogitsOp, self).deriv(feed_dict, backprop)
        probs = self._eval_softmax(feed_dict)
        dscores = probs
        dscores[range(dscores.shape[0]), self.labels] -= 1.
        dscores /= dscores.shape[0]
        self.logits.deriv(feed_dict, backprop * dscores)


class Session(object):
    def __init__(self):
        self.vars = {}
        self.derivs = {}

    def reset(self):
        self.vars = {}
        self.derivs = {}
