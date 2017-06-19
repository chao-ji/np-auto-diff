import numpy as np

class _BaseOp(object):
    def __init__(self, sess):
        self.sess = sess
        self.shape = None
        self.parent_total = 0
        self.parent_count = 0
        self.sess.vars.append(self)

    def deriv(self, feed_dict, backprop):
        if id(self) not in self.sess.derivs:
            self.sess.derivs[id(self)] = np.zeros(self.shape)
        self.sess.derivs[id(self)] += backprop


class PlaceholderOp(_BaseOp):
    def __init__(self, shape, sess):
        super(PlaceholderOp, self).__init__(sess)
        self.shape = shape

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            self.sess.vals[id(self)] = feed_dict[self]
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(PlaceholderOp, self).deriv(feed_dict, backprop)
        self.parent_count += 1

class SigmoidOp(_BaseOp):
    def __init__(self, arg, sess):
        super(SigmoidOp, self).__init__(sess)
        self.shape = arg.shape
        self.arg = arg
        self.arg.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            self.sess.vals[id(self)] = 1. / (1. + np.exp(-self.arg.eval(feed_dict)))
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(SigmoidOp, self).deriv(feed_dict, backprop)
        self.parent_count +=1
        if self.parent_count == self.parent_total:
            sig = self.eval(feed_dict)
            sig = sig * (1. - sig)
            deriv_val = self.sess.derivs[id(self)]
            self.arg.deriv(feed_dict, deriv_val * sig)


class ReLuOp(_BaseOp):
    def __init__(self, arg, sess):
        super(ReLuOp, self).__init__(sess)
        self.shape = arg.shape
        self.arg = arg
        self.arg.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            self.sess.vals[id(self)] = np.maximum(0, self.arg.eval(feed_dict))
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(ReLuOp, self).deriv(feed_dict, backprop)
        self.parent_count += 1
        if self.parent_count == self.parent_total:
            val = self.arg.eval(feed_dict)
            deriv_val = self.sess.derivs[id(self)]
            deriv_val[val <= 0] = 0
            self.arg.deriv(feed_dict, deriv_val)


class RegMatOp(_BaseOp):
    def __init__(self, arg, reg, sess):
        super(RegMatOp, self).__init__(sess)
        self.shape = 1, 1
        self.arg = arg
        self.reg = reg
        self.arg.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            val = self.arg.eval(feed_dict)
            self.sess.vals[id(self)] = .5 * self.reg * np.sum(val * val)
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(RegMatOp, self).deriv(feed_dict, backprop)
        self.parent_count += 1
        if self.parent_count == self.parent_total:
            val = self.arg.eval(feed_dict)
            deriv_val = self.sess.derivs[id(self)]
            self.arg.deriv(feed_dict, deriv_val * self.reg * val)


class AddOp(_BaseOp):
    def __init__(self, arg1, arg2, sess):
        super(AddOp, self).__init__(sess)
        self.shape = arg1.shape
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1.parent_total += 1
        self.arg2.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            self.sess.vals[id(self)] = self.arg1.eval(feed_dict) + self.arg2.eval(feed_dict)
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(AddOp, self).deriv(feed_dict, backprop)
        self.parent_count += 1
        if self.parent_count == self.parent_total:
            deriv_val = self.sess.derivs[id(self)]
            self.arg1.deriv(feed_dict, deriv_val)
            self.arg2.deriv(feed_dict, deriv_val) 


class MulOp(_BaseOp):
    def __init__(self, arg1, arg2, sess):
        super(MulOp, self).__init__(sess)
        self.shape = arg1.shape[0], arg2.shape[1]
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1.parent_total += 1
        self.arg2.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            self.sess.vals[id(self)] = np.dot(self.arg1.eval(feed_dict), self.arg2.eval(feed_dict))
        return self.sess.vals[id(self)]

    def deriv(self, feed_dict, backprop):
        super(MulOp, self).deriv(feed_dict, backprop)
        self.parent_count +=1
        if self.parent_count == self.parent_total:
            deriv_val = self.sess.derivs[id(self)]
            arg1_val = self.arg1.eval(feed_dict)
            arg2_val = self.arg2.eval(feed_dict)
            self.arg1.deriv(feed_dict, np.dot(deriv_val, arg2_val.T))
            self.arg2.deriv(feed_dict, np.dot(arg1_val.T, deriv_val))


class SoftmaxCrossEntropyWithLogitsOp(_BaseOp):
    def __init__(self, logits, labels, sess):
        super(SoftmaxCrossEntropyWithLogitsOp, self).__init__(sess)
        self.shape = 1, 1
        self.logits = logits
        self.labels = labels
        self.logits.parent_total += 1

    def eval(self, feed_dict):
        if id(self) not in self.sess.vals:
            probs = self._eval_softmax(feed_dict)
            cross_entropy = -np.log(probs[range(probs.shape[0]), self.labels])
            cross_entropy = cross_entropy.mean()
            self.sess.vals[id(self)] = cross_entropy
        return self.sess.vals[id(self)]
   
    def _eval_softmax(self, feed_dict):
        logits = self.logits.eval(feed_dict)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def deriv(self, feed_dict, backprop):
        super(SoftmaxCrossEntropyWithLogitsOp, self).deriv(feed_dict, backprop)
        self.parent_count += 1
        if self.parent_count == self.parent_total:
            deriv_val = self.sess.derivs[id(self)]
            probs = self._eval_softmax(feed_dict)
            dscores = probs
            dscores[range(dscores.shape[0]), self.labels] -= 1.
            dscores /= dscores.shape[0]
            self.logits.deriv(feed_dict, deriv_val * dscores)


class Session(object):
    def __init__(self):
        self.vals = {}
        self.derivs = {}
        self.vars = []

    def reset(self):
        self.vals = {}
        self.derivs = {}
        for tensor in self.vars:
            tensor.parent_count = 0
