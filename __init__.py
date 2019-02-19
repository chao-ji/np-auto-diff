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
try:
  import numpy
except ImportError:
  raise ImportError('numpy must be installed for auto differentiation.')

from autodiff.exports import placeholder
from autodiff.exports import variable 
from autodiff.exports import constant 
from autodiff.exports import add 
from autodiff.exports import multiply 
from autodiff.exports import matmul 
from autodiff.exports import reshape 
from autodiff.exports import reduce_mean 
from autodiff.exports import reduce_sum 
from autodiff.exports import pad 
from autodiff.exports import sigmoid 
from autodiff.exports import relu 
from autodiff.exports import dropout
from autodiff.exports import fused_batch_norm 
from autodiff.exports import conv2d 
from autodiff.exports import maxpool2d 
from autodiff.exports import avgpool2d 
from autodiff.exports import l2norm 
from autodiff.exports import softmax_cross_entropy_loss
 
from autodiff.core.environments import Graph
from autodiff.core.environments import RunTime 
from autodiff.core.environments import get_default_graph
from autodiff import initializers
from autodiff import optimizers 
