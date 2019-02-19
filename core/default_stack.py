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


# _DEFAULT_STACK stores a stack of dict, where each dict has only one key-value 
# pair that holds the default graph.
# 
# There are two ways to define the default graph:
# 
# 1. When the first Node is defined, it's given a default graph that register
# itself in `_DEFAULT_STACK`. Example:
#
# On running a = Node()
#
# _DEFAULT_STACK becomes [{'graph': Graph()}]
#
# where a.graph == _DEFAULT_STACK[0]['graph']
#
# 2. One can explicitly define a graph, and create a scope in which all nodes 
# defined inside are bound to that graph. Example:
#
# graph = Graph()
#
# with graph.as_default_graph():
#   a = Node()
#
# where a.graph == graph == _DEFAULT_STACK[-1]['graph']
 
_DEFAULT_STACK = [{}]
