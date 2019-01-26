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
