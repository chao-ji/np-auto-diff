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

def get_combinations(lists):
  """Generates lists of combinations. Example:
  lists = [[1, 2], ['a', 'b', 'c'], ['A', 'B']] 

  results:
    [[1, 'a', 'A'],
     [1, 'a', 'B'],
     [1, 'b', 'A'],
     [1, 'b', 'B'],
     [1, 'c', 'A'],
     [1, 'c', 'B'],
     [2, 'a', 'A'],
     [2, 'a', 'B'],
     [2, 'b', 'A'],
     [2, 'b', 'B'],
     [2, 'c', 'A'],
     [2, 'c', 'B']]

  Args:
    lists: a list of lists, each list in lists holds the choices to be 
      considered in one dimension.

  Returns:
     a list of lists, each list in lists holds one combination.
  """

  if len(lists) == 0: return []
  if len(lists) == 1: return [(_,) for _ in lists[0]]
  partial_results = get_combinations(lists[1:])
  results = []
  for x in lists[0]:
    for y in partial_results:
      new_combo = [x]
      new_combo.extend(y)
      results.append(new_combo)
  return results


def get_subsets(items, min_size=1):
  """Generates subsets from a given set of items.

  items = [1, 2, 3]

  subsets:
    [[3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]

  Args:
    items: a list of set items.
    min_size: int scalar, minimum size of the subset.

  Returns
    a list of lists, each list in lists holds one subset.
  """
  indicators_list = get_combinations([[0, 1]] * len(items))

  if min_size:
    indicators_list = list(filter(
        lambda indicators: sum(indicators) >= min_size, indicators_list))
  
  subsets = []
  for indicators in indicators_list:
    subsets.append(list(map(lambda p: p[1], 
                            filter(lambda p: p[0], zip(indicators, items)))))
  return subsets
