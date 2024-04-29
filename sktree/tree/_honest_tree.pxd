

# based on sklearn/tree/_tree.pxd
# see honest_tree.pyx for details

import numpy as np

cimport numpy as cnp
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from ._utils cimport UINT32_t, bool_t
from sklearn.tree._splitter cimport SplitRecord, Splitter