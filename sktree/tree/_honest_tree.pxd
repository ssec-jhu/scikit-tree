

# based on sklearn/tree/_tree.pxd
# see honest_tree.pyx for details

import numpy as np

cimport numpy as cnp
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from ._utils cimport UINT32_t, bool_t
from sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._tree cimport Node, Tree, TreeBuilder
from .._lib.sklearn.utils._typedefs cimport uint8_t,float64_t

cdef class HonestTree(Tree):
    cdef vector[uint8_t] honest_indicator  # array of bools pointing on Yi's used to decide where to place the splits

    # overridden methods
    cpdef initialize_node_queue(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const uint8_t[:] honest_indicator, # bool is a 8bit int, but this could/should be optimized?
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ) except -1 nogil

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const uint8_t[:] honest_indicator, # bool is a 8bit int, but this could/should be optimized?
        const float64_t[:] sample_weight=*,
        const unsigned char[::1] missing_values_in_feature_mask=*,
    )

    cpdef cnp.ndarray get_projection_matrix(self)