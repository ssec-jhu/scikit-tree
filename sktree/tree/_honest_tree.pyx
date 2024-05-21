# Adapted from: sklearn/tree/_tree.pyx

from cpython cimport Py_INCREF, PyObject, PyTypeObject
from cython.operator cimport dereference as deref
from libc.math cimport isnan
from libc.stdint cimport INTPTR_MAX
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset
from libcpp cimport bool
from libcpp.algorithm cimport pop_heap, push_heap
from libcpp.vector cimport vector

import struct

import numpy as np

cimport numpy as cnp

cnp.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from sklearn.tree._tree import Tree
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport float64_t, intp_t, uint8_t, uint32_t, uint64_t

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                intp_t nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, intp_t flags, object obj)
    intp_t PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

cdef extern from "<stack>" namespace "std" nogil:
    cdef cppclass stack[T]:
        ctypedef T value_type
        stack() except +
        bint empty()
        void pop()
        void push(T&) except +  # Raise c++ exception for bad_alloc -> MemoryError
        T& top()

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef float64_t INFINITY = np.inf
cdef float64_t EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef bint IS_FIRST = 1
cdef bint IS_NOT_FIRST = 0
cdef bint IS_LEFT = 1
cdef bint IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED



# ==============================================================================
# HonestTree
# ==============================================================================

cdef class HonestTreeBuilder(TreeBuilder):
    """Interface for honest tree builders."""

    cpdef initialize_node_queue(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const uint8_t[:] honest_indicator, # bool is a 8bit int, but this could/should be optimized?
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        """Build a decision tree from the training set (X, y)."""
        pass

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(
        self,
        Splitter splitter,
        intp_t min_samples_split,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        intp_t max_depth,
        float64_t min_impurity_decrease,
        unsigned char store_leaf_values=False,
        cnp.ndarray initial_roots=None,
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.store_leaf_values = store_leaf_values
        self.initial_roots = initial_roots

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return(DepthFirstTreeBuilder, (self.splitter,
                                       self.min_samples_split,
                                       self.min_samples_leaf,
                                       self.min_weight_leaf,
                                       self.max_depth,
                                       self.min_impurity_decrease,
                                       self.store_leaf_values,
                                       self.initial_roots))

    cpdef initialize_node_queue(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        """Initialize a list of roots"""
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # organize samples by decision paths
        paths = tree.decision_path(X)
        cdef intp_t PARENT
        cdef intp_t CHILD
        cdef intp_t i
        false_roots = {}
        X_copy = {}
        y_copy = {}
        for i in range(X.shape[0]):
            # collect depths from the node paths
            depth_i = paths[i].indices.shape[0] - 1
            PARENT = depth_i - 1
            CHILD = depth_i

            # find leaf node's & their parent node's IDs
            if PARENT < 0:
                parent_i = 0
            else:
                parent_i = paths[i].indices[PARENT]
            child_i = paths[i].indices[CHILD]
            left = 0
            if tree.children_left[parent_i] == child_i:
                left = 1  # leaf node is left child

            # organize samples by the leaf they fall into (false root)
            # leaf nodes are marked by parent node and
            # their relative position (left or right child)
            if (parent_i, left) in false_roots:
                false_roots[(parent_i, left)][0] += 1
                X_copy[(parent_i, left)].append(X[i])
                y_copy[(parent_i, left)].append(y[i])
            else:
                false_roots[(parent_i, left)] = [1, depth_i]
                X_copy[(parent_i, left)] = [X[i]]
                y_copy[(parent_i, left)] = [y[i]]

        X_list = []
        y_list = []

        # reorder the samples according to parent node IDs
        for key, value in reversed(sorted(X_copy.items())):
            X_list = X_list + value
            y_list = y_list + y_copy[key]
        cdef object X_new = np.array(X_list)
        cdef cnp.ndarray y_new = np.array(y_list)

        # initialize the splitter using sorted samples
        cdef Splitter splitter = self.splitter
        splitter.init(X_new, y_new, sample_weight, missing_values_in_feature_mask)

        # convert dict to numpy array and store value
        self.initial_roots = np.array(list(false_roots.items()))

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        cdef unsigned char store_leaf_values = self.store_leaf_values
        cdef cnp.ndarray initial_roots = self.initial_roots

        # Initial capacity
        cdef intp_t init_capacity
        cdef bint first = 0
        if initial_roots is None:
            # Recursive partition (without actual recursion)
            splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

            if tree.max_depth <= 10:
                init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
            else:
                init_capacity = 2047

            tree._resize(init_capacity)
            first = 1
        else:
            # convert numpy array back to dict
            false_roots = {}
            for key_value_pair in initial_roots:
                false_roots[tuple(key_value_pair[0])] = key_value_pair[1]

            # reset the root array
            self.initial_roots = None

        cdef intp_t start = 0
        cdef intp_t end = 0
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef intp_t node_id
        cdef float64_t right_child_min, left_child_min, right_child_max, left_child_max

        cdef SplitRecord split
        cdef SplitRecord* split_ptr = <SplitRecord *>malloc(splitter.pointer_size())

        cdef float64_t impurity = INFINITY
        cdef float64_t lower_bound
        cdef float64_t upper_bound
        cdef float64_t middle_value
        cdef intp_t n_constant_features
        cdef bint is_leaf
        cdef intp_t max_depth_seen = -1 if first else tree.max_depth

        cdef intp_t rc = 0

        cdef stack[StackRecord] builder_stack
        cdef stack[StackRecord] update_stack
        cdef StackRecord stack_record

        if not first:
            # push reached leaf nodes onto stack
            for key, value in reversed(sorted(false_roots.items())):
                end += value[0]
                update_stack.push({
                    "start": start,
                    "end": end,
                    "depth": value[1],
                    "parent": key[0],
                    "is_left": key[1],
                    "impurity": tree.impurity[key[0]],
                    "n_constant_features": 0,
                    "lower_bound": -INFINITY,
                    "upper_bound": INFINITY,
                })
                start += value[0]
        else:
            # push root node onto stack
            builder_stack.push({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_features": 0,
                "lower_bound": -INFINITY,
                "upper_bound": INFINITY,
            })

        with nogil:
            while not update_stack.empty():
                stack_record = update_stack.top()
                update_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                lower_bound = stack_record.lower_bound
                upper_bound = stack_record.upper_bound

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                # impurity == 0 with tolerance due to rounding errors
                is_leaf = is_leaf or impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(
                        impurity,
                        split_ptr,
                        &n_constant_features,
                        lower_bound,
                        upper_bound
                    )

                    # assign local copy of SplitRecord to assign
                    # pos, improvement, and impurity scores
                    split = deref(split_ptr)

                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._update_node(parent, is_left, is_leaf,
                                            split_ptr, impurity, n_node_samples,
                                            weighted_n_node_samples,
                                            split.missing_go_to_left)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)
                if splitter.with_monotonic_cst:
                    splitter.clip_node_value(tree.value + node_id * tree.value_stride, lower_bound, upper_bound)

                if not is_leaf:
                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[split.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = lower_bound
                        left_child_max = right_child_max = upper_bound
                    elif splitter.monotonic_cst[split.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = lower_bound
                        right_child_max = upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        right_child_min = middle_value
                        left_child_max = middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = lower_bound
                        left_child_max = upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        left_child_min = middle_value
                        right_child_max = middle_value

                    # Push right child on stack
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": n_constant_features,
                        "lower_bound": right_child_min,
                        "upper_bound": right_child_max,
                    })

                    # Push left child on stack
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": n_constant_features,
                        "lower_bound": left_child_min,
                        "upper_bound": left_child_max,
                    })
                elif store_leaf_values and is_leaf:
                    # copy leaf values to leaf_values array
                    splitter.node_samples(tree.value_samples[node_id])

                if depth > max_depth_seen:
                    max_depth_seen = depth

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                lower_bound = stack_record.lower_bound
                upper_bound = stack_record.upper_bound

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first=0

                # impurity == 0 with tolerance due to rounding errors
                is_leaf = is_leaf or impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(
                        impurity,
                        split_ptr,
                        &n_constant_features,
                        lower_bound,
                        upper_bound
                    )

                    # assign local copy of SplitRecord to assign
                    # pos, improvement, and impurity scores
                    split = deref(split_ptr)

                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split_ptr,
                                         impurity, n_node_samples,
                                         weighted_n_node_samples, split.missing_go_to_left)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)
                if splitter.with_monotonic_cst:
                    splitter.clip_node_value(tree.value + node_id * tree.value_stride, lower_bound, upper_bound)

                if not is_leaf:
                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[split.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = lower_bound
                        left_child_max = right_child_max = upper_bound
                    elif splitter.monotonic_cst[split.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = lower_bound
                        right_child_max = upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        right_child_min = middle_value
                        left_child_max = middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = lower_bound
                        left_child_max = upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        left_child_min = middle_value
                        right_child_max = middle_value

                    # Push right child on stack
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": n_constant_features,
                        "lower_bound": right_child_min,
                        "upper_bound": right_child_max,
                    })

                    # Push left child on stack
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": n_constant_features,
                        "lower_bound": left_child_min,
                        "upper_bound": left_child_max,
                    })
                elif store_leaf_values and is_leaf:
                    # copy leaf values to leaf_values array
                    splitter.node_samples(tree.value_samples[node_id])

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        # free the memory created for the SplitRecord pointer
        free(split_ptr)

        if rc == -1:
            raise MemoryError()