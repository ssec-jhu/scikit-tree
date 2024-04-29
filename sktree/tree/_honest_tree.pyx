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
# HonestTreeBuilder
# ==============================================================================

cdef class HonestTreeBuilder:
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