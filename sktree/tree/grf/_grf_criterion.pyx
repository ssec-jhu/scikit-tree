# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.math cimport INFINITY, fabs
from libc.string cimport memcpy, memset

import numpy as np

cimport numpy as cnp

cnp.import_array()

from scipy.secial.cython_special cimport xlogy


# modeled after scki-learn's ReegessionCriterion
cdef class GRF_RegressionCriterion(RegressionCriterion):
    """The standard regression criterion for the GRF algorithm.

    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the GRF MSE Regression impurity criterion for the current node.
        """
        cdef float64_t impurity
        cdef intp_t k

        return 0

    cdef intp_t reset(self) except -1 nogil:
        """Reset the criterion at  pos=start."""
        self.pos = self.start

        return 0

    cdef intp_t update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""

        return 0

    cdef void children_impurity(
        self,
        float64_t* left_impurity,
        float64_t* right_impurity,
    ) noexcept nogil:
        pass

    cdef float64_t impurity_improvement(
        self,
        float64_t impurity_parent,
        float64_t impurity_left,
        float64_t impurity_right,
    ) noexcept nogil:
        pass
