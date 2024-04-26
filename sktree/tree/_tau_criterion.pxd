from .._lib.sklearn.tree._splitter cimport (
    SplitConditionParameters,
    SplitConditionFunction,
    SplitConditionTuple,
    SplitCondition
    )

# from .._lib.sklearn.tree._criterion import Criterion


# cdef class TauCriterion(Criterion):
#     cdef intp_t[:] estimation_indices

#     cdef float64_t[:1] n_total_treat
#     cdef float64_t[:1] n_left_treat
#     cdef float64_t[:1] n_right_treat
#     cdef float64_t[:1] n_missing_treat

#     cdef float64_t[:1] n_total_control
#     cdef float64_t[:1] n_left_control
#     cdef float64_t[:1] n_right_control
#     cdef float64_t[:1] n_missing_control

#     cdef float64_t[::1] sq_sum_total_treat
#     cdef float64_t[::1] sq_sum_left_treat
#     cdef float64_t[::1] sq_sum_right_treat
#     cdef float64_t[::1] sq_sum_missing_treat

#     cdef float64_t[::1] sq_sum_total_control
#     cdef float64_t[::1] sq_sum_left_control
#     cdef float64_t[::1] sq_sum_right_control
#     cdef float64_t[::1] sq_sum_missing_control

#     cdef float64_t[::1] sum_total_treat    # The sum of w*y.
#     cdef float64_t[::1] sum_left_treat     # Same as above, but for the left side of the split
#     cdef float64_t[::1] sum_right_treat    # Same as above, but for the right side of the split
#     cdef float64_t[::1] sum_missing_treat  # Same as above, but for missing values in X

#     cdef float64_t[::1] sum_total_control    # The sum of (1 - w)*y.
#     cdef float64_t[::1] sum_left_control     # Same as above, but for the left side of the split
#     cdef float64_t[::1] sum_right_control    # Same as above, but for the right side of the split
#     cdef float64_t[::1] sum_missing_control  # Same as above, but for missing values in X
