from .._lib.sklearn.tree._splitter cimport (
    SplitConditionParameters,
    SplitConditionFunction,
    SplitConditionTuple,
    SplitCondition,
    Splitter,
    SplitRecord,
    intp_t,
    float64_t
    )

cdef bint min_estimator_sample_leaf_condition(
    Splitter splitter,
    SplitRecord* current_split,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionParameters split_condition_parameters
) noexcept nogil:
    # FOR NOW DON'T FUSS ABOUT missing_go_to_left
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef intp_t end_non_missing = splitter.end # - n_missing
    cdef intp_t n_left, n_right

    n_left = current_split.pos - splitter.start
    n_right = end_non_missing - current_split.pos # + n_missing

    # Reject if min_samples_leaf is not guaranteed
    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return False

    return True

cdef class MinSamplesLeafCondition(SplitCondition):
    pass

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
