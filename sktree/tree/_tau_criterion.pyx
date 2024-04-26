# cdef class TauCriterion(Criterion):
#     def __cinit__(self, intp_t n_outputs, intp_t n_samples):
#         """Initialize parameters for this criterion.

#         Parameters
#         ----------
#         n_outputs : intp_t
#             The number of targets to be predicted
#         """
#         # Default values
#         self.start = 0
#         self.pos = 0
#         self.end = 0

#         self.n_outputs = n_outputs
#         self.n_samples = n_samples
#         self.n_node_samples = 0
#         self.weighted_n_node_samples = 0.0
#         self.weighted_n_left = 0.0
#         self.weighted_n_right = 0.0
#         self.weighted_n_missing = 0.0

#         self.n_total_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.n_left_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.n_right_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.n_missing_treat = np.zeros(n_outputs, dtype=np.float64)

#         self.n_total_control = np.zeros(n_outputs, dtype=np.float64)
#         self.n_left_control = np.zeros(n_outputs, dtype=np.float64)
#         self.n_right_control = np.zeros(n_outputs, dtype=np.float64)
#         self.n_missing_control = np.zeros(n_outputs, dtype=np.float64)

#         self.sq_sum_total_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_left_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_right_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_missing_treat = np.zeros(n_outputs, dtype=np.float64)

#         self.sq_sum_total_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_left_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_right_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sq_sum_missing_control = np.zeros(n_outputs, dtype=np.float64)

#         self.sum_total_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_left_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_right_treat = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_missing_treat = np.zeros(n_outputs, dtype=np.float64)

#         self.sum_total_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_left_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_right_control = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_missing_control = np.zeros(n_outputs, dtype=np.float64)

#     def __reduce__(self):
#         return (type(self), (self.n_outputs), self.__getstate__())

#     cdef intp_t init(
#         self,
#         const float64_t[:, ::1] y,
#         const float64_t[:] sample_weight,
#         float64_t weighted_n_samples,
#         const intp_t[:] sample_indices,
#         const intp_t[:] estimation_indices,
#     ) except -1 nogil:
#         """Placeholder for a method which will initialize the criterion.

#         Returns -1 in case of failure to allocate memory (and raise MemoryError)
#         or 0 otherwise.

#         Parameters
#         ----------
#         y : ndarray, dtype=float64_t
#             y is a buffer that can store values for n_outputs target variables
#             stored as a Cython memoryview.
#         sample_weight : ndarray, dtype=float64_t
#             The weight of each sample stored as a Cython memoryview.
#         weighted_n_samples : float64_t
#             The total weight of the samples being considered
#         sample_indices : ndarray, dtype=intp_t
#             A mask on the samples. Indices of the samples in X and y we want to use,
#             where sample_indices[start:end] correspond to the training samples in this node.
#         estimation_indices : ndarray, dtype=intp_t
#             A mask on the samples. Indices of the samples in X and y we want to use,
#             where estimation_indices[start:end] correspond to the estimation samples in this node.
#         """
#         self.w = y[:, 0]
#         self.y = y[:, 1:]
#         self.sample_weight = sample_weight
#         self.weighted_n_samples = weighted_n_samples
#         self.sample_indices = sample_indices
#         self.estimation_indices = estimation_indices

#         return 0


#     cdef void set_sample_pointers(
#         self,
#         intp_t start,
#         intp_t end
#     ) noexcept nogil:
#         """Set sample pointers in the criterion."""
#         self.start = start
#         self.end = end

#         self.n_node_samples = end - start

#         self.weighted_n_node_samples = 0.

#         cdef intp_t i
#         cdef intp_t p
#         cdef intp_t k
#         cdef float64_t y_ik
#         cdef float64_t w_y_ik
#         cdef float64_t w = 1.0
#         memset(&self.sq_sum_total_treat[0], 0, self.n_outputs * sizeof(float64_t))
#         memset(&self.sq_sum_total_control[0], 0, self.n_outputs * sizeof(float64_t))
#         memset(&self.sum_total_treat[0], 0, self.n_outputs * sizeof(float64_t))
#         memset(&self.sum_total_control[0], 0, self.n_outputs * sizeof(float64_t))

#         for p in range(start, end):
#             i = self.sample_indices[p]

#             if self.sample_weight is not None:
#                 w = self.sample_weight[i]

#             for k in range(self.n_outputs):
#                 w_i = self.w[i]
#                 y_ik = self.y[i, k]
#                 w_y_ik = w * y_ik
#                 self.sum_total_treat[k] += w_i * w_y_ik
#                 self.sq_sum_total_treat[k] += w_i * w_y_ik * y_ik
#                 self.sum_total_control[k] += (1 - w_i) * w_y_ik
#                 self.sq_sum_total_control[k] += (1 - w_i) * w_y_ik * y_ik

#             self.weighted_n_node_samples += w

#         # Reset to pos=start
#         self.reset()
