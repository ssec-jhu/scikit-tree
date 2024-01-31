# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport numpy as np
import numpy as np

cnp.import_array()

cdef class GRF_RegressionCriterion(BaseCriterion):
    """Criterion for simple regression with Generalized Random Forests.

    """

cdef class GRF_ConditionalAveragePartialEffect(BaseCriterion):
    """Criterion for conditional average partial effect with Generalized Random Forests.

    """

cdef class GRF_InstrumentVariablesRegression(BaseCriterion):
    """Criterion for instrumental variables regression with Generalized Random Forests.

    """


