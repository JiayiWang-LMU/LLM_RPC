# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# _highs_solver.pxd
# This file contains the Cython declarations for the HiGHS C API functions

from ._types cimport DTYPE_t_1D

# Declare the HighsSolver class
cdef class HighsSolver:
    cdef void* highs
    cdef int* startIndex
    cdef double* rowLower
    cdef int n_rows
    cdef int n_cols

    cdef double* c
    cdef double* b
    cdef double* lb
    cdef double* ub

    # Constructor: Initializes HiGHS instance and preallocates memory for indices, values, etc.

    # Method: Solve the linear programming problem
    cdef void solve_lp(self, double[:, :] A, DTYPE_t_1D x) nogil
