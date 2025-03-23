# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Import necessary components from Cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
cimport numpy as np

# Import HiGHS C API declarations from the .pxd file
cdef extern from "highs_c_api.h":
    # HiGHS related function declarations
    cdef void* Highs_create() nogil
    cdef void Highs_destroy(void* highs) nogil
    cdef int Highs_run(void* highs) nogil
    cdef int Highs_addCols(void* highs, int numCol, const double* colCost, const double* colLower, const double* colUpper, int numNZ, const int* startIndex, const int* indices, const double* values) nogil
    cdef int Highs_addRows(void* highs, int numRow, const double* rowLower, const double* rowUpper, int numNZ, const int* startIndex, const int* indices, const double* values) nogil
    cdef int Highs_changeCoeff(void* highs, int row, int col, double value) nogil
    cdef int Highs_getSolution(const void* highs, double* colValue, double* rowValue, double* dualValue, double* rowDual) nogil
    cdef int Highs_setStringOptionValue(void* highs, const char* option, const char* value) nogil
    cdef int Highs_setIntOptionValue(void* highs, const char* option, int value) nogil
    cdef int Highs_setDoubleOptionValue(void* highs, const char* option, double value) nogil
    cdef int Highs_clearModel(void* highs) nogil

cdef class HighsSolver:

    def __init__(self, int n_cols, int n_rows):
        """Constructor that initializes the HiGHS instance and memory allocations."""
        # Store the number of rows and columns
        self.n_cols = n_cols
        self.n_rows = n_rows

        # Preallocate memory for the HiGHS solver based on the number of variables and constraints
        self.startIndex = <int*> malloc(n_rows * sizeof(int))
        self.rowLower = <double*> malloc(n_rows * sizeof(double))

        # Check for memory allocation failures
        if self.startIndex == NULL or self.rowLower == NULL:
            raise MemoryError("Memory allocation failed.")

        self.c = <double*> malloc(n_cols * sizeof(double))
        self.b = <double*> malloc(n_rows * sizeof(double))
        self.lb = <double*> malloc(n_cols * sizeof(double))
        self.ub = <double*> malloc(n_cols * sizeof(double))

        # Initialize values in the pointers
        for row in range(n_rows):
            self.rowLower[row] = -INFINITY
            self.b[row] = 0.0
        self.rowLower[n_rows - 1] = 1.0
        self.b[n_rows - 1] = 1.0

        for col in range(n_cols):
            self.c[col] = 1.0
            self.lb[col] = 0.0
            self.ub[col] = 1.0

        # Create high instance
        self.highs = Highs_create()
        if not self.highs:
            raise MemoryError("Failed to create HiGHS instance.")

        # Set options for HiGHS (optional)
        Highs_setStringOptionValue(self.highs, b"output_flag", b"false")  # Disable output
        Highs_setStringOptionValue(self.highs, b"presolve", b"off")  # Disable presolve

        # Add columns and rows initially
        Highs_addCols(self.highs, self.n_cols, self.c, self.lb, self.ub, 0, NULL, NULL, NULL)
        Highs_addRows(self.highs, self.n_rows, self.rowLower, self.b, 0, NULL, NULL, NULL)

    def __dealloc__(self):
        """Destructor to free the memory and destroy the HiGHS instance."""
        if self.startIndex is not NULL:
            free(self.startIndex)
        if self.rowLower is not NULL:
            free(self.rowLower)
        if self.c is not NULL:
            free(self.c)
        if self.b is not NULL:
            free(self.b)
        if self.lb is not NULL:
            free(self.lb)
        if self.ub is not NULL:
            free(self.ub)
        if self.highs is not NULL:
            Highs_destroy(self.highs)

    cdef void solve_lp(self, double[:, :] A, DTYPE_t_1D x) nogil:
        """Solve the linear programming problem using HiGHS."""
        cdef int row, col
        cdef double* colValue
        cdef int status

        # Loop through matrix A and update individual coefficients using Highs_changeCoeff
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                Highs_changeCoeff(self.highs, row, col, A[row, col])

        # Solve the LP
        status = Highs_run(self.highs)
        if status != 0:
            raise RuntimeError("HiGHS failed to solve the problem.")

        # Retrieve the solution
        colValue = &x[0]
        Highs_getSolution(self.highs, colValue, NULL, NULL, NULL)
