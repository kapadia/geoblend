
# Script to check if OpenMP is available by the compiler used by setup.py

cimport openmp

def check():
    cdef int max_threads = openmp.omp_get_max_threads()
    cdef int x
    
    return max_threads