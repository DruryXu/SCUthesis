import numpy
cimport numpy
cimport cython
def gradient_descent(double alpha, double lam_bda, int max_iter, numpy.ndarray[numpy.double_t, ndim = 2] P, numpy.ndarray[numpy.double_t, ndim = 2] Q, numpy.ndarray[numpy.int_t, ndim = 2] user_item_matrix, numpy.ndarray[numpy.double_t] rating_matrix):
    cdef int n = rating_matrix.shape[0]
    cdef int F = P.shape[1]
    cdef int step, i, user, item, f
    cdef double rui, eui

    for step in range(max_iter):
        print(step + 1)
        for i in range(n):
            user = user_item_matrix[i, 0]
            item = user_item_matrix[i, 1]
            rui = rating_matrix[i]

            eui = rui - P[user, :].dot(Q[:, item])
            for f in range(F):
                P[user, f] += alpha * (eui * Q[f, item] - lam_bda * P[user, f])
                Q[f, item] += alpha * (eui * P[user, f] - lam_bda * Q[f, item])

        alpha *= 0.9

    return P, Q
