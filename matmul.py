import math
from itertools import cycle, islice
import numpy as np


def shift(row, offset):

    offset = offset % len(row)
    return np.concatenate((row[offset:], row[:offset]))


def sigma(arr):

    assert(arr.ndim == 2, "shiftrows error: dimensions must be 2")

    rows, _ = arr.shape
    
    for idx in range(rows):
        
        arr[idx] = shift(arr[idx], idx)

    return arr


def theta(arr):

    assert(arr.ndim == 2, "shiftcols error: dimensions must be 2")

    _, cols = arr.shape
    
    for idx in range(cols):
        
        arr[:,idx] = shift(arr[:,idx], idx)

    return arr


def epsilon(arr, size, offset=0):

    assert(arr.ndim == 2, "repeatrows error: dimensions must be 2")

    rows, _ = arr.shape

    result = np.empty((rows, size))

    for idx in range(rows):

        result[idx] = np.array(list(
            islice(cycle(shift(arr[idx], offset)), size)
        ))

    return result


def omega(arr, size, offset=0):

    assert(arr.ndim == 2, "repeatcols error: dimensions must be 2")

    _, cols = arr.shape

    result = np.empty((size, cols))

    for idx in range(cols):

        result[:, idx] = np.array(list(
            islice(cycle(shift(arr[:, idx], offset)), size)
        ))

    return result


def matmul(a, b):

    assert(a.ndim == 2, "matmul error: dimensions must be 2")
    assert(b.ndim == 2, "matmul error: dimensions must be 2")

    m, l = a.shape
    l, n = b.shape

    arr = np.zeros((m, n))

    for k in range(l):
        lhs = epsilon(sigma(a.copy()), n, offset=k)
        rhs = omega(theta(b.copy()), m, offset=k)
        addend = np.multiply(lhs, rhs)
        arr+=addend

    return arr


def sigma_transformation_matrix(m, l):

    transformation_matrix = np.zeros((m*l, m*l))

    for h in range(m*l):
        for i in range(m):
            for j in range(l):

                if h == i + ((i+j) % l) * m:
                    transformation_matrix[i+j*m, h] = 1

    return transformation_matrix


def theta_transformation_matrix(l, n):

    transformation_matrix = np.zeros((l*n, l*n))

    for h in range(l*n):
        for i in range(l):
            for j in range(n):

                if h == ((i+j) % l) + j * l:
                    transformation_matrix[i+j*l, h] = 1

    return transformation_matrix

def epsilon_transformation_matrix(m, n, l, offset = 0):

    transformation_matrix = np.zeros((m*n, m*l))

    for i in range(m*n):
        for j in range(m*l):

            if j == (offset * m + i) % (m*l):
                transformation_matrix[i, j] = 1

    return transformation_matrix


def omega_transformation_matrix(m, n, l, offset = 0):

    transformation_matrix = np.zeros((m*n, n*l))

    for i in range(m*n):
        for j in range(n*l):

            if j == ((offset + (i % m)) % l) + math.floor(i/m) * l:
                transformation_matrix[i, j] = 1

    return transformation_matrix
