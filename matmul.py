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


a = np.arange(8*32).reshape(32,8)
b = np.arange(8*32).reshape(8,32)
print(np.matmul(a, b))
print(matmul(a, b))

