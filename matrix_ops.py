import math
import random
from itertools import cycle, islice
import numpy as np
from tqdm import tqdm


def shift(row, offset):

    offset = offset % len(row)
    return np.concatenate((row[offset:], row[:offset]))


def sigma(arr):

    rows, _ = arr.shape

    for idx in range(rows):

        arr[idx] = shift(arr[idx], idx)

    return arr


def theta(arr):

    _, cols = arr.shape

    for idx in range(cols):

        arr[:,idx] = shift(arr[:,idx], idx)

    return arr


def epsilon(arr, size, offset=0):

    rows, _ = arr.shape

    result = np.empty((rows, size))

    for idx in range(rows):

        result[idx] = np.array(list(
            islice(cycle(shift(arr[idx], offset)), size)
        ))

    return result


def omega(arr, size, offset=0):

    _, cols = arr.shape

    result = np.empty((size, cols))

    for idx in range(cols):

        result[:, idx] = np.array(list(
            islice(cycle(shift(arr[:, idx], offset)), size)
        ))

    return result


def hegmm(a, b):

    assert a.ndim == 2, "matmul error: a-matrix dimensions must be 2"
    assert b.ndim == 2, "matmul error: b-matrix dimensions must be 2"
    assert a.shape[1] == b.shape[0], "matmul error: dimensions incompatable"

    m, l = a.shape
    l, n = b.shape

    arr = np.zeros((m, n))

    for k in range(l):
        lhs = epsilon(sigma(a.copy()), n, offset=k)
        rhs = omega(theta(b.copy()), m, offset=k)
        addend = np.multiply(lhs, rhs)
        arr+=addend

    return arr


if __name__ == '__main__':

    print("Running test cases...")

    for i in tqdm(range(100)):
        m = random.randint(2,20)
        l = random.randint(2,20)
        n = random.randint(2,20)

        a = np.random.rand(m, l)
        b = np.random.rand(l, n)

        numpy_matmul = np.matmul(a.copy(), b.copy())
        hegmm_matmul = hegmm(a.copy(), b.copy())

        assert(np.allclose(numpy_matmul, hegmm_matmul))

    print("PASSED")
