import math
import copy
import numpy as np
import time

from seal import (
    EncryptionParameters, scheme_type, SEALContext, CKKSEncoder, KeyGenerator,
    Encryptor, Evaluator, Decryptor, CoeffModulus, Ciphertext
)
from seal_helpers import *
from linear_ops import *


# Define encryption parameters
parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 16384
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 40, 40, 60]))

# Define scaling factor
scale = 2.0 ** 40

# Create SEAL context
context = SEALContext(parms)
print_parameters(context)

# Initialize encoder, key generator, and keys
ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()
print(f'Number of slots: {slot_count}')

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_keys = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()

# Initialize encryptor, evaluator, and decryptor
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)


def get_general_diagonals(matrix):
    """
    Extracts all generalized diagonals from a matrix.

    Args:
        matrix (numpy.ndarray): A matrix from which the diagonals are extracted.

    Returns:
        list of numpy.ndarray: A list containing diagonals of the matrix.
    """
    tiles = np.tile(matrix, 2)
    return [np.diag(tiles, idx) for idx in range(matrix.shape[1])]


def encrypt_vector(vector):
    """
    Encrypts a vector using the predefined encryption scheme.

    Args:
        vector (numpy.ndarray): The vector to be encrypted.

    Returns:
        Ciphertext: The encrypted vector.
    """
    return encryptor.encrypt(ckks_encoder.encode(vector, scale))


def decrypt_vector(vector):
    """
    Decrypts and decodes a ciphertext vector.

    Args:
        vector (Ciphertext): The encrypted vector to be decrypted.

    Returns:
        list: The decrypted and decoded vector.
    """
    return ckks_encoder.decode(decryptor.decrypt(vector))


def rotate_vector(vector, steps):
    """
    Rotates (left) a ciphertext vector by a specified number of steps.

    Args:
        vector (Ciphertext): The encrypted vector to be rotated.
        steps (int): The number of steps to rotate the vector.

    Returns:
        Ciphertext: The rotated ciphertext vector.
    """
    return evaluator.rotate_vector(vector, -steps, galois_keys)


def tile_encrypt(vector, length):
    """
    Homomorphically encrypted equivelant of np.tile(arr, 2)

    Args:
        vector (Ciphertext): The encrypted vector to be tiled.
        length (int): The number of data elements in the original vector.

    Returns:
        Ciphertext: The same ciphertext repeated/tiled twice.
    """
    vector_rot = Ciphertext(vector)
    vector_rot = evaluator.rotate_vector(vector_rot, -length, galois_keys)
    return evaluator.add(vector, vector_rot)


def matrix_vector_multiplication(matrix, vector):
    """
    Performs matrix-vector multiplication on an encrypted vector.

    Args:
        matrix (numpy.ndarray): The matrix to be multiplied.
        vector (Ciphertext): The encrypted vector to be multiplied with the matrix.

    Returns:
        Ciphertext: The result of the matrix-vector multiplication, encrypted.
    """
    # Extract diagonals from the matrix
    diagonals = get_general_diagonals(matrix)

    # Encrypt each diagonal
    encrypted_diagonals = [encrypt_vector(diagonal) for diagonal in diagonals]

    # Check and perform modulus switching if necessary
    parms = vector.parms_id() if vector.parms_id() != encrypted_diagonals[0].parms_id() else None
    if parms:
        encrypted_diagonals = [
            evaluator.mod_switch_to(diagonal, parms) for diagonal in encrypted_diagonals
        ]

    # Initialize multiplication with the first diagonal
    init = evaluator.multiply(encrypted_diagonals[0], vector)
    init = evaluator.relinearize(init, relin_keys)

    # Accumulate results of subsequent diagonal multiplications
    addends = [init]
    for idx, diagonal in enumerate(encrypted_diagonals[1:]):
        addend = evaluator.multiply(
            diagonal,
            rotate_vector(vector, -(idx + 1))
        )
        addend = evaluator.relinearize(addend, relin_keys)
        addends.append(addend)

    # Sum all the accumulated results
    result = evaluator.add_many(addends)

    return result


def matrix_multiplication(a_plain, b_encrypt, m, l, n):
    """
    Performs matrix multiplication of two matrices.

    Args:
        a_plain (numpy.ndarray): The plaintext matrix to be multiplied.
        b_encrypt (Ciphertext): The encrypted matrix to be multiplied. Must be
                                flattened column-wise.
        m (int): The number of rows in the first matrix.
        l (int): The number of columns in the first matrix and rows in the second matrix.
        n (int): The number of columns in the second matrix.

    Returns:
        Ciphertext: The result of the matrix multiplication, encrypted.
    """
    # Encrypt and tile the second matrix
    b_encrypt = tile_encrypt(b_encrypt, l * n)

    # Compute the product of sigma transformation and the first matrix
    A = np.matmul(
        sigma_transformation_matrix(m, l).astype(int),
        a_plain.copy().flatten(order='F')
    )

    # Perform matrix-vector multiplication
    B = matrix_vector_multiplication(
        theta_transformation_matrix(l, n).astype(int),
        b_encrypt
    )

    B = evaluator.relinearize(B, relin_keys)
    B = tile_encrypt(B, l * n)

    # Initialize list to accumulate results
    addends = []

    for k in range(l):
        # Compute the left-hand side matrix multiplication
        lhs = np.matmul(
            epsilon_transformation_matrix(m, n, l, k).astype(int),
            A.copy()
        )

        # Compute the right-hand side matrix-vector multiplication
        rhs = matrix_vector_multiplication(
            omega_transformation_matrix(m, n, l, k).astype(int),
            B
        )

        # Encrypt and adjust parameters for lhs
        lhs = encrypt_vector(lhs)
        lhs = evaluator.mod_switch_to(lhs, rhs.parms_id())

        # Multiply encrypted matrices and accumulate results
        product = evaluator.multiply(lhs, rhs)
        product = evaluator.relinearize(product, relin_keys)
        addends.append(product)

    # Sum all accumulated products
    C = evaluator.add_many(addends)

    return C


def run_tests():
    """
    Runs matrix multiplication tests for square matrices of sizes from 4 to 10.
    Times the execution and ensures the result is close to the plaintext result.
    """
    sizes = range(4, 11)  # Matrix sizes from 4x4 to 10x10

    for size in sizes:
        # Generate test matrices
        A = np.arange(1, size * size + 1).astype(int).reshape(size, size)
        B = np.arange(1, size * size + 1).astype(int).reshape(size, size)

        # Display the matrices
        print(f"Matrix A ({size}x{size})...\n{A}\n")
        print(f"Matrix B ({size}x{size})...\n{B}\n")
        print(f"Plain result...\n{np.matmul(A, B)}\n")

        # Encrypt matrix B
        B_flat = B.flatten(order='F')
        B_encrypt = encrypt_vector(B_flat)

        # Measure execution time
        start_time = time.time()
        C_encrypt = matrix_multiplication(A, B_encrypt, size, size, size)
        end_time = time.time()

        # Decrypt result
        C_flat = decrypt_vector(C_encrypt)
        C = np.array(C_flat[:(size * size)])
        C = C.reshape(size, size, order='F')
        C = np.rint(C)
        print(f"Encrypted result...\n{C}\n")

        # Calculate plaintext result
        C_plain = np.matmul(A, B)

        # Check if the result is close to the plaintext
        if np.allclose(C, C_plain):
            print(f"Test passed for size {size}x{size}.\n")
        else:
            print(f"Test failed for size {size}x{size}.\n")

        # Print execution time
        print(f"Execution time for size {size}x{size}: {end_time - start_time:.3f} seconds\n")


if __name__ == '__main__':
    run_tests()

