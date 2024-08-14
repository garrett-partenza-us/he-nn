from seal import *
from seal_helpers import *
import numpy as np
import math
from linear_ops import *
import copy

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 16384
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 40, 40, 60]))
scale = 2.0**40
context = SEALContext(parms)
print_parameters(context)

ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()
print(f'Number of slots: {slot_count}')

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_keys = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

def get_general_diagonals(matrix):

    tiles = np.tile(matrix, 2)
    return list(np.diag(tiles, idx) for idx in range(matrix.shape[1]))


def encrypt_vector(vector):
 
    return encryptor.encrypt(ckks_encoder.encode(vector, scale))

def decrypt_vector(vector):
    
    return ckks_encoder.decode(decryptor.decrypt(vector))

def rotate_vector(vector, steps):
    
    return evaluator.rotate_vector(vector, -steps, galois_keys)


def matrix_vector_multiplication(matrix, vector):

    diagonals = get_general_diagonals(matrix)

    encrypted_diagonals = [encrypt_vector(diagonal) for diagonal in diagonals]

    parms = vector.parms_id() if vector.parms_id() != encrypted_diagonals[0].parms_id() else None

    if parms:
        encrypted_diagonals = [
            evaluator.mod_switch_to(diagonal, parms) for diagonal in encrypted_diagonals
        ]

    init = evaluator.multiply(encrypted_diagonals[0], vector)

    init = evaluator.relinearize(init, relin_keys)

    addends = [init]

    for idx, diagonal in enumerate(encrypted_diagonals[1:]):

        addend = evaluator.multiply(
            diagonal,
            rotate_vector(vector, - (idx + 1))
        )

        addend = evaluator.relinearize(addend, relin_keys)

        addends.append(addend)

    sum = evaluator.add_many(addends)

    return sum


# NOTE: b_encrypt must be column flattened and tiled twice
def matrix_multiplication(a_plain, b_encrypt, m, l, n):

    addends = []

    A = np.matmul(
        sigma_transformation_matrix(m, l).astype(int),
        a_plain.copy().flatten(order='F')
    )

    B = matrix_vector_multiplication(
        theta_transformation_matrix(l, n).astype(int),
        b_encrypt
    )

    B = evaluator.relinearize(B, relin_keys)

    # TODO: Figure out how to do matrix vector multiplication without tiling
    # This decryption makes the output correct, because the B matrix has to 
    # be tiled for the rhs operation in the forloop.
    B = decrypt_vector(B)
    B = np.array(B[:16])
    B = np.rint(B).astype(int)
    B = encrypt_vector(np.tile(B,2))

    for k in range(l):
        
        lhs = np.matmul(
            epsilon_transformation_matrix(m, n, l, k).astype(int),
            A.copy()
        )

        rhs = matrix_vector_multiplication(
            omega_transformation_matrix(m, n, l, k).astype(int),
            B
        )

        lhs = encrypt_vector(lhs)
        lhs = evaluator.mod_switch_to(lhs, rhs.parms_id())

        product = evaluator.multiply(lhs, rhs)
        product = evaluator.relinearize(product, relin_keys)
        addends.append(product)

    C = evaluator.add_many(addends)

    return C




if __name__ == '__main__':

    m, l, n = 4, 4, 4

    A = np.arange(1, 17).astype(int).reshape(m,l)
    B = np.arange(1, 17).astype(int).reshape(l,n)

    print("Matrix A...\n{}\n".format(A))
    print("Matrix B...\n{}\n".format(B))
    print("Plain...\n{}\n".format(np.matmul(A,B)))

    B = B.flatten(order='F')
    B = np.tile(B, 2)
    B = encrypt_vector(B)

    C = matrix_multiplication(A, B, m, l, n)

    C = decrypt_vector(C)
    C = np.array(C[:(m*n)])
    C = C.reshape(m, n, order='F')
    C = np.rint(C)
    print("Cryptic...\n{}\n".format(C))

    # Example of matrix-vector multiplication
    """
    A = np.random.randint(1,20,(8,8))
    B = np.array([1,2,3,4,5,6,7,8])

    print("Plain: ", np.matmul(A, B))

    B_encrypt = encrypt_vector(np.tile(B, 2))

    C_encrypt = matrix_vector_multiplication(A, B_encrypt)
    C_plain = decryptor.decrypt(C_encrypt)
    C = ckks_encoder.decode(C_plain)

    print("Cryptic: ", np.array(C[:8]).astype(int))
    """
    

    """
    # Example of matrix-vector multiplication where B is encrypted

    A = np.arange(1,17).reshape(4,4)
    T = sigma_transformation_matrix(4, 4).astype(int)
    print("Transformation Matrix: ", T)
    A = A.flatten(order='F')
    A = np.tile(A, 2)
    A = encrypt_vector(A)
    R = matrix_vector_multiplication(T, A)
    R = decrypt_vector(R)
    R = np.array(R[:16]).reshape(4,4, order='F')
    print("Encrypted Result: ", np.rint(R))

    """


