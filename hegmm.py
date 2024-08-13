from seal import *
from seal_helpers import *
import numpy as np
import math

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

    init = evaluator.multiply(encrypted_diagonals[0], vector)

    init = evaluator.relinearize(init, relin_keys)

    init = evaluator.rescale_to_next(init)

    addends = [init]

    for idx, diagonal in enumerate(encrypted_diagonals[1:]):

        addend = evaluator.multiply(
            diagonal,
            rotate_vector(vector, - (idx + 1))
        )

        addend = evaluator.relinearize(addend, relin_keys)

        addend = evaluator.rescale_to_next(addend)

        addends.append(addend)

    sum = evaluator.add_many(addends)

    return sum


if __name__ == '__main__':

    A = np.random.randint(1,20,(8,8))
    B = [1,2,3,4,5,6,7,8]

    print(np.matmul(A, B))

    B_encrypt = encrypt_vector(np.tile(B, 2))

    C_encrypt = matrix_vector_multiplication(A, B_encrypt)
    C_plain = decryptor.decrypt(C_encrypt)
    C = ckks_encoder.decode(C_plain)

    print(np.array(C[:8]).astype(int))
