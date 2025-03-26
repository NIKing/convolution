
import numpy as np

def test_matrix_split():
    a = np.random.rand(6, 6)
    print(a)
    print(a[:3, (0 * 2):(3 + 0)])
    print(a[:3, (1 * 2):(3 + 1 * 2)])


test_matrix_split()
