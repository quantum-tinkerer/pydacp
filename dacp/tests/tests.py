import numpy as np
import matplotlib.pyplot as plt
from dacp.dacp import eigh
from scipy.sparse import csr_matrix, kron
import unittest
from scipy.sparse import diags, eye

# +
N = 400
N_block = 100
deg_n = 4
loop_n = 30
window_size = 0.1


def random_ham(N):
    c = 2 * (np.random.rand(N - 1) + np.random.rand(N - 1) * 1j - 0.5 * (1 + 1j))
    b = 2 * (np.random.rand(N) - 0.5)

    H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
    return csr_matrix(H)


def random_ham_deg(N, deg):
    H = random_ham(N)
    return csr_matrix(kron(H, eye(deg)))


# +
def eigv_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp = eigh(H, window_size, **dacp_kwargs)
    else:
        eigv_dacp = eigh(H, window_size)

    N_dacp = len(eigv_dacp)

    map_eigv = []
    for value in eigv_dacp:
        closest = np.abs(eigv - value).min()
        map_eigv.append(eigv[np.abs(eigv - value) == closest][0])
    map_eigv = np.array(map_eigv)

    map_eigv = map_eigv[np.argsort(np.abs(map_eigv))]
    eigv_dacp = eigv_dacp[np.argsort(np.abs(eigv_dacp))]

    relative_error = np.abs((eigv_dacp - map_eigv) / map_eigv)[map_eigv < window_size]

    # Compute the theoretical error eta
    delta = np.finfo(float).eps
    # 0.1 comes from error window
    a_w = window_size * 1.1
    # 12 comes from filter order
    c_i_sq = np.exp(4 * 12 * np.sqrt(a_w**2 - (map_eigv[map_eigv < window_size])**2) / a_w)
    eta = delta * np.exp(4 * 12) / (np.abs(map_eigv[map_eigv < window_size]) * c_i_sq)

    diff = relative_error - eta
    return np.log10(np.heaviside(diff, 0) * diff / eta)


def eigv_errors_test(loop_n, deg=False, **dacp_kwargs):
    relative_error_list = []

    for i in range(loop_n):
        if deg:
            H = random_ham_deg(N_block, deg_n)
        else:
            H = random_ham(N)
        relative_error = eigv_errors(H, window_size, **dacp_kwargs)
        if relative_error.size != 0:
            relative_error_list.append(np.max(relative_error))

    return np.asarray(relative_error_list)


def eigs_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp, eigs_dacp = eigh(
            H, window_size, return_eigenvectors=True, **dacp_kwargs
        )
    else:
        eigv_dacp, eigs_dacp = eigh(H, window_size, return_eigenvectors=True)

    map_eigv = []
    for value in eigv_dacp:
        closest = np.abs(eigv - value).min()
        indx = np.where(np.abs(eigv - value) == closest)[0][0]
        map_eigv.append(eigv[indx])
        eigv = np.delete(eigv, indx)
    map_eigv = np.array(map_eigv)

    map_eigv = map_eigv[np.argsort(np.abs(map_eigv))]
    eigv_dacp = eigv_dacp[np.argsort(np.abs(eigv_dacp))]

    relative_error = np.abs((eigv_dacp - map_eigv) / map_eigv)[map_eigv < window_size]

    # Compute the theoretical error eta
    delta = np.finfo(float).eps
    # 0.1 comes from error window
    a_w = window_size * 1.1
    # 12 comes from filter order
    c_i_sq = np.exp(4 * 12 * np.sqrt(a_w**2 - (map_eigv[map_eigv < window_size])**2) / a_w)
    eta = delta * np.exp(4 * 12) / (np.abs(map_eigv[map_eigv < window_size]) * c_i_sq)

    diff = relative_error - eta
    return np.log10(np.heaviside(diff, 0) * diff / eta)


def eigs_errors_test(loop_n, deg=False, **dacp_kwargs):
    relative_error_list = []
    r_list = []

    for i in range(loop_n):
        if deg:
            H = random_ham_deg(N_block, deg_n)
        else:
            H = random_ham(N)
        relative_error = eigs_errors(H, window_size, **dacp_kwargs)
        if relative_error.size != 0:
            relative_error_list.append(np.max(relative_error))

    return np.asarray(relative_error_list)

# +
class TestEigh(unittest.TestCase):
    def test_eigvals(self):
        """
        Test the eigenvalue onlymethod
        """
        error_diff = eigv_errors_test(loop_n)
        self.assertTrue(
            error_diff.any() < 2,
            msg=f"Errors don't match the theoretical value.",
        )

    def test_eigvals_deg(self):
        """
        Test the eigenvalue only method
        """
        error_diff = eigv_errors_test(loop_n, deg=True, random_vectors=2)
        self.assertTrue(
            error_diff.any() < 2,
            msg=f"Errors don't match the theoretical value.",
        )

    def test_eigvecs(self):
        """
        Test the eigenvector method
        """
        error_diff = eigs_errors_test(loop_n)
        self.assertTrue(
            error_diff.any() < 2,
            msg=f"Errors don't match the theoretical value.",
        )



    def test_eigvecs_deg(self):
        """
        Test the eigenvector method
        """
        error_diff = eigs_errors_test(
            loop_n, deg=True, random_vectors=2
        )
        self.assertTrue(
            error_diff.any() < 2,
            msg=f"Errors don't match the theoretical value.",
        )


if __name__ == "__main__":
    unittest.main()
# -