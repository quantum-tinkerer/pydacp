import numpy as np
import matplotlib.pyplot as plt
from dacp.dacp import eigh
from scipy.sparse import csr_matrix, kron
import unittest
from scipy.sparse import diags, eye

# +
max_error_tresh = 1e-4
max_eigvec_tresh = 1e-3
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

    relative_error = np.abs((eigv_dacp - map_eigv) / map_eigv)

    # Compute the theoretical error eta
    delta = np.finfo(float).eps
    # 0.1 comes from error window
    a_w = window_size * 1.1
    # 12 comes from filter order
    c_i_sq = np.exp(4 * 12 * np.sqrt(a_w**2 - map_eigv**2) / a_w)
    eta = delta * np.exp(4 * 12) / (np.abs(map_eigv) * c_i_sq)

    return np.log10(np.abs(relative_error - eta))


def eigv_errors_test(loop_n, deg=False, **dacp_kwargs):
    relative_error_list = []

    for i in range(loop_n):
        if deg:
            H = random_ham_deg(N_block, deg_n)
        else:
            H = random_ham(N)
        relative_error = eigv_errors(H, window_size, **dacp_kwargs)
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

    N_dacp = len(eigv_dacp)

    window_args = np.abs(eigv) < window_size
    eigv = eigv[window_args]
    eigs = eigs[window_args, :]

    order_args = np.argsort(np.abs(eigv))
    eigv = eigv[order_args]
    eigs = eigs[order_args, :]

    order_args = np.argsort(np.abs(eigv_dacp))
    eigv_dacp = eigv_dacp[order_args]
    eigs_dacp = eigs_dacp[order_args, :]

    N_scipy = len(eigv)
    N_min = np.min([N_dacp, N_scipy])

    relative_error = np.abs(np.abs(eigv_dacp[:N_min]) - np.abs(eigv[:N_min])) / np.abs(
        eigv[:N_min]
    )
    missed_evals = eigv[N_dacp:]
    excess_evals = len(eigv_dacp[N_min:])

    r = np.einsum("ij, kj -> ki", H.todense(), eigs_dacp[:N_min]) - np.einsum(
        "i, ij -> ij", eigv_dacp[:N_min], eigs_dacp[:N_min]
    )
    r = np.linalg.norm(r, axis=1)
    return relative_error, missed_evals, excess_evals, r


def eigs_errors_test(loop_n, deg=False, **dacp_kwargs):
    relative_error_list = []
    missed_evals_list = []
    excess_evals_list = []
    r_list = []
    for i in range(loop_n):
        if deg:
            H = random_ham_deg(N_block, deg_n)
        else:
            H = random_ham(N)
        relative_error, missed_evals, excess_evals, r = eigs_errors(
            H, window_size, **dacp_kwargs
        )
        relative_error_list.append(np.max(relative_error))
        missed_evals_list.append(len(missed_evals))
        excess_evals_list.append(excess_evals)
        r_list.append(np.max(r))
    return (
        np.max(relative_error_list),
        missed_evals_list,
        excess_evals_list,
        np.max(r_list),
    )


# +
class TestEigh(unittest.TestCase):
    def test_eigvals(self):
        """
        Test the eigenvalue only method
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

    # def test_eigvecs(self):
    #     """
    #     Test the eigenvector method
    #     """
    #     relative_error, missed_evals, excess_evals, r = eigs_errors_test(loop_n)
    #     self.assertEqual(
    #         np.sum(missed_evals),
    #         0,
    #         msg=f"The Algorithm failed to find {np.sum(missed_evals)} evals in {loop_n} loops. Single run max missing evals is {np.max(missed_evals)} ",
    #     )
    #     self.assertEqual(
    #         np.sum(excess_evals),
    #         0,
    #         msg=f"The Algorithm found faulty excess {np.sum(excess_evals)} evals in {loop_n} loops. Single run max excess evals is {np.max(excess_evals)} ",
    #     )
    #     max_error = np.max(relative_error)
    #     self.assertTrue(
    #         max_error < max_error_tresh,
    #         msg=f"Eigenvalue relative errors too high: {max_error}",
    #     )
    #     max_vec_error = np.max(r)
    #     self.assertTrue(
    #         max_vec_error < max_eigvec_tresh,
    #         msg=f"Eigenvector relative errors too high: {max_vec_error}",
    #     )

    # def test_eigvecs_deg(self):
    #     """
    #     Test the eigenvector method
    #     """
    #     relative_error, missed_evals, excess_evals, r = eigs_errors_test(
    #         loop_n, deg=True, random_vectors=2
    #     )
    #     self.assertEqual(
    #         np.sum(missed_evals),
    #         0,
    #         msg=f"The Algorithm failed to find {np.sum(missed_evals)} evals in {loop_n} loops. Single run max missing evals is {np.max(missed_evals)} ",
    #     )
    #     self.assertEqual(
    #         np.sum(excess_evals),
    #         0,
    #         msg=f"The Algorithm found faulty excess {np.sum(excess_evals)} evals in {loop_n} loops. Single run max excess evals is {np.max(excess_evals)} ",
    #     )
    #     max_error = np.max(relative_error)
    #     self.assertTrue(
    #         max_error < max_error_tresh,
    #         msg=f"Eigenvalue relative errors too high: {max_error}",
    #     )
    #     max_vec_error = np.max(r)
    #     self.assertTrue(
    #         max_vec_error < max_eigvec_tresh,
    #         msg=f"Eigenvector relative errors too high: {max_vec_error}",
    #     )


if __name__ == "__main__":
    unittest.main()
# -