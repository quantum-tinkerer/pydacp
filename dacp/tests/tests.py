import numpy as np
import matplotlib.pyplot as plt
from dacp.dacp import eigh
from scipy.sparse import csr_matrix
import unittest

# +
max_error_tresh = 1e-6
N = 400
window_size = 0.1

def random_values(shape):
    # defines random values from -1 to 1
    return (np.random.rand(*shape)-1/2)*2

H = random_values((N,N)) + random_values((N,N))*1j
H = (H.conj().T + H)/(2*np.sqrt(N))

H = csr_matrix(H)


# +
def eigv_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp = eigh(H, window_size, **dacp_kwargs)
    else:
        eigv_dacp = eigh(H, window_size)

    N_dacp = len(eigv_dacp)

    eigv = eigv[np.abs(eigv) < window_size]
    eigv = eigv[np.argsort(np.abs(eigv))]
    eigv_dacp = eigv_dacp[np.argsort(np.abs(eigv_dacp))]

    N_scipy = len(eigv)
    N_min = np.min([N_dacp, N_scipy])

    relative_error = np.abs(np.abs(eigv_dacp[:N_min])-np.abs(eigv[:N_min]))/np.abs(eigv[:N_min])
    missed_evals = eigv[N_dacp:]
    excess_evals = len(eigv_dacp[N_min:])

    return relative_error, missed_evals, excess_evals


def eigs_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp, eigs_dacp = eigh(H, window_size, return_eigenvectors=True, **dacp_kwargs)
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

    relative_error = np.abs(np.abs(eigv_dacp[:N_min])-np.abs(eigv[:N_min]))/np.abs(eigv[:N_min])
    missed_evals = eigv[N_dacp:]
    excess_evals = len(eigv_dacp[N_min:])

    r = np.einsum('ij, kj -> ki', H.todense(), eigs_dacp[:N_min]) - np.einsum('i, ij -> ij', eigv_dacp[:N_min], eigs_dacp[:N_min])
    r = np.linalg.norm(r, axis=1)
    return relative_error, missed_evals, excess_evals, r


# +
class TestEigh(unittest.TestCase):
    def test_eigvals(self):
        """
        Test the eigenvalue only method
        """
        errors_val, missed_vals, excess_evals = eigv_errors(H, window_size)
        self.assertEqual(len(missed_vals), 0, msg=f"The Algorithm failed to find the following eigenvalues: {missed_vals}")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found excess faulty {excess_evals} eigenvalues")
        max_error = np.max(errors_val)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")

    def test_eigvecs(self):
        """
        Test the eigenvector method
        """
        errors_val, missed_vals, excess_evals, r = eigs_errors(H, window_size)
        self.assertEqual(len(missed_vals), 0, msg=f"The Algorithm failed to find the following eigenvalues: {missed_vals}")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found excess faulty {excess_evals} eigenvalues")
        max_error = np.max(errors_val)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")
        max_vec_error = np.max(r)
        self.assertTrue(max_vec_error < max_error_tresh, msg=f"Eigenvector relative errors too high: {max_vec_error}")


if __name__ == '__main__':
    unittest.main()
