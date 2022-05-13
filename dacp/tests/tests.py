import numpy as np
import matplotlib.pyplot as plt
from dacp.dacp import eigh
from scipy.sparse import csr_matrix
import unittest

# +
max_error_tresh = 1e-6
max_eigvec_tresh = 1e-4
N = 400
N_block = 20
deg_n = 20
loop_n = 5
window_size = 0.1

def random_values(shape):
    # defines random values from -1 to 1
    return (np.random.rand(*shape)-1/2)*2

def random_ham(N):
    H = random_values((N,N)) + random_values((N,N))*1j
    H = (H.conj().T + H)/(2*np.sqrt(N))
    return csr_matrix(H)

def random_ham_deg(N, deg):
    H = random_values((N,N)) + random_values((N,N))*1j
    H = (H.conj().T + H)/(2*np.sqrt(N))
    return csr_matrix(np.kron(H, np.identity(deg)))


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

def eigv_errors_test(loop_n, deg=False, **dacp_kwargs):
    relative_error_list = []
    missed_evals_list = []
    excess_evals_list = []
    for i in range(loop_n):
        if deg:
            H = random_ham_deg(N_block, deg_n)
        else:
            H = random_ham(N)
        relative_error, missed_evals, excess_evals = eigv_errors(H, window_size, **dacp_kwargs)
        relative_error_list.append(np.max(relative_error))
        missed_evals_list.append(len(missed_evals))
        excess_evals_list.append(excess_evals)
    
    return np.max(relative_error_list), np.sum(missed_evals_list), np.sum(excess_evals_list)
    
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
        relative_error, missed_evals, excess_evals, r = eigs_errors(H, window_size, **dacp_kwargs)
        relative_error_list.append(np.max(relative_error))
        missed_evals_list.append(len(missed_evals))
        excess_evals_list.append(excess_evals)
        r_list.append(np.max(r))
    return np.max(relative_error_list), np.sum(missed_evals_list), np.sum(excess_evals_list), np.max(r_list)


# +
class TestEigh(unittest.TestCase):
    def test_eigvals(self):
        """
        Test the eigenvalue only method
        """
        relative_error, missed_evals, excess_evals = eigv_errors_test(loop_n)
        self.assertEqual(missed_evals, 0, msg=f"The Algorithm failed to find {missed_evals} eigenvalues.")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found {excess_evals} excess faulty eigenvalues")
        max_error = np.max(relative_error)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")
        
    def test_eigvals_deg(self):
        """
        Test the eigenvalue only method
        """
        relative_error, missed_evals, excess_evals = eigv_errors_test(loop_n, deg=True, random_vectors=3)
        self.assertEqual(missed_evals, 0, msg=f"The Algorithm failed to find {missed_evals} eigenvalues.")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found {excess_evals} excess faulty eigenvalues")
        max_error = np.max(relative_error)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")

    def test_eigvecs(self):
        """
        Test the eigenvector method
        """
        relative_error, missed_evals, excess_evals, r = eigs_errors_test(loop_n)
        self.assertEqual(missed_evals, 0, msg=f"The Algorithm failed to find {missed_evals} eigenvalues.")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found {excess_evals} excess faulty eigenvalues")
        max_error = np.max(relative_error)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")
        max_vec_error = np.max(r)
        self.assertTrue(max_vec_error < max_eigvec_tresh, msg=f"Eigenvector relative errors too high: {max_vec_error}")
        
    def test_eigvecs_deg(self):
        """
        Test the eigenvector method
        """
        relative_error, missed_evals, excess_evals, r = eigs_errors_test(loop_n, deg=True, random_vectors=3)
        self.assertEqual(missed_evals, 0, msg=f"The Algorithm failed to find {missed_evals} eigenvalues.")
        self.assertEqual(excess_evals, 0, msg=f"The Algorithm found {excess_evals} excess faulty eigenvalues")
        max_error = np.max(relative_error)
        self.assertTrue(max_error < max_error_tresh, msg=f"Eigenvalue relative errors too high: {max_error}")
        max_vec_error = np.max(r)
        self.assertTrue(max_vec_error < max_eigvec_tresh, msg=f"Eigenvector relative errors too high: {max_vec_error}")
        
if __name__ == '__main__':
    unittest.main()
# -


