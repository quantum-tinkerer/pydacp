from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, csr_matrix
from scipy.linalg import eigh, eigvalsh
from . import chebyshev
import numpy as np
from math import ceil


def dacp_eig(
    matrix,
    a,
    eps=0.1,
    bounds=None,
    random_vectors=2,
    return_eigenvectors=False,
    filter_order=20
):
    """
    Find the eigendecomposition within the given spectral bounds of a given matrix.

    Parameters
    ----------
    matrix : 2D array or sparse matrix
        Initial matrix.
    eps : float
        Ensures that the bounds are strict.
    bounds : tuple, or None
        Boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.
    random_vectors : int
        When return_eigenvectors=False, specifies the maximum expected
        degeneracy of the matrix.
    return_eigenvectors : bool
        If True, returns eigenvectors and processes general degeneracies.
        However, if False, the algorithm conserves memory
        and only processes random_vectors>degenerecies.
    filter_order : int
        The number of times a vector is filtered is given by filter_order*E_max/a.
    """
    matrix = csr_matrix(matrix)

    if bounds is None:
        # Relative tolerance to which to calculate eigenvalues.  Because after
        # rescaling we will add eps / 2 to the spectral bounds, we don't need
        # to know the bounds more accurately than eps / 2.
        tol = eps / 2

        lmax = float(
            eigsh(matrix, k=1, which="LA", return_eigenvectors=False, tol=tol)
        )
        lmin = float(
            eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=tol)
        )

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                "The matrix has a single eigenvalue, it is not possible to "
                "obtain a spectral density."
            )

        bounds = [lmin, lmax]

    Emin = bounds[0] * (1 + eps)
    Emax = bounds[1] * (1 + eps)
    E0 = (Emax - Emin) / 2
    Ec = (Emax + Emin) / 2
    G_operator = (matrix - eye(matrix.shape[0]) * Ec) / E0

    Emax = np.max(np.abs(bounds)) * (1 + eps)
    E0 = (Emax ** 2 - a ** 2) / 2
    Ec = (Emax ** 2 + a ** 2) / 2
    F_operator = (matrix @ matrix - eye(matrix.shape[0]) * Ec) / E0

    def get_filtered_vector():
        v_rand = 2 * (
            np.random.rand(matrix.shape[0], random_vectors)
            + np.random.rand(matrix.shape[0], random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand, axis=0)
        K_max = int(filter_order * np.max(np.abs(bounds)) / a)
        vec = chebyshev.low_E_filter(v_rand, F_operator, K_max)
        return vec / np.linalg.norm(vec, axis=0)

    a_r = a / np.max(np.abs(bounds))
    dk = np.pi / a_r
    if return_eigenvectors:
        # First run
        Q, R = chebyshev.basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk
        )
        # Second run
        Qi, Ri = chebyshev.basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk,
            Q=Q,
            R=R,
            first_run=False,
        )
        # Other runs to solve higher degeneracies
        while Q.shape[1] < Qi.shape[1]:
            Q, R = Qi, Ri
            Qi, Ri = chebyshev.basis(
                v_proj=get_filtered_vector(),
                G_operator=G_operator,
                dk=dk,
                Q=Q,
                R=R,
                first_run=False,
            )
        v_basis = Q
        matrix_proj = v_basis.conj().T @ matrix.dot(v_basis)
        eigvals, eigvecs = eigh(matrix_proj)
        return eigvals, eigvecs @ v_basis.T

    else:
        # First run
        S, matrix_proj = chebyshev.basis_no_store(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            matrix=matrix,
            dk=ceil(dk),
            random_vectors=random_vectors
        )
        
        # Second run
        Qi, Ri = chebyshev.basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk,
            Q=Q,
            R=R,
            first_run=False,
        )

        while

        # s, V = eigh(S)
        # indx = np.abs(s) > 1e-12
        # lambda_s = np.diag(1/np.sqrt(s[indx]))
        # U = V[:, indx]@lambda_s

        return eigvalsh(matrix_proj, S)#U.T.conj() @ matrix_proj @ U)


# TODO: Delete the class in the future
class DACP_reduction:
    def __init__(
        self,
        matrix,
        a,
        eps=0.1,
        bounds=None,
        random_vectors=2,
        return_eigenvectors=False
    ):
        """Find the spectral bounds of a given matrix.

        Parameters
        ----------
        matrix : 2D array
            Initial matrix.
        eps : float
            Ensures that the bounds are strict.
        bounds : tuple, or None
            Boundaries of the spectrum. If not provided the maximum and
            minimum eigenvalues are calculated.
        random_vectors : int
            When return_eigenvectors=False, specifies the maximum expected
            degeneracy of the matrix.
        return_eigenvectors : bool
            If True, returns eigenvectors. However, if False, the algorithm
            conserves memory.
        """
        self.matrix = csr_matrix(matrix)
        self.a = a
        self.eps = eps
        self.return_eigenvectors = return_eigenvectors
        if bounds:
            self.bounds = bounds
        else:
            self.find_bounds()
        self.random_vectors = random_vectors

    def find_bounds(self):
        # Relative tolerance to which to calculate eigenvalues.  Because after
        # rescaling we will add eps / 2 to the spectral bounds, we don't need
        # to know the bounds more accurately than eps / 2.
        tol = self.eps / 2

        lmax = float(
            eigsh(self.matrix, k=1, which="LA", return_eigenvectors=False, tol=tol)
        )
        lmin = float(
            eigsh(self.matrix, k=1, which="SA", return_eigenvectors=False, tol=tol)
        )

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                "The matrix has a single eigenvalue, it is not possible to "
                "obtain a spectral density."
            )

        self.bounds = [lmin, lmax]

    def G_operator(self):
        # TODO: generalize for intervals away from zero energy
        Emin = self.bounds[0] * (1 + self.eps)
        Emax = self.bounds[1] * (1 + self.eps)
        E0 = (Emax - Emin) / 2
        Ec = (Emax + Emin) / 2
        return (self.matrix - eye(self.matrix.shape[0]) * Ec) / E0

    def F_operator(self):
        # TODO: generalize for intervals away from zero energy
        Emax = np.max(np.abs(self.bounds)) * (1 + self.eps)
        E0 = (Emax ** 2 - self.a ** 2) / 2
        Ec = (Emax ** 2 + self.a ** 2) / 2
        return (self.matrix @ self.matrix - eye(self.matrix.shape[0]) * Ec) / E0

    def get_filtered_vector(self, filter_order=20):
        v_rand = 2 * (
            np.random.rand(self.matrix.shape[0], self.random_vectors)
            + np.random.rand(self.matrix.shape[0], self.random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand)
        K_max = int(filter_order * np.max(np.abs(self.bounds)) / self.a)
        vec = chebyshev.low_E_filter(v_rand, self.F_operator(), K_max)
        return vec / np.linalg.norm(vec, axis=0)

    def direct_eigenvalues(self):
        a_r = self.a / np.max(np.abs(self.bounds))
        dk = int(np.pi / a_r)

        S, matrix_proj = chebyshev.basis_no_store(
            v_proj=self.get_filtered_vector(),
            matrix=self.G_operator(),
            H=self.matrix,
            dk=dk,
            random_vectors=self.random_vectors
        )

        s, V = eigh(S)
        indx = np.abs(s) > 1e-12
        lambda_s = np.diag(1/np.sqrt(s[indx]))
        U = V[:, indx]@lambda_s

        return eigvalsh(U.T.conj() @ matrix_proj @ U)

    def span_basis(self):
        a_r = self.a / np.max(np.abs(self.bounds))
        dk = np.pi / a_r
        # First run
        Q, R = chebyshev.basis(
            v_proj=self.get_filtered_vector(),
            matrix=self.G_operator(),
            dk=dk
        )
        # Second run
        Qi, Ri = chebyshev.basis(
            v_proj=self.get_filtered_vector(),
            matrix=self.G_operator(),
            dk=dk,
            Q=Q,
            R=R,
            first_run=False,
        )
        # Other runs to solve higher degeneracies
        while Q.shape[1] < Qi.shape[1]:
            Q, R = Qi, Ri
            Qi, Ri = chebyshev.basis(
                v_proj=self.get_filtered_vector(),
                matrix=self.G_operator(),
                dk=dk,
                Q=Q,
                R=R,
                first_run=False,
            )
        self.v_basis = Q

    def eigenvalues_and_eigenvectors(self):
        self.span_basis()
        S = self.v_basis.conj().T @ self.v_basis
        matrix_proj = self.v_basis.conj().T @ self.matrix.dot(self.v_basis)
        eigvals, eigvecs = eigh(matrix_proj)
        return eigvals, eigvecs @ self.v_basis.T

    def get_subspace_matrix(self):
        if self.return_eigenvectors:
            return self.eigenvalues_and_eigenvectors()
        else:
            return self.direct_eigenvalues()
