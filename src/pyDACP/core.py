from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from scipy.linalg import eigh, qr, qr_insert
from scipy.integrate import quad
import kwant
from . import chebyshev
import numpy as np
from math import floor, ceil
import itertools as it


class DACP_reduction:
    def __init__(
        self,
        matrix,
        a,
        eps,
        bounds=None,
        sampling_subspace=1.5,
        random_vectors=2,
        return_eigenvectors=False
    ):
        """Find the spectral bounds of a given matrix.

        Parameters
        ----------
        matrix : 2D array
            Initial matrix.
        eps : scalar
            Ensures that the bounds are strict.
        bounds : tuple, or None
            Boundaries of the spectrum. If not provided the maximum and
            minimum eigenvalues are calculated.
        """
        self.matrix = matrix.tocsr()
        self.a = a
        self.eps = eps
        self.return_eigenvectors = return_eigenvectors
        if bounds:
            self.bounds = bounds
        else:
            self.find_bounds()
        self.sampling_subspace = sampling_subspace
        self.random_vectors = random_vectors

    def find_bounds(self, method="sparse_diagonalization"):
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
        # TODO: check whether we need complex vector
        v_rand = 2 * (
            np.random.rand(self.matrix.shape[0], self.random_vectors)
            + np.random.rand(self.matrix.shape[0], self.random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand)
        K_max = int(filter_order * np.max(np.abs(self.bounds)) / self.a)
        vec = chebyshev.low_E_filter(v_rand, self.F_operator(), K_max)
        return vec / np.linalg.norm(vec, axis=0)

    def estimate_subspace_dimenstion(self):
        dos_estimate = kwant.kpm.SpectralDensity(
            self.matrix, energy_resolution=self.a / 4, mean=True, bounds=self.bounds
        )
        return int(np.abs(quad(dos_estimate, -self.a, self.a))[0])

    def svd_matrix(self, matrix_proj, S):
        s, V = eigh(S)
        indx = np.abs(s) > 1e-12
        lambda_s = np.diag(1 / np.sqrt(s[indx]))
        U = V[:, indx] @ lambda_s
        return U.T.conj() @ matrix_proj @ U, U.T.conj() @ S @ U

    def direct_eigenvalues(self):
        d = self.estimate_subspace_dimenstion()
        n = int(np.abs((d*self.sampling_subspace - 1)/2))
        a_r = self.a / np.max(np.abs(self.bounds))
        dk = np.pi / a_r

        S, matrix_proj = chebyshev.basis_no_store(
            v_proj=self.get_filtered_vector(),
            matrix=self.G_operator(),
            H=self.matrix,
            dk=int(dk),
            random_vectors=self.random_vectors
        )

        return self.svd_matrix(matrix_proj, S)

    def span_basis(self):
        a_r = self.a / np.max(np.abs(self.bounds))
        dk = np.pi / a_r / self.random_vectors
        # First run
        Q, R = chebyshev.basis(
            v_proj=self.get_filtered_vector(),
            matrix=self.G_operator(),
            dk = dk
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

    def eigenvalues_and_eigenvectors_filter(self):
        self.v_basis = self.span_basis_filter()
        S = self.v_basis.conj().T @ self.v_basis
        matrix_proj = self.v_basis.conj().T @ self.matrix.dot(self.v_basis)
        return matrix_proj, S

    def eigenvalues_and_eigenvectors(self):
        self.span_basis()
        S = self.v_basis.conj().T @ self.v_basis
        matrix_proj = self.v_basis.conj().T @ self.matrix.dot(self.v_basis)
        return self.svd_matrix(matrix_proj, S)

    def get_subspace_matrix(self):
        if self.return_eigenvectors:
            return self.eigenvalues_and_eigenvectors()
        else:
            return self.direct_eigenvalues()
