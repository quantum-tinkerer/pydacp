from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from scipy.linalg import eigh
import math
from scipy.integrate import quad
import kwant
from . import chebyshev
import numpy as np


class DACP_reduction:

    def __init__(self, matrix, a, eps, bounds, sampling_subspace=1.5, max_iter=1e2, deg_tresh=1e-5):
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
        self.matrix = matrix
        self.a = a
        self.eps = eps
        if bounds:
            self.bounds = bounds
        else:
            self.find_bounds()
        self.sampling_subspace = sampling_subspace
        self.max_iter = int(max_iter)
        self.deg_tresh = deg_tresh

    def find_bounds(self, method='sparse_diagonalization'):
        # Relative tolerance to which to calculate eigenvalues.  Because after
        # rescaling we will add eps / 2 to the spectral bounds, we don't need
        # to know the bounds more accurately than eps / 2.
        tol = self.eps / 2

        lmax = float(eigsh(self.matrix, k=1, which='LA',
                           return_eigenvectors=False, tol=tol))
        lmin = float(eigsh(self.matrix, k=1, which='SA',
                           return_eigenvectors=False, tol=tol))

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                'The matrix has a single eigenvalue, it is not possible to '
                'obtain a spectral density.')

        self.bounds = [lmin, lmax]

    def G_operator(self):
        # TODO: generalize for intervals away from zero energy
        Emin = self.bounds[0] * (1 + self.eps)
        Emax = self.bounds[1] * (1 + self.eps)
        E0 = (Emax - Emin)/2
        Ec = (Emax + Emin)/2
        return (self.matrix - eye(self.matrix.shape[0]) * Ec) / E0

    def F_operator(self):
        # TODO: generalize for intervals away from zero energy
        Emax = np.max(np.abs(self.bounds)) * (1 + self.eps)
        E0 = (Emax**2 - self.a**2)/2
        Ec = (Emax**2 + self.a**2)/2
        return (self.matrix @ self.matrix - eye(self.matrix.shape[0]) * Ec) / E0

    def get_filtered_vector(self):
        # TODO: check whether we need complex vector
        v_rand = 2 * (np.random.rand(self.matrix.shape[0]) + np.random.rand(
            self.matrix.shape[0])*1j - 0.5 * (1 + 1j))
        v_rand = v_rand/np.linalg.norm(v_rand)
        K_max = int(12 * np.max(np.abs(self.bounds)) / self.a)
        vec = chebyshev.low_E_filter(v_rand, self.F_operator(), K_max)
        return vec / np.linalg.norm(vec)

    def estimate_subspace_dimenstion(self):
        dos_estimate = kwant.kpm.SpectralDensity(
            self.matrix,
            energy_resolution=self.a/4,
            mean=True,
            bounds=self.bounds
        )
        return int(np.abs(quad(dos_estimate, -self.a, self.a))[0])

    def span_basis(self, v_proj):
        d = self.estimate_subspace_dimenstion()
        n = math.ceil(np.abs((d*self.sampling_subspace - 1)/2))
        a_r = self.a / np.max(np.abs(self.bounds))
        n_array = np.arange(1, n+1, 1)
        indicesp1 = (n_array*np.pi/a_r).astype(int)
        indices = np.sort(np.array([*indicesp1, *indicesp1-1]))
        return chebyshev.basis(v_proj=v_proj, matrix=self.G_operator(), indices=indices)
        
    def get_orthogonal_basis(self, v_proj):
        v_basis = self.span_basis(v_proj)
        S = v_basis.conj() @ v_basis.T
        s, V = eigh(S)
        indx = np.abs(s) > 1e-12
        lambda_s = np.diag(1/np.sqrt(s[indx]))
        U = V[:, indx]@lambda_s
        return U.T.conj()@v_basis

    def get_subspace_matrix(self):
        v = self.get_filtered_vector()
        U = self.get_orthogonal_basis(v)

        for i in range(self.max_iter):
            v = self.get_filtered_vector()
            v = v - U.T@U.conj()@v

            if np.abs(np.sum(v)) < self.deg_tresh:
                break

            U_inc = self.get_orthogonal_basis(v)
            U = np.concatenate((U, U_inc), axis=0)

        H_red = U.conj()@self.matrix@U.T
        return H_red
