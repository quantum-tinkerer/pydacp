from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from scipy.linalg import eigh
from scipy.integrate import quad
import kwant
from . import chebyshev
import numpy as np


class DACP_reduction:

    def __init__(self, matrix, a, eps, bounds=None, sampling_subspace=2, random_vectors=1):
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
        self.random_vectors=random_vectors

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

    def direct_eigenvalues(self):
        d = self.estimate_subspace_dimenstion()
        n = int(np.abs((d*self.sampling_subspace - 1)/2))
        a_r = self.a / np.max(np.abs(self.bounds))
        n_array = np.arange(1, n+1, 1)
        indices = (n_array*np.pi/a_r).astype(int)
        indices_to_store = np.sort(np.array([*indices+1, *indices, *indices-1, *indices-2, 0, 1, 2]))

        S = np.zeros((n, n), dtype=np.complex128)
        H = np.zeros((n, n), dtype=np.complex128)
        v_proj = self.get_filtered_vector()
        S_xy, H_xy = chebyshev.basis_no_store(v_proj=v_proj, matrix=self.G_operator(), H = self.matrix, indices_to_store=indices_to_store)

        # def ijselector(i, j):
        #     dk = np.pi / a_r
        #     if i % 2 == 0 and j % 2 == 0:
        #         k_p = int(dk * i / 2) + int(dk * j / 2) - 2
        #         k_m = abs(int(dk * i / 2) - int(dk * j / 2))
        #     elif i % 2 == 1 and j % 2 == 0:
        #         k_p = int(dk * (i - 1) / 2) + int(dk * j / 2) - 1
        #         k_m = abs(int(dk * (i - 1) / 2) - int(dk * j / 2) + 1)
        #     elif i % 2 == 0 and j % 2 == 1:
        #         k_p = int(dk * i / 2) - 1 + int(dk * (j - 1) / 2)
        #         k_m = abs(int(dk * i / 2) - 1 - int(dk * (j - 1) / 2))
        #     else:
        #         k_p = int(dk * (i - 1) / 2) + int(dk * (j - 1) / 2)
        #         k_m = abs(int(dk * (i - 1) / 2) - int(dk * (j - 1) / 2))
        #     return k_p, k_m

        def ijselector(i, j):
            dk = np.pi / a_r
            if i % 2 == 0 and j % 2 == 0:
                k_p = int(dk * i / 2 + dk * j / 2 - 2)
                k_m = abs(int(dk * i / 2 - dk * j / 2))
            elif i % 2 == 1 and j % 2 == 0:
                k_p = int(dk * (i - 1) / 2 + dk * j / 2 - 1)
                k_m = abs(int(dk * (i - 1) / 2 - dk * j / 2 + 1))
            elif i % 2 == 0 and j % 2 == 1:
                k_p = int(dk * i / 2 - 1 + dk * (j - 1) / 2)
                k_m = abs(int(dk * i / 2 - 1 - dk * (j - 1) / 2))
            else:
                k_p = int(dk * (i - 1) / 2 + dk * (j - 1) / 2)
                k_m = abs(int(dk * (i - 1) / 2 - dk * (j - 1) / 2))
            return k_p, k_m

        for i in range(n):
            for j in range(n):
                x, y = ijselector(i+1, j+1)
                ind_p = np.where(indices_to_store == x)[0][0]
                ind_m = np.where(indices_to_store == y)[0][0]
                S[i, j] = 0.5 * (S_xy[ind_p] + S_xy[ind_m])
                H[i, j] = 0.5 * (H_xy[ind_p] + H_xy[ind_m])

        return S, H

    def span_basis(self):
        d = self.estimate_subspace_dimenstion()
        n = int(np.abs((d*self.sampling_subspace - 1)/2))
        # Divide by the number of random vectors
        n = int(n/self.random_vectors)
        a_r = self.a / np.max(np.abs(self.bounds))
        n_array = np.arange(1, n+1, 1)
        indicesp1 = (n_array*np.pi/a_r).astype(int)
        indices = np.sort(np.array([*indicesp1, *indicesp1-1]))
        basis=[]
        for i in range(self.random_vectors):
            v_proj = self.get_filtered_vector()
            basis.append(chebyshev.basis(
                v_proj=v_proj, matrix=self.G_operator(), indices=indices))
        self.v_basis=np.concatenate(np.asarray(basis))

    def get_subspace_matrix(self):
        self.span_basis()
        S = self.v_basis.conj() @ self.v_basis.T
        matrix_proj = self.v_basis.conj() @ self.matrix.dot(self.v_basis.T)
        s, V = eigh(S)
        indx = np.abs(s) > 1e-12
        lambda_s = np.diag(1/np.sqrt(s[indx]))
        U = V[:, indx]@lambda_s
        self.subspace_matrix = U.T.conj() @ matrix_proj @ U
        return self.subspace_matrix