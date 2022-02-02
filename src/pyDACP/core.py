from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, csr_matrix
from scipy.linalg import eigh, eigvalsh, qr, qr_insert
from . import chebyshev
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


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

    def get_filtered_vector(qr_decomp = False):
        v_rand = 2 * (
            np.random.rand(matrix.shape[0], random_vectors)
            + np.random.rand(matrix.shape[0], random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand, axis=0)
        K_max = int(filter_order * np.max(np.abs(bounds)) / a)
        vec = chebyshev.low_E_filter(v_rand, F_operator, K_max)
        if random_vectors > 1 and qr_decomp:
            return qr(vec / np.linalg.norm(vec, axis=0), mode='economic')
        else:
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
        # Generate a QR orthogonalized collection of random vectors.
        Q, R = get_filtered_vector(qr_decomp=True)
        # Make sure they are indeed orthogonal by removing the non_orthogonal ones.
        # MUST BE CAREFUL HERE BECAUSE THE COLLECTION OF VECTORS IS NO LONGER SIZE (M, random_vectors)
        # So we update everywhere `random_vectors = filtered_qr_vectors.shape[1]`.
        ortho_condition = np.abs(np.diag(R)) < 1e-9
        if ortho_condition.any():
            indices = np.invert(ortho_condition)
            Q, R = Q[:, indices], R[indices, :][:, indices]
        # Now run
        S, matrix_proj, dim, indices_prev = chebyshev.basis_no_store(
            v_proj=Q,
            G_operator=G_operator,
            matrix=matrix,
            dk=ceil(dk),
            random_vectors=Q.shape[1],
        )
        # Second run
        # Same thing, but we don't perform QR in the first step.
        v_i = get_filtered_vector()
        # Instead, we QR wrt all the vectors collected so far.
        Q_i, R_i = qr_insert(Q, R, u=v_i, k=Q.shape[1], which="col")
        # Remove non-orthogonal vectors
        ortho_condition = np.abs(np.diag(R_i)) < 1e-9
        if ortho_condition.any():
            indices = np.invert(ortho_condition)
            Q_i, R_i = Q_i[:, indices], R_i[indices, :][:, indices]
        # Q_i contain all vectors, we just want the new ones.
        v_i = Q_i[:,Q.shape[1]:]

        # This is just to adjust the dimenstion:
        v_prev = np.stack([Q])
        Si, matrix_proj_i, dim_i, indices_prev = chebyshev.basis_no_store(
            v_proj=v_i,
            v_prev=v_prev,
            G_operator=G_operator,
            matrix=matrix,
            dk=ceil(dk),
            random_vectors=v_i.shape[1],
            S_prev = S,
            matrix_prev = matrix_proj,
            first_run=False,
            indices_prev = indices_prev
        )
        # Again adjust dimenstion
        v_prev_i = np.stack([v_i])

        # We have two stop conditions here:
        # 1. Dimension of the subspace no longer increases.
        # 2. There are no more orthogonal random vectors.
        while (dim_i > dim) and (v_i.shape[1] > 0):
            # If both conditions are fulfilled, we go to next interation.
            S, matrix_proj, R, Q, dim = Si, matrix_proj_i, R_i, Q_i, dim_i
            # Stack vectors together and make sure dimensions match.
            v_prev = np.vstack([v_prev, v_prev_i])
            # Generate a new set of random vectors.
            v_i = get_filtered_vector()
            # Remove linearly dependent ones.
            Q_i, R_i = qr_insert(Q, R, u=v_i, k=Q.shape[1], which="col")
            ortho_condition = np.abs(np.diag(R_i)) < 1e-9
            if ortho_condition.any():
                indices = np.invert(ortho_condition)
                Q_i, R_i = Q_i[:, indices], R_i[indices, :][:, indices]
            v_i = Q_i[:,Q.shape[1]:]
            # If there's any vector to input, we run it one more time.
            if v_i.shape[1] > 0:
                Si, matrix_proj_i, dim_i, indices_prev = chebyshev.basis_no_store(
                    v_proj=v_i,
                    v_prev=v_prev,
                    G_operator=G_operator,
                    matrix=matrix,
                    dk=int(dk),
                    random_vectors=v_i.shape[1],
                    S_prev = S,
                    matrix_prev = matrix_proj,
                    first_run=False,
                    indices_prev = indices_prev
                )
                # Adjust dimension of vector.
                v_prev_i = np.stack([v_i])

        # Normalization
        # norms = np.outer(np.diag(Si), np.diag(Si))
        # Si = np.multiply(Si, norms)
        # matrix_proj_i = np.multiply(matrix_proj_i, norms)

        return matrix_proj_i, Si
        # # return eigvalsh(matrix_proj_i, Si)


#         s, V = eigh(Si)
#         indx = s > 1e-12
#         lambda_s = np.diag(1/np.sqrt(s[indx]))
#         U = V[:, indx]@lambda_s

#         return eigvalsh(U.T.conj() @ matrix_proj_i @ U)
        # return U.T.conj() @ matrix_proj_i @ U