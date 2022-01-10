import numpy as np
from scipy.linalg import qr, qr_insert


def low_E_filter(v_rand, matrix, k):
    for i in range(k + 1):
        if i == 0:
            v_n = v_rand
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix.dot(v_nm1)
        else:
            v_np1 = 2 * matrix.dot(v_n) - v_nm1
            v_nm1 = v_n
            v_n = v_np1
    return v_n / np.linalg.norm(v_n, axis=0)


def basis(v_proj, matrix, dk, first_run=True, Q=None, R=None):
    count = 0
    for i in range(matrix.shape[0]):
        if i == 0:
            v_n = v_proj
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix.dot(v_nm1)
        else:
            v_np1 = 2 * matrix.dot(v_n) - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        if i == int(count * dk):
            vec = v_n / np.linalg.norm(v_n, axis=0)
            if count == 0 and first_run:
                Q, R = qr(vec, mode="economic")
            else:
                Q, R = qr_insert(
                    Q=Q, R=R, u=vec, k=Q.shape[1], which="col", overwrite_qru=True
                )
                ortho_condition = np.abs(np.diag(R)) < 1e-9
                if ortho_condition.any():
                    indices = np.invert(ortho_condition)
                    return Q[:, indices], R[indices, :][:, indices]
            count += 1
    return Q, R


def basis_no_store(v_proj, matrix, H, indices_to_store):
    S_xy = []
    H_xy = []
    # TODO: If k is too large, the norms of the vectors are from some large order.
    Kmax = indices_to_store[-1]
    for i in range(Kmax + 1):
        if i == 0:
            v_n = v_proj
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix.dot(v_nm1)
        else:
            v_np1 = 2 * matrix.dot(v_n) - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        if i in indices_to_store:
            v_store = v_n
            S_xy.append(v_proj.conj() @ v_n)
            H_xy.append(v_proj.conj() @ H.dot(v_n))
    return np.asarray(S_xy), np.asarray(H_xy)
