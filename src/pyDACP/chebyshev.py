import numpy as np

def low_E_filter(v_rand, matrix, k):
    for i in range(k+1):
        if i == 0:
            v_n = v_rand
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
        else:
            v_np1 = 2*matrix @ v_n - v_nm1
            v_nm1 = v_n
            v_n = v_np1
    return v_n/np.linalg.norm(v_n)


def basis(v_proj, matrix, indices):
    v_basis = []
    # TODO: If k is too large, the norms of the vectors are from some large order.
    k = indices[-1]
    for i in range(k+1):
        if i == 0:
            v_n = v_proj
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
        else:
            v_np1 = 2 * matrix @ v_n - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        if i in indices:
            v_basis.append(v_n / np.linalg.norm(v_n))
    return np.asarray(v_basis)


def basis_no_store(v_proj, matrix, H, Kmax):
    S_xy = []
    H_xy = []
    # TODO: If k is too large, the norms of the vectors are from some large order.
    for i in range(Kmax+1):
        if i == 0:
            v_n = v_proj
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
        else:
            v_np1 = 2 * matrix @ v_n - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        # if i in indices_to_store:
        v_store = v_n
        S_xy.append(v_proj.conj() @ v_n)
        H_xy.append(v_proj.conj() @ H @ v_n)
    return np.asarray(S_xy), np.asarray(H_xy)
