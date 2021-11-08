import numpy as np


def low_E_filter(v_rand, matrix, k):
    for i in range(k+1):
        if i == 0:
            v_n = v_rand
            continue

        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
            continue
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
            continue
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
            continue
        else:
            v_np1 = 2 * matrix @ v_n - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        if i in indices:
            v_basis.append(v_n / np.linalg.norm(v_n))
    return np.asarray(v_basis)
