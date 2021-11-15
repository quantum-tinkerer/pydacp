from pyDACP import core
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import eye, diags
import math
from scipy.linalg.lapack import zlarf, zlarfg

# +
N=10
M=7
vs=np.random.rand(N-1, M)
vs=np.append(vs, np.sum(vs, axis=1).reshape(N-1, 1), axis=1)
# vs=np.random.rand(N-1, M)
vs=np.concatenate((vs[:2,:], np.zeros((1,M+1)), vs[2:,:]))

P = np.eye(N)
Pi = np.eye(N)

for i in range(vs.shape[1]):
    r = np.abs(np.linalg.norm((P @ vs)[i:, i]))
    print(r)
    if np.isclose(r, 0):
        print('Ended with ' + str(i) + ' vectors.')
        basis=(P@Pi).T.conj()[:, :i]
        break
    else:
        vs = P @ vs
        Pi = P @ Pi
        beta, v_orth, tau = zlarfg(N - i, vs[i,i], vs[i+1:,i])
        v_orth = np.array([1, *v_orth])
        P=(np.eye(N-i) - tau * np.outer(v_orth, v_orth.conj()))/beta
        P = scipy.linalg.block_diag(np.eye(i), P)
    if i+1 == vs.shape[1]:
        print('All the vectors are linearly independent.')
        basis=(P@Pi).T.conj()[:, :i+1]
# -

plt.matshow(np.abs(basis))
plt.colorbar()

N=10
M=8
# vs=np.random.rand(N, M)
# vs=np.append(vs, np.sum(vs, axis=1).reshape(N, 1), axis=1)
vs=np.random.rand(N-1, M)
vs=np.concatenate((vs[:2,:], np.zeros((1,M)), vs[2:,:]))
plt.matshow(np.abs(vs))
