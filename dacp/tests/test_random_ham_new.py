# -*- coding: utf-8 -*-
# ## pyDACP: a python package for the Dual applications of Chebyshev polynomials method
#
# ### Why to even care?
#
# Do you work with:
# * Systems with large Hamiltonains?
# * Interested only within a small energy window?
#
# Then DACP is perfect for you: the idea is to span the low-energy subspace with two applications of Chebyshev polynomials. So if you original Hamiltonian had $n$ states and within the small window there are only $m$ states, you only have to diagonalize an $m\times m$ matrix.

# ### First application of Chebyshev polynomials
#
# We write an arbitrary vector in terms of the eigenvectors of the Hamiltonian $\mathcal{H}$:
# \begin{equation}
# |r\rangle = \sum_{E_i \in [-a, a]} \alpha_i |\psi_i\rangle + \sum_{E_i \notin [-a, a]} \beta_i |\phi_i\rangle.
# \end{equation}
#
# The idea now is to obtain an energy-filtered vector $|r_E\rangle$ by removing the second term of the equation above. To do so, we define the operator
# \begin{equation}
# \mathcal{F} := \frac{\mathcal{H}^2 - E_c}{E_0}
# \end{equation}
# with $E_c = E_{max}^2 + a^2$, and $E_0 = E_{max}^2 - a^2$.
#
# For a large enough $k$, the $k$-th order Chebyshev polynomial of $\mathcal{F}$ is
# \begin{equation}
# T_k(\mathcal{F}) \approx e^{\frac{2k}{E_{max}}\sqrt{a^2 - \mathcal{H}^2}},
# \end{equation}
# which indeed filters states within the $[-a, a]$ window. So,
# \begin{equation}
# T_k(\mathcal{F})|r\rangle = |r_E\rangle.
# \end{equation}
#
# Let's see an example!

# +
from dacp.dacp import eigh
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16

# +
N = int(1e3)
np.random.seed(1)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

# %%time
evals = eigh(
    H, window_size=0.1, eps=0.05, random_vectors=2, return_eigenvectors=False, filter_order=14
)

# %%time
true_vals, true_vecs=eigsh(H, return_eigenvectors=True, sigma=0, k=evals.shape[0])

true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.ylim(-0.1, 0.1)
# plt.xlim(np.min(n), np.max(n))
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
# plt.xlim(-0.1, 0.1)
# plt.yscale('log')
plt.show()

# ## The degeneracy problem and next steps
#
# This method works fine if there are no degeneracies. For degenerate states, however, we get a single eigenstate. The way to work around this problem is to first use as many random vectors as there are degenerate states. Since random vectors are in general linearly independent, they should lead to different degenerate states. 
#
# However, "in general" is not as formal as we wanted it to be. So ideally we want to make sure the random vectors are actually orthogonal. And it turns out that this is painful.
#
# But let's see how it goes anyway.

# +
import kwant

a, t = 1, 1
mu=0
L = 50

lat = kwant.lattice.honeycomb(a, norbs=1)

syst = kwant.Builder()

# Define the quantum dot
def circle(pos):
    (x, y) = pos
    return (-L < x < L) and (-L < y < L)

syst[lat.shape(circle, (0, 0))] = - mu
# hoppings in x-direction
syst[lat.neighbors()] = -t
# hoppings in y-directions
syst[lat.neighbors()] = -t

syst.eradicate_dangling()

fsyst=syst.finalized()

kwant.plot(syst)
plt.show()
# -

H=fsyst.hamiltonian_submatrix(sparse=True)

# %%time
evals=eigh(H, window_size=0.1, eps=0.05, random_vectors=2, return_eigenvectors=False, filter_order=12)

# %%time
true_vals = eigsh(H, return_eigenvectors=False, sigma=1e-10, k=evals.shape[0], which='LM')

n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, np.sort(true_vals), c='r', s=4)
#plt.ylim(-0.1, 0.1)
plt.ylabel(r'$E_n \ [t]$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(np.sort(true_vals) - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.xlim(-0.1, 0.1)
plt.yscale('log')
plt.show()

# ### Gram-Schmidt othogonalization
#
# A simple way to perform this orthogonalization is via Gram-Schmidt method. Say we already collected $m$ vectors after the first Chebyshev evolution, and let's assume all of them are orthogonal. And then we generate a second random vector $|r_2\rangle$. Then we orthogonalize it by simply computing:
# \begin{equation}
# |r_2^{\perp}\rangle = |r_2\rangle - \sum_{k=1}^{m} \langle \psi_k |r_2\rangle |\psi_k\rangle.
# \end{equation}
#
# This method is, however, unstable: the error is too large.

# ### QR decomposition with Householder reflections
#
# A stable method is orthogonalization via Householder reflections.
#
# #### Householder reflections
#
# Say you have an arbitrary $n$-dimensional vector $|x_1\rangle$, and you want to make a reflection via an operator $H_x$ on it such that $H_1|x_1\rangle = e_1$. Thus, you only have to find the vector $|v\rangle$ perpendicular to plane for this reflection, and then:
# $$
# H_1 = I - |v\rangle \langle v|.
# $$
#
# #### QR decomposition
#
# The QR decomposition is a decomposition of a matrix $A$ as $A = QR$, where $Q$ is an unitary matrix and $R$ is an upper triangular matrix. We can actually go from $A$ to $R$ by performing a sucession of Householder reflections:
# * Write $A$ as a sequence of vectors: $A = [|x_1\rangle, |x_2\rangle, \cdots, |x_m\rangle]$.
# * Perform a Householder reflection for $|x_1\rangle$, so $H_1 A = [e_1, H_1|x_2\rangle, \cdots, H_1|x_m\rangle]$.
# * Perform a Householder reflection $\tilde{H}_2$ in the last $n-1$ components of $|x_2\rangle$, such that:
# $$
# H_2 H_1 A = \left[ \begin{array}{cccc}
# 1 & H_1|x_2\rangle & \ & \ \\
# 0 & 1 & \cdots & H_2 H_1|x_m\rangle \\
# 0 & 0 & \ & \ \\
# \vdots & \vdots & \ddots & \vdots 
# \end{array} \right]
# $$
# * If all the vectors are linearly independent, we end up with a upper triangular matrix after performing these rotations $m$ times. Therefore,
# $$
# A = QR = (H_m \cdots H_1)^{\dagger} R.
# $$
#
# Since $Q$ is unitary, we get an orthogonal basis.
#
# #### Our workflow
#
# At each step of the Chebyshev evolution, we get a vector $|\psi_k\rangle$, which we Household reflect and generate $Q^{\dagger}$. Then we go to the next evolution step, and compute $Q^{\dagger}|\psi_{k+1}\rangle$. If all the last $m-(k+1)$ components of this vector are zero, we know this vector is linearly dependent to the previous ones. Then we stop the evolution. Otherwise, we go to the next step.
#
# When we stop the evolution (at step $m$), we collect `Q.T.conj()[:, :m]`, which is the current basis. Then ensure the next random vector is within the orthogonal complement of this basis and keep going until there are no more degeneracies.
#
# #### How to know there are no more degeneracies?
#
# * Start with 2 perpendicular random vectors.
# * See whether they lead to degenerate states.
# * If so, add one more vector, and check if there are 3-fold degeneracies.
# * Go on until at step $n$ the filtered vector in the orthogonal complement of the current basis is the null vector.

# ### Summary
#
# * Can use DACP method to find low-energy Hamiltonians.
# * I hate degeneracies.
