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

from pyDACP import core, chebyshev
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.linalg import eigh, eig, eigvalsh
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16

# +
N = 1000
np.random.seed(1)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)

plt.matshow(H.toarray().real, cmap='bwr', vmin=-1, vmax=1)
plt.show()
# -

# %%time
dacp=core.DACP_reduction(H, a=0.2, eps=0.05)

# %%time
true_eigvals, true_eigvecs = eig(H.todense())
v_proj = dacp.get_filtered_vector()

plt.scatter(np.real(true_eigvals), np.log(np.abs(true_eigvecs.T.conj()@v_proj)), c='k')
plt.xlim(-3*dacp.a, 3*dacp.a)
plt.axvline(-dacp.a, ls='--', c='k')
plt.axvline(dacp.a, ls='--', c='k')
plt.ylabel(r'$|c|$')
plt.xlabel(r'$E$')
plt.show()

# ### Second application of Chebyshev polynomials: the Chebyshev evolution
#
# Now we have one single vector $|r_E\rangle$ within the energy window we want. And we use again Chebyshev polynomials to span the full basis of the subspace $\mathbb{L}$ with $E \in [-a, a]$. For that, we define a second operator, $\mathcal{G}$, which is simply the rescaled Hamiltonian such that all eigenvalues are within $[-1, 1]$:
# \begin{equation}
# \mathcal{G} = \frac{\mathcal{H} - E_c'}{E_0'}
# \end{equation}
# with $E'_c = (E_{max} + E_{min})/2$, and $E'_0 = (E_{max} - E_{min})/2$.
#
# A full basis is then simply:
# \begin{equation}
# \left\lbrace I, \sin(X), \cdots, \sin(nX), cos(X), \cdots, cos(nX)\right\rbrace |r_E\rangle.
# \end{equation}
# with $X:=\pi\mathcal{G}/a_r$, and $a_r = a/\max(|E_{max}|, |E_{min}|)$.
#
# In fact, we can span the basis above by, instead of computing $\sin$ and $\cos$ of a matrix, computing simply several Chebyshev **polynomials** of $\mathcal{G}$.
#
# The remaininig problem is that we don't know the value of $n$, so we must (over)estimate the dimension of this subspace. And guess what: we use **again** Chebyshev polynomials by performing a low-resolution KPM. Since we overestimate the dimension, we also want to get rid of linearly dependent vectors, so we do SVD.
#
# The final set of vectors $\lbrace \psi_k \rbrace$ is then used to compute the projected low-energy Hamiltonian:
# \begin{equation}
# H_{\text{eff}}^{ij} = \langle \psi_i |\mathcal{H}|\psi_j\rangle.
# \end{equation}
#
# Let's see how good this Hamiltonian is:

# %%time
ham_red=dacp.get_subspace_matrix()

plt.matshow(ham_red.real, vmin=-.1, vmax=.1, cmap='bwr')
plt.show()

# %%time
red_eigvals, red_eigvecs = eig(ham_red)

# This part is quite unstable.
# Half of the bandwidth
half_bandwidth=np.max(np.abs(true_eigvals))
# Number 50 times smaller than level spacing.
err=half_bandwidth/N/2
# Get indices
indx1=np.min(red_eigvals)-err<=true_eigvals
indx2=true_eigvals<=np.max(red_eigvals)+err
indx=indx1 * indx2
# Extract real eigenvalues within the desired window.
window_eigvals=true_eigvals[indx]

plt.plot(np.sort(window_eigvals), np.sort(red_eigvals), '-o', c='k')
plt.xlabel(r'$E_n^{big}$')
plt.ylabel(r'$E_n^{small}$')
plt.xlim(-dacp.a, dacp.a)
plt.ylim(-dacp.a, dacp.a)
plt.show()

# res=np.sort(red_eigvals).copy()[:window_eigvals.shape[0]]
# res-=np.sort(window_eigvals)
res=np.sort(red_eigvals)-np.sort(window_eigvals)
plt.plot(np.sort(window_eigvals), np.log(res/dacp.a), '-o', c='k')
plt.ylabel(r'$\log(E_n^{big} - E_n^{small})$')
plt.xlabel(r'$E_n^{big}$')
plt.xlim(-dacp.a, dacp.a)
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
L = 15

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

fsyst=syst.finalized()

kwant.plot(syst)
plt.show()
# -

# %%time
H=fsyst.hamiltonian_submatrix(sparse=True)
dacp=core.DACP_reduction(H, a=0.2, eps=0.05, random_vectors=1)

# %%time
ham_red=dacp.get_subspace_matrix()

evals = eigvalsh(ham_red)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
true_vals = eigvalsh(H.todense())
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.xlim(-20, 20)
plt.ylim(-dacp.a, dacp.a)
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
