# -*- coding: utf-8 -*-
# %%
from dacp import estimated_errors, eigvalsh
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse import kron

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18

# %% [markdown]
# Let's show a basic usage of the algorithm. Let us start creating a random tri-diagonal hermitian matrix.

# %%
N = int(5e2)
c = 2 * (np.random.rand(N - 1) + np.random.rand(N - 1) * 1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
H = diags(b, offsets=0)

# %% [markdown]
# We will compare our results with dense diagonalization. We also normalize the eigenvalues with the window.

# %%
true_vals, true_vecs = np.linalg.eigh(H.todense())

Emax = np.max(true_vals)
true_vals /= Emax
H /= Emax

# %% [markdown]
# We now run the DACP eigenvalue solver. We set up a tolerance `tol`, the number of random vectors per run `k`, and define an eigenvalue window $[-a, 2a]$.

# %%
k = 12
a = 0.1
tol = 1e-3
window = [-a, 2 * a]
evals = eigvalsh(H, window=window, random_vectors=10, filter_order=k, tol=tol)

# %% [markdown]
# For comparison, we find the closest eigenvalues found by DACP to the ones found by dense diagonalization.

# %%
map_eigv = []
for value in evals:
    closest = np.abs(true_vals - value).min()
    map_eigv.append(true_vals[np.abs(true_vals - value) == closest][0])
true_vals = np.array(map_eigv)

# %% [markdown]
# We first observe that they match.

# %%
true_vals = np.sort(true_vals)
n = np.arange(-evals.shape[0] / 2, evals.shape[0] / 2)
plt.scatter(n, evals, c="k")
n_true = np.arange(-true_vals.shape[0] / 2, true_vals.shape[0] / 2)
plt.scatter(n_true, true_vals, c="r", s=4)
plt.ylabel(r"$\lambda_i$")
plt.xlabel(r"$n$")
plt.show()

# %% [markdown]
# And visualize the relative error.

# %%
error = np.abs((true_vals - evals) / evals)
true_vals = np.sort(true_vals)
n = np.arange(-evals.shape[0] / 2, evals.shape[0] / 2)
plt.scatter(n, evals, c=np.log10(error), s=20, cmap="inferno")
n_true = np.arange(-true_vals.shape[0] / 2, true_vals.shape[0] / 2)
plt.colorbar()
plt.scatter(n_true, true_vals, c="k", s=2)
plt.ylabel(r"$\lambda_i$")
plt.xlabel(r"$n$")
plt.show()

# %% [markdown]
# We can also visualize the relative error for each eigenvalue.

# %%
plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r"$\delta \lambda_i$")
plt.xlabel(r"$\lambda_i$")
plt.yscale("log")
plt.axhline(np.finfo(float).eps, ls="--", c="k")
plt.axhline(1 / N, ls="--", c="k")
plt.show()

# %% [markdown]
# Finally, we can also estimate the errors.

# %%
Ei = np.linspace(window[0], window[1], 300)
eta = estimated_errors(
    Ei,
    window,
    tol=tol,
    filter_order=k,
)


plt.plot(Ei, eta, "r")
plt.fill_between(Ei, 0.01 * eta, 100 * eta, alpha=0.4, fc="r")
plt.scatter(evals, np.abs((true_vals - evals) / evals), c="k", zorder=10, s=1)
plt.ylabel(r"$|\delta \lambda_i/\lambda_i|$")
plt.xlabel(r"$\lambda_i$")
plt.yscale("log")
plt.xlim(window[0], window[1])
plt.show()

# %%
plt.scatter(evals, estimated_errors(evals, window), c="b", s=10, marker="+")
plt.plot(Ei, eta, "r")
plt.ylabel(r"$|\delta \lambda_i/\lambda_i|$")
plt.xlabel(r"$\lambda_i$")
plt.yscale("log")
plt.xlim(window[0], window[1])
plt.show()

# %% [markdown]
# ## Degenerate case

# %% [markdown]
# We create a random tridiagonal matrix for which all eigenvalues are 4-fold degenerate.

# %%
N = int(250)
c = 2 * (np.random.rand(N - 1) + np.random.rand(N - 1) * 1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
H = kron(H, eye(4))

# %%
# %%time
window = [-a, a]
evals = eigvalsh(H, window=window, random_vectors=5, filter_order=k, tol=tol)

# %%
# %%time
true_vals = np.linalg.eigvalsh(H.todense())

# %%
map_eigv = []
for value in evals:
    closest = np.abs(true_vals - value).min()
    map_eigv.append(true_vals[np.abs(true_vals - value) == closest][0])
true_vals = np.array(map_eigv)

# %%
true_vals = np.sort(true_vals)
n = np.arange(-evals.shape[0] / 2, evals.shape[0] / 2)
plt.scatter(n, evals, c="k")
n_true = np.arange(-true_vals.shape[0] / 2, true_vals.shape[0] / 2)
plt.scatter(n_true, true_vals, c="r", s=4)
plt.ylabel(r"$\lambda_i$")
plt.xlabel(r"$n$")
plt.show()

# %%
plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r"$\delta \lambda_i$")
plt.xlabel(r"$\lambda_i$")
plt.yscale("log")
plt.axhline(np.finfo(float).eps, ls="--", c="k")
plt.show()

# %%
Ei = np.linspace(window[0], window[1], 300)
eta = estimated_errors(
    Ei,
    window,
    tol=tol,
    filter_order=k,
)

plt.plot(Ei, eta, "r")
plt.fill_between(Ei, 0.01 * eta, 100 * eta, alpha=0.4, fc="r")
plt.scatter(evals, np.abs((true_vals - evals) / evals), c="k", zorder=10, s=1)
plt.ylabel(r"$|\delta \lambda_i/\lambda_i|$")
plt.xlabel(r"$\lambda_i$")
plt.yscale("log")
plt.xlim(-a, a)
plt.show()
