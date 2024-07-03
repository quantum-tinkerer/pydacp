# pyDACP

A python package to compute eigenvalues using the dual applications of Chebyshev polynomials algorithm. The algorithm is described in [SciPost Phys. 11, 103 (2021)](https://scipost.org/SciPostPhys.11.6.103).

This package implements an algorithm that computes the eigenvalues of hermitian linear operators within a given window. Besides the original algorithm, we also provide a way to deal with degeneracies systematically, and remove the need of prior estimations of the number of eigenvalues. The algorithm is useful for large, sufficiently sparse matrices.

This is an experimental version, so use at your own risk.

## Content

* Installation
* The algorithm
    + First application of Chebyshev polynomials
    + Second application of Chebyshev polynomials
    + Dealing with degeneracies
    + How do we save memory?
* Usage example

## Installation

After cloning the repository, simply run:
```
pip install .
```

## The algorithm

### First application of Chebyshev polynomials

We write an arbitrary vector in terms of the eigenvectors of an operator $\mathcal{H}$:

$$
|r\rangle = \sum_{E_i \in [-a, a]} \alpha_i |\psi_i\rangle + \sum_{E_i \notin [-a, a]} \beta_i |\phi_i\rangle.
$$

where $E_i$ are the eigenvalues corresponding to the eigenvectors $|\psi_i\rangle$.
We define an eigenvalue filter

$$
\mathcal{F} := \frac{\mathcal{H}^2 - E_c}{E_0}~.
$$

with $E_c = E_{max}^2 + a^2$, and $E_0 = E_{max}^2 - a^2$.
For sufficiently large $k$, the $k$-th order Chebyshev polynomial of $\mathcal{F}$ is

$$
T_k(\mathcal{F}) \approx e^{\frac{2k}{E_{max}}\sqrt{a^2 - \mathcal{H}^2}}.
$$

Thus, it removes the vector components with eigenvalues outside the window $[-a, a]$:

$$
|r_E\rangle = \mathcal{F}|r\rangle = \sum_{E_i \in [-a, a]} \alpha_i |\psi_i\rangle
$$

> The Chebyshev filter is less effective at the edge of the interval $[-a, a]$. As a consequence, incorrect eigenvalues leak into the window. We remove them by running the algorithm twice and identifying unstable values.

### Second application of Chebyshev polynomials

Now we have one vector $|r_E\rangle$ within the energy window we want.
And we use again Chebyshev polynomials to span the full basis of the subspace $\mathcal{L}$ with $E \in [-a, a]$.
For that, we define a second operator, $\mathcal{G}$, which is simply the rescaled operator such that all eigenvalues are within $[-1, 1]$:

$$
\mathcal{G} = \frac{\mathcal{H} - E_c'}{E_0'}
$$

with $E'_c = (E_{max} + E_{min})/2$, and $E'_0 = (E_{max} - E_{min})/2$.

One can generate a complete basis within the desired subspace as:

$$
\left\lbrace I, \sin(X), \cdots, \sin(nX), \cos(X), \cdots, \cos(nX)\right\rbrace |r_E\rangle.
$$

with $X:=\pi\mathcal{G}/a_r$, and $a_r = a/\mathrm{max}(|E_{max}|, |E_{min}|)$.

In fact, we can span the basis above by, instead of computing trigonometric functions of a matrix, computing several Chebyshev **polynomials** of $\mathcal{G}$.
This expansion is called *Chebychev evolution*.

The remaininig problem is that we do not know the value of $n$, so we must (over)estimate the dimension of this subspace.
To avoid an overestimation, we rather orthogonalize the subspace (with a QR decomposition) and stop the calculation when we cannot find another orthogonal vector to the existing set.

The final set of vectors $\lbrace \psi_k \rbrace$ is then used to compute the projected operator within the desired eigenvalue window:

$$
H_{\text{eff}}^{ij} = \langle \psi_i |\mathcal{H}|\psi_j\rangle.
$$

### Dealing with degeneracies

The method above is not able to resolve degeneracies: each random vector can only span a non-degenerate subspace of $\mathcal{L}$.
We solve the problem by adding more random vectors.

To make sure a complete basis is generated, after the end of each *Chebyshev evolution*, we check the orthogonality of the subspace by performing a QR decomposition of the overlap matrix $S$.
We stop the calculation when adding a random vector does not increase the dimension of the target subspace.

#### How do we save memory?

DACP provides an algorithm to directly compute the projected and overlap matrices without storing all vectors.
This is possible because of two properties combined:
1. $[T_i(\mathcal{H}), \mathcal{H}]=0$
2. $T_i(\mathcal{H})T_j(\mathcal{H}) = \frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right)~.$

Combining those two properties, we only need to store the filtered vectors from previous runs $|r_E^{pref}\rangle$ and compute:

$$
S_{ij} = \left\langle r_E^{prev}\right| \frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right)\rangle \left|r_E^{current}\right\rangle~.
$$

and

$$
H_{ij} = \left\langle r_E^{prev}\right| \mathcal{H} \left[\frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right) \right] \left|r_E^{current}\right\rangle~.
$$

> Because there reduction of memory usage when the eigenvectors are computed, we do not include an eigenvector calculator. For this particular use we recommend other established sparse methods such as Arnoldi.

## Usage example

```python
from dacp.dacp import eigvalsh, eigh
# Generate a random matrix with size 100 x 100
N = 100
matrix = random_values((N,N)) + random_values((N,N))*1j
matrix = (matrix.conj().T + matrix)/(2*np.sqrt(N))
matrix = csr_matrix(matrix)
# Compute eigenvalues
eigvals = eigvalsh(matrix, window_size)
```
