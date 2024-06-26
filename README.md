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
        - Eigenvalues + eigenvectors method (under development)
        - Eigenvalues-only method
* Usage example

## Installation

After cloning the repository, simply run:
```
pip install .
```

## The algorithm

### First application of Chebyshev polynomials

We write an arbitrary vector in terms of the eigenvectors of the Hamiltonian $`\mathcal{H}`$:
```math
|r\rangle = \sum_{E_i \in [-a, a]} \alpha_i |\psi_i\rangle + \sum_{E_i \notin [-a, a]} \beta_i |\phi_i\rangle.
```

The idea now is to obtain an energy-filtered vector $`|r_E\rangle`$ by removing the second term of the equation above.
To do so, we define the operator
```math
\mathcal{F} := \frac{\mathcal{H}^2 - E_c}{E_0}
```
with $`E_c = E_{max}^2 + a^2`$, and $`E_0 = E_{max}^2 - a^2`$.

For a large enough $`k`$, the $`k`$-th order Chebyshev polynomial of $`\mathcal{F}`$ is
```math
T_k(\mathcal{F}) \approx e^{\frac{2k}{E_{max}}\sqrt{a^2 - \mathcal{H}^2}},
```
which indeed filters states within the $`[-a, a]`$ window. So,
```math
T_k(\mathcal{F})|r\rangle = |r_E\rangle.
```

### Second application of Chebyshev polynomials

Now we have one single vector $`|r_E\rangle`$ within the energy window we want.
And we use again Chebyshev polynomials to span the full basis of the subspace $`\mathcal{L}`$
with $`E \in [-a, a]`$.
For that, we define a second operator, $`\mathcal{G}`$,
which is simply the rescaled Hamiltonian such that all eigenvalues are within $`[-1, 1]`$:
```math
\mathcal{G} = \frac{\mathcal{H} - E_c'}{E_0'}
```
with $`E'_c = (E_{max} + E_{min})/2`$, and $`E'_0 = (E_{max} - E_{min})/2`$.

A full basis is then simply:
```math
\left\lbrace I, \sin(X), \cdots, \sin(nX), \cos(X), \cdots, \cos(nX)\right\rbrace |r_E\rangle.
```
with $`X:=\pi\mathcal{G}/a_r`$, and $`a_r = a/\mathrm{max}(|E_{max}|, |E_{min}|)`$.

In fact, we can span the basis above by, instead of computing trigonometric functions of a matrix, computing simply several Chebyshev **polynomials** of $`\mathcal{G}`$.

The remaininig problem is that we don't know the value of $`n`$, so we must (over)estimate the dimension of this subspace.
And guess what: we use **again** Chebyshev polynomials by performing a low-resolution KPM.
Since we overestimate the dimension, we also want to get rid of linearly dependent vectors, so we do SVD.

The final set of vectors $`\lbrace \psi_k \rbrace`$ is then used to compute the projected low-energy Hamiltonian:
```math
H_{\text{eff}}^{ij} = \langle \psi_i |\mathcal{H}|\psi_j\rangle.
```

### Dealing with degeneracies

The method above is not able to resolve degeneracies: each random vector can only span a non-degenerate subspace of $`\mathcal{L}`$.
We solve the problem by adding more random vectors.

To make sure a complete basis is generated, after the end of each Chebolution&trade run finishes, we perform a QR decomposition of the overlap matrix $`S`$:
```math
S = QR~.
```
When adding new random vectors no longer increase number of non-zero elements in $`\mathrm{diag} R`$, we stop the calculation.

#### How do we save memory?

DACP provides an algorithm to directly compute the projected and overlap matrices without storing all vectors.
This is possible because of two properties combined:
1. $`[T_i(\mathcal{H}), \mathcal{H}]=0`$
2. $`T_i(\mathcal{H})T_j(\mathcal{H}) = \frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right)~.`$

Combining those two properties, we only need to store the filtered vectors from previous runs $`|r_E^{pref}\rangle`$ and compute:
```math
S_{ij} = \left r_E^{prev}| \left[\frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right) |r_E^{current}\rangle\right]~.
```
and
```math
H_{ij} = \left r_E^{prev}| \mathcal{H} \left[\frac12 \left(T_{i+j}(\mathcal{H}) + T_{|i-j|}(\mathcal{H}) \right) |r_E^{current}\rangle\right]~.
```


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
