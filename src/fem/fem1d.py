"""1D linear (P1) finite elements for -(eps phi')' = f on [0, L] (protocol Sec. 9).

Assembly: element stiffness (eps_e / h_e) [[1,-1],[-1,1]], with eps_e the exact
element average of eps (equals the material value when the element does not
contain the interface). Load: consistent load for constant f, f*h_e/2 per node.
Dirichlet values imposed by row replacement. Field: E_e = -(phi_{j+1}-phi_j)/h_e,
piecewise constant; D_e = eps_e * E_e. Points exactly on a node use the element on
the requested material side, so a field is never averaged across the interface.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def element_eps(nodes, a, eps1, eps2):
    """Permittivity of each element; an element cut by the interface gets the length-weighted arithmetic mean."""
    l, r = nodes[:-1], nodes[1:]
    left_len = np.clip(np.minimum(r, a) - l, 0, None)
    right_len = np.clip(r - np.maximum(l, a), 0, None)
    return (eps1 * left_len + eps2 * right_len) / (r - l)


def solve(problem, n_elements, f=0.0):
    """Solve the 1D problem with linear finite elements on a uniform mesh of n_elements."""
    nodes = np.linspace(0.0, problem.L, n_elements + 1)
    h = np.diff(nodes)
    epse = element_eps(nodes, problem.a, problem.eps1, problem.eps2)
    n = nodes.size
    k = epse / h
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n - 1), np.arange(1, n), np.arange(1, n), np.arange(n - 1)])
    vals = np.concatenate([k, k, -k, -k])
    K = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tolil()
    b = np.zeros(n)
    if f:
        b[:-1] += f * h / 2; b[1:] += f * h / 2
    for i, val in ((0, problem.V), (n - 1, 0.0)):
        K.rows[i] = [i]; K.data[i] = [1.0]; b[i] = val
    phi = spla.spsolve(K.tocsr(), b)
    return FEMSolution(nodes, phi, epse, problem.a)


class FEMSolution:
    """Piecewise-linear FEM solution that can be evaluated like a prediction (potential and derivative)."""
    def __init__(self, nodes, phi, epse, a):
        self.nodes, self.phi, self.epse, self.a = nodes, phi, epse, a
        self.slope = np.diff(phi) / np.diff(nodes)
        self.conforming = bool(np.any(np.isclose(nodes, a, rtol=0, atol=1e-14)))
        self.n_elements = nodes.size - 1
        self.dofs = nodes.size - 2

    def _element(self, x, side):
        n_el = self.nodes.size - 1
        idx = np.searchsorted(self.nodes, x, side="right") - 1
        on_node = np.isclose(x, self.nodes[np.clip(idx, 0, n_el)], rtol=0, atol=1e-14)
        # on a node: material-1 side uses the element to the left
        idx = np.where(on_node & (np.asarray(side) == 1), idx - 1, idx)
        return np.clip(idx, 0, n_el - 1)

    def __call__(self, x, side):
        x = np.asarray(x, dtype=np.float64)
        e = self._element(x, side)
        phi = self.phi[e] + self.slope[e] * (x - self.nodes[e])
        return phi, self.slope[e]
