"""2D FEM reference/cross-check (change log v1.2.0).
Mesh: gmsh, box [-1,1]^2; interface-conforming meshes embed the material interface (body-fitted,
straight-sided triangles: polygonal interface), non-conforming meshes do not.
Discretisation: scikit-fem, Lagrange P2 (or P1), element-wise constant eps, Dirichlet data
h interpolated at boundary DOFs. Evaluation: pred(X, side) evaluates the polynomial of the
triangle containing X when its material matches `side`, otherwise the polynomial of the nearest
triangle of material `side` (extrapolated a distance O(h^2)); this gives one-sided interface
limits and never mixes materials. For non-conforming meshes side is ignored (no material-wise
representation exists) and the element coefficient is eps at the element centroid."""
import time
import numpy as np
import gmsh
import skfem
from skfem.helpers import dot, grad
from scipy.spatial import cKDTree
import matplotlib.tri as mtri


def mesh_box(h, geometry=None, conforming=True):
    """geometry: callable adding interface entities inside the box and returning (tags of material-2 surfaces)
    via gmsh.model.occ; returns nodes (2,N), triangles (3,M), material tag per triangle (1/2)."""
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("m"); occ = gmsh.model.occ
    box = occ.addRectangle(-1, -1, 0, 2, 2)
    mat2 = []
    if conforming and geometry is not None:
        inc = geometry(occ)
        out, mp = occ.fragment([(2, box)], [(2, t) for t in inc])
        occ.synchronize()
        inc_set = set()
        for i, t in enumerate(inc):
            inc_set |= {e[1] for e in mp[len([box]) + i]}
        mat2 = sorted(inc_set)
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    idx = {t: i for i, t in enumerate(tags)}; P = coords.reshape(-1, 3)[:, :2].T
    T, M = [], []
    for dim, s in gmsh.model.getEntities(2):
        et, _, enodes = gmsh.model.mesh.getElements(2, s)
        for e, nds in zip(et, enodes):
            if e == 2:
                tri = np.vectorize(idx.get)(nds.reshape(-1, 3)).T; T.append(tri); M.append(np.full(tri.shape[1], 2 if s in mat2 else 1))
    gmsh.finalize()
    T = np.concatenate(T, 1); M = np.concatenate(M)
    used = np.unique(T); remap = -np.ones(P.shape[1], int); remap[used] = np.arange(len(used))
    return P[:, used], remap[T], M


@skfem.BilinearForm
def _lap(u, v, w): return dot(grad(u), grad(v))


def solve(problem, h, order=2, conforming=True, geometry=None):
    t0 = time.perf_counter()
    P, T, M = mesh_box(h, geometry, conforming)
    t_mesh = time.perf_counter() - t0
    t1 = time.perf_counter()
    mesh = skfem.MeshTri(P, T)
    if not conforming:   # element coefficient from the true material at the centroid
        c = P[:, T].mean(1).T; M = problem.side_of(c)
    e = skfem.ElementTriP2() if order == 2 else skfem.ElementTriP1()
    basis = skfem.Basis(mesh, e)
    K = None
    for mat, epsv in ((1, problem.eps1), (2, problem.eps2)):
        el = np.nonzero(M == mat)[0]
        if len(el) == 0: continue
        Ki = epsv * skfem.asm(_lap, skfem.Basis(mesh, e, elements=el))
        K = Ki if K is None else K + Ki
    D = basis.get_dofs().all() if hasattr(basis.get_dofs(), "all") else basis.get_dofs().nodal["u"]
    x = np.zeros(basis.N); x[D] = problem.h_train_np(basis.doflocs[:, D].T)
    b = np.zeros(basis.N)
    if hasattr(problem, "interface_load"):          # weak form: int eps grad w . grad v = int_Gamma Q v ds
        fac = problem.interface_facets(mesh)
        if len(fac):
            fb = skfem.FacetBasis(mesh, e, facets=fac)
            b = skfem.asm(skfem.LinearForm(lambda v, w: problem.interface_load(w.x) * v), fb)
    u = skfem.solve(*skfem.condense(K, b, x=x, D=D))
    t_solve = time.perf_counter() - t1
    sol = FEM2DSolution(mesh, basis, u, M, conforming, order, t_mesh, t_solve); sol.problem = problem; return sol


class FEM2DSolution:
    def __init__(self, mesh, basis, u, M, conforming, order, t_mesh, t_solve):
        self.mesh, self.basis, self.u, self.M, self.conforming, self.order = mesh, basis, u, M, conforming, order
        self.t_mesh, self.t_solve = t_mesh, t_solve
        p, t = mesh.p, mesh.t
        self.tri = mtri.Triangulation(p[0], p[1], t.T); self.finder = self.tri.get_trifinder()
        cent = p[:, t].mean(1).T
        self.trees = {m: (cKDTree(cent[M == m]), np.nonzero(M == m)[0]) for m in (1, 2) if np.any(M == m)}
        self.edofs = basis.element_dofs              # (ndofs_per_elem, nelem)
        self.n_elements = t.shape[1]; self.dofs = basis.N

    def _eval_in(self, X, el):
        p, t = self.mesh.p, self.mesh.t
        A = p[:, t[0, el]].T; B = p[:, t[1, el]].T; C = p[:, t[2, el]].T
        det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
        g1 = np.c_[C[:, 1] - A[:, 1], A[:, 0] - C[:, 0]] / det[:, None]     # grad lambda_1
        g2 = np.c_[A[:, 1] - B[:, 1], B[:, 0] - A[:, 0]] / det[:, None]     # grad lambda_2
        g0 = -(g1 + g2)
        l1 = ((X[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (X[:, 1] - A[:, 1])) / det
        l2 = ((B[:, 0] - A[:, 0]) * (X[:, 1] - A[:, 1]) - (X[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])) / det
        l0 = 1 - l1 - l2
        L = [l0, l1, l2]; G = [g0, g1, g2]
        U = self.u[self.edofs[:, el]]
        if self.order == 1:
            phi = sum(U[i] * L[i] for i in range(3)); gr = sum(U[i][:, None] * G[i] for i in range(3))
            return phi, gr
        # P2 basis, skfem ordering: vertices 0,1,2 then edges (0,1), (1,2), (0,2)
        Nf = [L[0] * (2 * L[0] - 1), L[1] * (2 * L[1] - 1), L[2] * (2 * L[2] - 1), 4 * L[0] * L[1], 4 * L[1] * L[2], 4 * L[0] * L[2]]
        Ng = [(4 * L[0] - 1)[:, None] * G[0], (4 * L[1] - 1)[:, None] * G[1], (4 * L[2] - 1)[:, None] * G[2],
              4 * (L[0][:, None] * G[1] + L[1][:, None] * G[0]), 4 * (L[1][:, None] * G[2] + L[2][:, None] * G[1]),
              4 * (L[0][:, None] * G[2] + L[2][:, None] * G[0])]
        phi = sum(U[i] * Nf[i] for i in range(6)); gr = sum(U[i][:, None] * Ng[i] for i in range(6))
        return phi, gr

    def __call__(self, X, side):
        X = np.asarray(X, float); side = np.asarray(side)
        el = np.asarray(self.finder(X[:, 0], X[:, 1]))
        if self.conforming:
            for m in (1, 2):
                if m not in self.trees: continue
                bad = (side == m) & ((el < 0) | (self.M[np.clip(el, 0, None)] != m))
                if bad.any():
                    tree, ids = self.trees[m]; el[bad] = ids[tree.query(X[bad])[1]]
        out = el < 0
        if out.any():
            tree = cKDTree(self.mesh.p[:, self.mesh.t].mean(1).T); el[out] = tree.query(X[out])[1]
        phi, g = self._eval_in(X, el)
        return self.problem.to_physical(X, phi, g, side) if hasattr(self, "problem") else (phi, g)
