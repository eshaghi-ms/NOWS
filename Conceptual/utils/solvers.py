import time

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import diags, block_diag
from scipy.sparse.linalg import spsolve, cg
# import pyamg
# from petsc4py import PETSc


def Poisson1D_solve(f):
    r"""
    Functions for solving the Poisson 1D equation by the finite difference/finite
    element method

    Solves the 1D Poisson equation:
        -u''(x) = f(x) for x in (0,1)
        with boundary conditions u(0)=u(1)=0

    Parameters
    ----------
    f : (tensor) 2D array
        the value of the function f evaluated at equally spaced points (x_1...x_N)
        with 0<x_1<x_2<...<x_N<1 for each input function

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at points x_1, ..., x_N

    """
    n = f.shape[1]
    main_diag = 2. * np.ones(n)
    off_diag = -1. * np.ones(n - 1)
    lhs = diags((main_diag, off_diag, off_diag), offsets=[0, -1, 1], format='csc')
    u = spsolve(lhs, f.T / (n + 1) ** 2).T

    return u


def Poisson2D_solve(f):
    r"""
    Solves the 2D Poisson equation:
        -u''(x,y) = f(x,y) for x, y in (0,1)
        with boundary conditions u(0,y) = u(1,y) = 0, u(x,0) = 0, u(x,1) = 0

    Parameters
    ----------
    f : 2D array
        the value of the function f evaluated at equally spaced grids (x_1...x_N) and (y_1...y_N)
        with 0<x_1<x_2<...<x_N<1 and 0<y_1<y_2<...<y_N<1 for each input function

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at spaced grids (x_1...x_N) and (y_1...y_N)

    """
    n = f.shape[1]
    n_train = f.shape[0]
    f = f.reshape(n_train, n * n)
    h = 1.0 / (n + 1)

    diagonals = [4 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
    a_1d = diags(diagonals, [0, -1, 1])

    blocks = [a_1d] * n
    a = block_diag(blocks)
    a -= diags([np.ones(n * (n - 1)), np.ones(n * (n - 1))], [-n, n])

    t_start = time.time()
    u = spsolve(a / h ** 2, f.T).T
    print(f"Elapsed time for spsolve = {time.time() - t_start}")

    u = u.reshape(n_train, n, n)
    return u


def Poisson2D_itsolve(f, u0=None, rtol=1e-5):
    r"""
    Solves the 2D Poisson equation:
        -u''(x,y) = f(x,y) for x, y in (0,1)
        with boundary conditions u(0,y) = u(1,y) = 0, u(x,0) = 0, u(x,1) = 0

    Parameters
    ----------
    f : 2D array
        the value of the function f evaluated at equally spaced grids (x_1...x_N) and (y_1...y_N)
        with 0<x_1<x_2<...<x_N<1 and 0<y_1<y_2<...<y_N<1 for each input function
    u0: initial guess
    rtol: relative error

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at spaced grids (x_1...x_N) and (y_1...y_N)
    """
    t_start = time.time()
    n = f.shape[1]
    n_train = f.shape[0]
    f = f.reshape(n_train, n * n)
    u0 = u0.reshape(n_train, n * n) if u0 is not None else None
    h = 1.0 / (n + 1)

    diagonals = [4 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
    a_1d = diags(diagonals, [0, -1, 1])

    blocks = [a_1d] * n
    a = block_diag(blocks)
    a -= diags([np.ones(n * (n - 1)), np.ones(n * (n - 1))], [-n, n])

    u = np.zeros_like(f)
    time_residual_hist = []
    for i in range(n_train):
        b_i = f[i, :] * h ** 2
        t0 = time.perf_counter()
        tr_hist = []

        def cb(xk):
            r_norm = np.linalg.norm(b_i - a @ xk)
            tr_hist.append((time.perf_counter() - t0, r_norm))

        if u0 is None:
            u[i, :], info = cg(a, b_i, rtol=rtol, callback=cb)
        else:
            u[i, :], info = cg(a, b_i, x0=u0[i], rtol=rtol, callback=cb)

        time_residual_hist.append(np.asarray(tr_hist))
        if info != 0:
            print(f"Warning: CG did not converge for column {i}")

    total_time = time.time() - t_start
    u = u.reshape(n_train, n, n)
    return u, total_time, time_residual_hist


def Poisson2D_pyagm_solve(f, u0=None, rtol=1e-5):
    """
    Solves the 2D Poisson equation: -u''(x,y) = f(x,y) for x, y in (0,1)
    with zero Dirichlet boundary conditions.

    Parameters
    ----------
    f  : (n_train, n, n) array
         Right–hand side evaluated on an n×n interior grid.
    u0 : (n_train, n, n) array or None
         Initial guess.
    rtol

    Returns
    -------
    u : (n_train, n, n) array
        Solution on the interior grid.
    total_time : float
        Wall-clock time spent inside the solver loop (seconds).
    time_residual_hist : list of (n_steps, 2) arrays
        For each training sample, an array whose columns are
        [elapsed_seconds, residual_norm] recorded at every AMG cycle.
    """
    n = f.shape[1]
    n_train = f.shape[0]
    f = f.reshape(n_train, n * n)
    u0 = u0.reshape(n_train, n * n) if u0 is not None else None
    h = 1.0 / (n + 1)

    # Build the 5-point Laplacian exactly as before
    diagonals = [4 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
    a_1d = diags(diagonals, [0, -1, 1])
    blocks = [a_1d] * n
    a = block_diag(blocks)
    a -= diags([np.ones(n * (n - 1)), np.ones(n * (n - 1))], [-n, n])

    u = np.zeros_like(f)

    # AMG preconditioner (setup cost is *not* counted in total_time)
    ml = pyamg.ruge_stuben_solver(a)

    time_residual_hist = []
    t_start = time.time()

    for i in range(n_train):
        b_i = f[i, :] * h ** 2
        x0 = None if u0 is None else u0[i]

        t0 = time.perf_counter()
        residuals = []  # will be filled by pyamg
        u[i, :] = ml.solve(b_i,
                           x0=x0,
                           tol=rtol,
                           residuals=residuals)
        elapsed = time.perf_counter() - t0
        # Convert residual list to (elapsed_seconds, residual_norm) pairs
        # We linearly scale the cycle index to elapsed time for simplicity
        n_cycles = len(residuals)
        times = np.linspace(0, elapsed, n_cycles)
        time_residual_hist.append(
            np.column_stack((times, residuals))
        )

    total_time = time.time() - t_start

    u = u.reshape(n_train, n, n)
    return u, total_time, time_residual_hist


def Poisson2D_petsc_solve(f, u0=None, rtol=1e-5):
    r"""
    Solves the 2D Poisson equation:
        -u''(x,y) = f(x,y) for x, y in (0,1)
        with boundary conditions u(0,y) = u(1,y) = 0, u(x,0) = 0, u(x,1) = 0

    Parameters
    ----------
    f : 2D array
        the value of the function f evaluated at equally spaced grids (x_1...x_N) and (y_1...y_N)
        with 0<x_1<x_2<...<x_N<1 and 0<y_1<y_2<...<y_N<1 for each input function
    u0: initial guess
    rtol: relative tolerance for iterative solver

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at spaced grids (x_1...x_N) and (y_1...y_N)
    total_time : float
        total elapsed wall-clock time
    time_residual_hist : list of arrays
        for each sample, array of (time, residual_norm)
    """
    n = f.shape[1]
    n_train = f.shape[0]
    f = f.reshape(n_train, n * n)
    u0 = u0.reshape(n_train, n * n) if u0 is not None else None
    h = 1.0 / (n + 1)

    diagonals = [4 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
    a_1d = diags(diagonals, [0, -1, 1])

    blocks = [a_1d] * n
    a = block_diag(blocks)
    a -= diags([np.ones(n * (n - 1)), np.ones(n * (n - 1))], [-n, n])

    # Convert to PETSc matrix
    a = PETSc.Mat().createAIJ(size=(n * n, n * n), csr=(a.indptr, a.indices, a.data))

    u = np.zeros_like(f)

    total_time_start = time.time()
    time_residual_hist = []

    for i in range(f.shape[0]):
        b_i = PETSc.Vec().createWithArray(np.copy(f[i, :] * h ** 2), comm=PETSc.COMM_SELF)

        if u0 is None:
            x = PETSc.Vec().createWithArray(np.copy(u[i, :]), comm=PETSc.COMM_SELF)
        else:
            x = PETSc.Vec().createWithArray(np.copy(u0[i]), comm=PETSc.COMM_SELF)

        # Setup solver for each rhs
        ksp = PETSc.KSP().create()
        ksp.setOperators(a)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ICC)  # Preconditioner
        ksp.setTolerances(rtol=rtol)

        # Collect residual history with timing
        tr_hist = []
        t0 = time.perf_counter()

        def monitor(ksp, its, rnorm):
            tr_hist.append((time.perf_counter() - t0, rnorm))

        ksp.setMonitor(monitor)

        if u0 is not None:
            ksp.setInitialGuessNonzero(True)

        ksp.solve(b_i, x)
        u[i, :] = x.getArray()

        time_residual_hist.append(np.asarray(tr_hist))

        if ksp.getConvergedReason() <= 0:
            print(f"Warning: PETSc KSP did not converge for sample {i}")

    total_time = time.time() - total_time_start

    u = u.reshape(n_train, n, n)
    return u, total_time, time_residual_hist


def solve_gwf(coef, f):
    k = coef.shape[0]
    coef = coef.T

    f = f[1:k - 1, 1:k - 1]

    d = [[sparse.csr_matrix((k - 2, k - 2)) for _ in range(k - 2)] for _ in range(k - 2)]
    for j in range(1, k - 1):
        main_diag = (coef[:k - 2, j] + coef[1:k - 1, j]) / 2 + \
                    (coef[2:, j] + coef[1:k - 1, j]) / 2 + \
                    (coef[1:k - 1, j - 1] + coef[1:k - 1, j]) / 2 + \
                    (coef[1:k - 1, j + 1] + coef[1:k - 1, j]) / 2

        off_diag = -((coef[1:k - 1, j] + coef[2:, j]) / 2)[:-1]
        lower_diag = -((coef[:k - 2, j] + coef[1:k - 1, j]) / 2)[1:]

        d[j - 1][j - 1] = sparse.diags([lower_diag, main_diag, off_diag], [-1, 0, 1])
        if j < k - 2:
            d[j - 1][j] = sparse.diags(-(coef[1:k - 1, j] + coef[1:k - 1, j + 1]) / 2, 0)
            d[j][j - 1] = d[j - 1][j]

    a = sparse.bmat(d, format="csc") * (k - 1) ** 2
    return a, f


def Darcy2D_solve(coefs):
    r"""
    Solve the Darcy 2D equation -\nabla dot(a(x,y)\nabla u(x,y)) = f(x,y), for
    (x,y) in \Omega
    with boundary conditions u(x,y) = 1 for (x,y) in \partial \Omega

    Parameters
    ----------
    coefs : (array of size [N, s, s])
        the coefficients a(x,y) sampled at equally spaced points on the domain \Omega
        (including the boundary) for each of the N inputs

    Returns
    -------
    U_all : (array of size [N, s, s])
        the values of U at equally spaced points in the domain \Omega (including the
        boundary) for each of the N inputs

    """
    u_all = np.zeros_like(coefs)
    assert coefs.shape[1] == coefs.shape[2], "The second and third dimensions should have the same size"
    n, s, _ = coefs.shape

    f = np.ones((s, s))
    t_start = time.time()
    for i in range(n):
        # if i % 10 == 0:
        # print(f"Generating the {i}th solution")
        coef = coefs[i, :, :]
        a, _f = solve_gwf(coef, f)
        u_interior = spsolve(a, _f.ravel()).reshape(s - 2, s - 2)
        u_all[i] = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')
    print(f"Elapsed time for direct solver = {time.time() - t_start}")
    return u_all


def Darcy2D_FEM(coefs):
    """
    Solve the Darcy 2D equation -div(a(x,y) grad u) = f(x,y), u=1 on boundary
    using P1 finite elements on a uniform triangular mesh.

    Parameters
    ----------
    coefs : (array of size [N, s, s])
        the coefficients a(x,y) sampled at equally spaced points on the domain Ω
        (including the boundary) for each of the N inputs

    Returns
    -------
    U_all : (array of size [N, s, s])
        FEM solution values at grid points (same grid as coefs)
    """

    n, s, _ = coefs.shape
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    node_id = lambda i, j: i * s + j

    # Build mesh connectivity (2 triangles per square)
    elements = []
    for i in range(s - 1):
        for j in range(s - 1):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    n_nodes = nodes.shape[0]
    boundary = np.where((np.isclose(nodes[:, 0], 0)) | (np.isclose(nodes[:, 0], 1)) |
                        (np.isclose(nodes[:, 1], 0)) | (np.isclose(nodes[:, 1], 1)))[0]
    interior = np.setdiff1d(np.arange(n_nodes), boundary)

    f = lambda x, y: 1.0  # RHS = 1, same as in your code

    U_all = np.zeros((n, s, s))
    t_start = time.time()

    for k in range(n):
        coef_grid = coefs[k]

        # coefficient interpolator (nearest-neighbor for speed)
        def coef_func(xq, yq):
            ix = min(s - 1, max(0, int(round(xq * (s - 1)))))
            iy = min(s - 1, max(0, int(round(yq * (s - 1)))))
            return coef_grid[ix, iy]

        A = sparse.lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # Assembly
        for tri in elements:
            pts = nodes[tri]
            x0, y0 = pts[0];
            x1, y1 = pts[1];
            x2, y2 = pts[2]
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            # Gradients of shape functions
            C = np.array([
                [x1 * y2 - x2 * y1, y1 - y2, x2 - x1],
                [x2 * y0 - x0 * y2, y2 - y0, x0 - x2],
                [x0 * y1 - x1 * y0, y0 - y1, x1 - x0]
            ]) / (2 * area)
            grads = C[:, 1:]

            centroid = pts.mean(axis=0)
            a_val = coef_func(centroid[0], centroid[1])
            f_val = f(centroid[0], centroid[1])

            Ke = a_val * area * (grads @ grads.T)
            Fe = f_val * area / 3 * np.ones(3)

            for i_local, i_global in enumerate(tri):
                b[i_global] += Fe[i_local]
                for j_local, j_global in enumerate(tri):
                    A[i_global, j_global] += Ke[i_local, j_local]

        # Apply Dirichlet BCs
        U = np.zeros(n_nodes)
        U[boundary] = 0.0
        Aii = A[interior, :][:, interior].tocsr()
        bi = b[interior] - A[interior, :][:, boundary] @ U[boundary]

        U[interior] = spsolve(Aii, bi)

        U_all[k] = U.reshape((s, s))

    print(f"Elapsed time for FEM solver = {time.time() - t_start:.3f}s")
    return U_all


def Darcy2D_itsolve(coefs, u0=None, rtol=1e-5):
    r"""
    Solves the 2D Darcy equation:
        -∇ · (a(x,y) ∇u(x,y)) = f(x,y),  (x,y) ∈ Ω
        with boundary conditions u(x,y) = 1,  (x,y) ∈ ∂Ω

    Parameters
    ----------
    coefs : array of shape [N, s, s]
        Coefficients a(x,y) sampled at equally spaced points on Ω (including boundary).
    u0 : array, optional
        Initial guess for the interior values. Shape [N, (s-2), (s-2)] or padded [N, s, s].
    rtol : float
        Relative tolerance for the CG solver.

    Returns
    -------
    u_all : array of shape [N, s, s]
        Solutions at equally spaced points in Ω (including boundary).
    total_time : float
        Total wall-clock time of assembly + solve.
    time_residual_hist : list of arrays
        Each entry is an array of (time, residual norm) tuples for one sample.
    """
    t_start = time.time()
    s = coefs.shape[1]
    n = coefs.shape[0]
    assert coefs.shape[1] == coefs.shape[2], \
        "The second and third dimensions of coefs must match."

    # prepare initial guess for interior unknowns
    if u0 is not None:
        u0 = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))

    f = np.ones((s, s))
    u_all = np.zeros_like(coefs)
    time_residual_hist = []

    for i in range(n):
        coef = coefs[i, :, :]
        a, _f = solve_gwf(coef, f)  # build operator and RHS

        b_i = _f.ravel()
        tr_hist = []
        t0 = time.perf_counter()

        def cb(xk):
            r_norm = np.linalg.norm(b_i - a @ xk) / np.linalg.norm(b_i)
            tr_hist.append((time.perf_counter() - t0, r_norm))

        if u0 is None:
            u_interior, info = cg(a.tocsr(), b_i, rtol=rtol, callback=cb)
        else:
            u_interior, info = cg(a.tocsr(), b_i, x0=u0[i], rtol=rtol, callback=cb)

        time_residual_hist.append(np.asarray(tr_hist))

        if info != 0:
            print(f"Warning: CG did not converge for sample {i}")

        # pad interior back to full domain
        u_interior = u_interior.reshape(s - 2, s - 2)
        u_all[i] = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')

    total_time = time.time() - t_start
    return u_all, total_time, time_residual_hist


def Darcy2D_FEM_itsolve(coefs, u0=None, rtol=1e-5):
    """
    FEM iterative solver (CG) for -div(a grad u) = f on [0,1]^2 with u=0 on boundary.
    coefs: (N, s, s) sampled including boundary.
    Returns (u_all, total_time, time_residual_hist).
    """
    t_start = time.time()
    n, s, _ = coefs.shape
    assert s == coefs.shape[2]

    # prepare u0 interior if provided
    u0_interior = None
    if u0 is not None:
        if u0.shape == (n, s, s):
            u0_interior = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, s - 2, s - 2):
            u0_interior = u0.reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, (s - 2) * (s - 2)):
            u0_interior = u0
        else:
            raise ValueError("u0 shape not recognized")

    # mesh
    x = np.linspace(0, 1, s); y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    node_id = lambda i, j: i * s + j

    elements = []
    for i in range(s - 1):
        for j in range(s - 1):
            n0 = node_id(i, j); n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1); n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3]); elements.append([n0, n3, n2])
    elements = np.array(elements)

    n_nodes = nodes.shape[0]
    boundary = np.where((np.isclose(nodes[:, 0], 0)) |
                        (np.isclose(nodes[:, 0], 1)) |
                        (np.isclose(nodes[:, 1], 0)) |
                        (np.isclose(nodes[:, 1], 1)))[0]
    interior = np.setdiff1d(np.arange(n_nodes), boundary)

    f_fun = lambda x, y: 1.0

    u_all = np.zeros((n, s, s))
    time_residual_hist = []

    def make_nearest_coef_func(coef_grid):
        def coef_func(xq, yq):
            ix = min(s - 1, max(0, int(round(xq * (s - 1)))))
            iy = min(s - 1, max(0, int(round(yq * (s - 1)))))
            return coef_grid[ix, iy]
        return coef_func

    for k in range(n):
        coef_grid = coefs[k]
        coef_func = make_nearest_coef_func(coef_grid)

        A = sparse.lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # assemble
        for tri in elements:
            pts = nodes[tri]
            x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            C = np.array([
                [x1 * y2 - x2 * y1, y1 - y2, x2 - x1],
                [x2 * y0 - x0 * y2, y2 - y0, x0 - x2],
                [x0 * y1 - x1 * y0, y0 - y1, x1 - x0]
            ]) / (2 * area)
            grads = C[:, 1:]

            centroid = pts.mean(axis=0)
            a_val = coef_func(centroid[0], centroid[1])
            f_val = f_fun(centroid[0], centroid[1])

            Ke = a_val * area * (grads @ grads.T)
            Fe = f_val * area / 3.0 * np.ones(3)

            for il, ig in enumerate(tri):
                b[ig] += Fe[il]
                for jl, jg in enumerate(tri):
                    A[ig, jg] += Ke[il, jl]

        # homogeneous Dirichlet: u_boundary = 0  => bi = b_i (no boundary term)
        A = A.tocsr()
        Aii = A[interior, :][:, interior]
        bi = b[interior].copy()

        # CG with residual history
        tr_hist = []
        t0 = time.perf_counter()
        bnorm = np.linalg.norm(bi)
        if bnorm == 0: bnorm = 1.0

        def cb(xk):
            tr_hist.append((time.perf_counter() - t0, np.linalg.norm(bi - Aii.dot(xk)) / bnorm))

        x0_vec = u0_interior[k] if u0_interior is not None else None
        x_sol, info = cg(Aii, bi, x0=x0_vec, rtol=rtol, callback=cb)

        time_residual_hist.append(np.asarray(tr_hist))
        if info != 0:
            print(f"Warning: CG did not converge for sample {k} (info={info})")

        U = np.zeros(n_nodes)
        U[interior] = x_sol
        U[boundary] = 0.0
        u_all[k] = U.reshape((s, s))

    total_time = time.time() - t_start
    return u_all, total_time, time_residual_hist


def Darcy2D_pyagm_solve(coefs, u0=None, rtol=1e-5):
    r"""
    Solves the 2D Darcy equation:
        -∇ · (a(x,y) ∇u(x,y)) = f(x,y),  (x,y) ∈ Ω
        with boundary conditions u(x,y) = 1,  (x,y) ∈ ∂Ω

    Parameters
    ----------
    coefs : array of shape [N, s, s]
        Coefficients a(x,y) sampled at equally spaced points on Ω (including boundary).
    u0 : array, optional
        Initial guess for the interior values. Shape [N, (s-2), (s-2)] or padded [N, s, s].
    rtol : float
        Relative tolerance for the AMG solver.

    Returns
    -------
    u_all : array of shape [N, s, s]
        Solutions at equally spaced points in Ω (including boundary).
    total_time : float
        Total wall-clock time of solves (not counting AMG setup cost).
    time_residual_hist : list of arrays
        Each entry is an array of (time, residual norm) tuples for one sample.
    """
    n = coefs.shape[0]
    s = coefs.shape[1]
    assert s == coefs.shape[2], "The second and third dimensions of coefs must match."

    # prepare initial guess for interior unknowns
    if u0 is not None:
        u0 = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))

    f = np.ones((s, s))
    u_all = np.zeros_like(coefs)
    time_residual_hist = []

    t_start = time.time()

    for i in range(n):
        coef = coefs[i, :, :]
        a, _f = solve_gwf(coef, f)
        a = a.tocsr()

        # AMG setup (done per-sample since operator depends on coef)
        ml = pyamg.ruge_stuben_solver(a)

        b_i = _f.ravel()
        norm_b = np.linalg.norm(b_i)
        x0 = None if u0 is None else u0[i]

        # Track residuals during solve
        residuals = []
        t0 = time.perf_counter()
        u_interior = ml.solve(b_i,
                              x0=x0,
                              tol=rtol,
                              residuals=residuals)
        elapsed = time.perf_counter() - t0

        # Convert to (time, residual_norm) pairs
        n_cycles = len(residuals)
        times = np.linspace(0, elapsed, n_cycles)
        rel_residuals = np.array(residuals) / norm_b
        time_residual_hist.append(np.column_stack((times, rel_residuals)))

        # reshape interior and pad with BC
        u_interior = u_interior.reshape(s - 2, s - 2)
        u_all[i] = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')

    total_time = time.time() - t_start
    return u_all, total_time, time_residual_hist


def Darcy2D_FEM_pyamg_solve(coefs, u0=None, rtol=1e-5):
    """
    FEM + pyamg solver for -div(a grad u) = f on [0,1]^2 with u=0 on boundary.
    Mirrors the behavior/API of your Darcy2D_pyagm_solve but using FEM assembly.

    Parameters
    ----------
    coefs : array [N, s, s]
        coefficient fields sampled on grid (including boundary)
    u0 : array, optional
        initial guess. Shape [N, s, s] (padded) or [N, s-2, s-2] (interior only)
    rtol : float
        pyamg solver relative tolerance

    Returns
    -------
    u_all : array [N, s, s]
        solutions including boundary (boundary = 0)
    total_time : float
        wall-clock time (assembly + AMG setup + solves)
    time_residual_hist : list of arrays
        for each sample, array of shape (m,2) with (time_since_start, rel_res_norm)
    """
    try:
        import pyamg
    except ImportError as e:
        raise ImportError("pyamg is required for this function. Install it (pip install pyamg).") from e

    t_start = time.time()
    n, s, _ = coefs.shape
    assert s == coefs.shape[2], "Second and third dims of coefs must match."

    # prepare u0 interior if provided
    u0_interior = None
    if u0 is not None:
        if u0.shape == (n, s, s):
            u0_interior = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, s - 2, s - 2):
            u0_interior = u0.reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, (s - 2) * (s - 2)):
            u0_interior = u0
        else:
            raise ValueError("u0 shape not recognized. Expect (N,s,s) or (N,s-2,s-2) or (N,(s-2)*(s-2)).")

    # Build mesh once
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    node_id = lambda i, j: i * s + j

    elements = []
    for i in range(s - 1):
        for j in range(s - 1):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    n_nodes = nodes.shape[0]
    boundary = np.where((np.isclose(nodes[:, 0], 0)) |
                        (np.isclose(nodes[:, 0], 1)) |
                        (np.isclose(nodes[:, 1], 0)) |
                        (np.isclose(nodes[:, 1], 1)))[0]
    interior = np.setdiff1d(np.arange(n_nodes), boundary)

    f_fun = lambda x, y: 1.0

    u_all = np.zeros((n, s, s))
    time_residual_hist = []

    # nearest-neighbor coefficient lookup (keeps parity with earlier FEM assembly)
    def make_nearest_coef_func(coef_grid):
        def coef_func(xq, yq):
            ix = min(s - 1, max(0, int(round(xq * (s - 1)))))
            iy = min(s - 1, max(0, int(round(yq * (s - 1)))))
            return coef_grid[ix, iy]
        return coef_func

    for k in range(n):
        coef_grid = coefs[k]
        coef_func = make_nearest_coef_func(coef_grid)

        # Assemble global stiffness A and load b (full)
        A = sparse.lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        for tri in elements:
            pts = nodes[tri]
            x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            # Gradients of P1 shape functions
            C = np.array([
                [x1 * y2 - x2 * y1, y1 - y2, x2 - x1],
                [x2 * y0 - x0 * y2, y2 - y0, x0 - x2],
                [x0 * y1 - x1 * y0, y0 - y1, x1 - x0]
            ]) / (2 * area)
            grads = C[:, 1:]

            centroid = pts.mean(axis=0)
            a_val = coef_func(centroid[0], centroid[1])
            f_val = f_fun(centroid[0], centroid[1])

            Ke = a_val * area * (grads @ grads.T)
            Fe = f_val * area / 3.0 * np.ones(3)

            for i_local, i_global in enumerate(tri):
                b[i_global] += Fe[i_local]
                for j_local, j_global in enumerate(tri):
                    A[i_global, j_global] += Ke[i_local, j_local]

        # homogeneous Dirichlet: u = 0 on boundary -> bi = b_i (no boundary term)
        A = A.tocsr()
        Aii = A[interior, :][:, interior].copy()
        bi = b[interior].copy()

        # AMG setup
        ml = pyamg.ruge_stuben_solver(Aii)

        # solve with AMG (collect residual history)
        residuals = []
        t0 = time.perf_counter()
        x0 = None if u0_interior is None else u0_interior[k]
        u_interior = ml.solve(bi, x0=x0, tol=rtol, residuals=residuals)
        elapsed = time.perf_counter() - t0

        # convert residuals to (time, relative residual) pairs
        norm_b = np.linalg.norm(bi)
        if norm_b == 0:
            norm_b = 1.0
        n_cycles = len(residuals)
        if n_cycles == 0:
            # pyamg may return no recorded residuals; still return an empty array
            time_residual_hist.append(np.zeros((0, 2)))
        else:
            times = np.linspace(0.0, elapsed, n_cycles)
            rel_residuals = np.array(residuals) / norm_b
            time_residual_hist.append(np.column_stack((times, rel_residuals)))

        # reshape interior and pad with BC (zeros)
        u_interior = u_interior.reshape(s - 2, s - 2)
        u_all[k] = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')

    total_time = time.time() - t_start
    return u_all, total_time, time_residual_hist


def Darcy2D_petsc_solve(coefs, u0=None, rtol=1e-5):
    r"""
    Solves the 2D Darcy equation:
        -∇ · (a(x,y) ∇u(x,y)) = f(x,y),  (x,y) ∈ Ω
        with boundary conditions u(x,y) = 1,  (x,y) ∈ ∂Ω

    Parameters
    ----------
    coefs : array of shape [N, s, s]
        Coefficients a(x,y) sampled at equally spaced points on Ω (including boundary).
    u0 : array, optional
        Initial guess for the interior values. Shape [N, (s-2), (s-2)] or padded [N, s, s].
    rtol : float
        Relative tolerance for the PETSc KSP solver.

    Returns
    -------
    u_all : array of shape [N, s, s]
        Solutions at equally spaced points in Ω (including boundary).
    total_time : float
        Total wall-clock time for all solves (assembly + solve).
    time_residual_hist : list of arrays
        Each entry is an array of (time, relative residual norm) tuples for one sample.
    """
    n = coefs.shape[0]
    s = coefs.shape[1]
    assert s == coefs.shape[2], "The second and third dimensions of coefs must match."

    # Prepare initial guess for interior unknowns
    if u0 is not None:
        u0 = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))

    f = np.ones((s, s))
    u_all = np.zeros_like(coefs)
    time_residual_hist = []

    total_time_start = time.time()

    for i in range(n):
        coef = coefs[i, :, :]
        a, _f = solve_gwf(coef, f)

        # Convert scipy sparse matrix to PETSc
        a_petsc = PETSc.Mat().createAIJ(size=((s - 2) * (s - 2), (s - 2) * (s - 2)),
                                        csr=(a.indptr, a.indices, a.data))

        b_np = np.copy(_f.ravel())
        b_i = PETSc.Vec().createWithArray(b_np, comm=PETSc.COMM_SELF)
        norm_b = np.linalg.norm(b_np)

        if u0 is None:
            x = PETSc.Vec().createSeq((s - 2) * (s - 2), comm=PETSc.COMM_SELF)
        else:
            x = PETSc.Vec().createWithArray(np.copy(u0[i]), comm=PETSc.COMM_SELF)

        # Setup KSP solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(a_petsc)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ICC)
        ksp.setTolerances(rtol=rtol)
        ksp.setNormType(PETSc.KSP.NormType.NORM_NATURAL)

        # Track true relative residuals
        tr_hist = []
        t0 = time.perf_counter()

        def monitor(ksp, its, rnorm_unused):
            # Compute true residual r = b - A*x
            r_vec = b_i.copy()
            a_petsc.mult(x, r_vec)  # r_vec = A*x
            r_vec.scale(-1)
            r_vec.axpy(1.0, b_i)  # r_vec = b - A*x
            rel_rnorm = r_vec.norm() / norm_b
            tr_hist.append((time.perf_counter() - t0, rel_rnorm))

        ksp.setMonitor(monitor)

        if u0 is not None:
            ksp.setInitialGuessNonzero(True)

        ksp.solve(b_i, x)
        u_interior = x.getArray().reshape(s - 2, s - 2)
        u_all[i] = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')

        time_residual_hist.append(np.asarray(tr_hist))

        if ksp.getConvergedReason() <= 0:
            print(f"Warning: PETSc KSP did not converge for sample {i}")

    total_time = time.time() - total_time_start
    return u_all, total_time, time_residual_hist


def Darcy2D_FEM_petsc_solve(coefs, u0=None, rtol=1e-5):
    """
    FEM + PETSc solver for -div(a grad u) = f on [0,1]^2 with u=0 on boundary.

    Parameters
    ----------
    coefs : array [N, s, s]
        Coefficient fields sampled on grid (including boundary).
    u0 : array, optional
        Initial guess. Shape [N,s,s] (padded) or [N,s-2,s-2] (interior only).
    rtol : float
        Relative tolerance for PETSc KSP solver.

    Returns
    -------
    u_all : array [N,s,s]
        Solutions including boundary (u=0 on ∂Ω).
    total_time : float
        Wall-clock time (assembly + all PETSc solves).
    time_residual_hist : list of arrays
        Each entry: shape (m,2), columns = (time_since_start, rel_residual_norm).
    """
    n, s, _ = coefs.shape
    assert s == coefs.shape[2], "coefs must be square grids"

    # Prepare initial guesses (interior only)
    u0_interior = None
    if u0 is not None:
        if u0.shape == (n, s, s):
            u0_interior = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, s - 2, s - 2):
            u0_interior = u0.reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, (s - 2) * (s - 2)):
            u0_interior = u0
        else:
            raise ValueError("u0 shape not recognized. Expect (N,s,s) or (N,s-2,s-2) or (N,(s-2)*(s-2)).")

    # Mesh
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    node_id = lambda i, j: i * s + j

    # Connectivity
    elements = []
    for i in range(s - 1):
        for j in range(s - 1):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    n_nodes = nodes.shape[0]
    boundary = np.where((np.isclose(nodes[:, 0], 0)) |
                        (np.isclose(nodes[:, 0], 1)) |
                        (np.isclose(nodes[:, 1], 0)) |
                        (np.isclose(nodes[:, 1], 1)))[0]
    interior = np.setdiff1d(np.arange(n_nodes), boundary)

    f_fun = lambda x, y: 1.0

    def make_nearest_coef_func(coef_grid):
        def coef_func(xq, yq):
            ix = min(s - 1, max(0, int(round(xq * (s - 1)))))
            iy = min(s - 1, max(0, int(round(yq * (s - 1)))))
            return coef_grid[ix, iy]
        return coef_func

    u_all = np.zeros((n, s, s))
    time_residual_hist = []

    total_time_start = time.time()

    for k in range(n):
        coef_grid = coefs[k]
        coef_func = make_nearest_coef_func(coef_grid)

        # Assemble stiffness A and load b
        A = sparse.lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        for tri in elements:
            pts = nodes[tri]
            x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            # Gradients of P1 shape functions
            C = np.array([
                [x1 * y2 - x2 * y1, y1 - y2, x2 - x1],
                [x2 * y0 - x0 * y2, y2 - y0, x0 - x2],
                [x0 * y1 - x1 * y0, y0 - y1, x1 - x0]
            ]) / (2 * area)
            grads = C[:, 1:]

            centroid = pts.mean(axis=0)
            a_val = coef_func(centroid[0], centroid[1])
            f_val = f_fun(centroid[0], centroid[1])

            Ke = a_val * area * (grads @ grads.T)
            Fe = f_val * area / 3.0 * np.ones(3)

            for i_local, i_global in enumerate(tri):
                b[i_global] += Fe[i_local]
                for j_local, j_global in enumerate(tri):
                    A[i_global, j_global] += Ke[i_local, j_local]

        # Dirichlet BC: u = 0 on boundary
        A = A.tocsr()
        Aii = A[interior, :][:, interior].copy()
        bi = b[interior]  # no subtraction needed, since u_boundary = 0

        # Convert to PETSc
        A_petsc = PETSc.Mat().createAIJ(size=Aii.shape,
                                        csr=(Aii.indptr, Aii.indices, Aii.data))
        b_vec = PETSc.Vec().createWithArray(bi, comm=PETSc.COMM_SELF)
        norm_b = np.linalg.norm(bi)

        if u0_interior is None:
            x_vec = PETSc.Vec().createSeq(len(interior), comm=PETSc.COMM_SELF)
        else:
            x_vec = PETSc.Vec().createWithArray(np.copy(u0_interior[k]), comm=PETSc.COMM_SELF)

        # PETSc KSP setup
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ICC)
        ksp.setTolerances(rtol=rtol)
        ksp.setNormType(PETSc.KSP.NormType.NORM_NATURAL)

        # Monitor true residuals
        tr_hist = []
        t0 = time.perf_counter()

        def monitor(ksp_obj, its, rnorm_unused):
            r_vec = b_vec.duplicate()
            A_petsc.mult(x_vec, r_vec)   # r = A*x
            r_vec.scale(-1)
            r_vec.axpy(1.0, b_vec)       # r = b - A*x
            rel_rnorm = r_vec.norm() / norm_b
            tr_hist.append((time.perf_counter() - t0, rel_rnorm))

        ksp.setMonitor(monitor)
        if u0_interior is not None:
            ksp.setInitialGuessNonzero(True)

        # Solve
        ksp.solve(b_vec, x_vec)
        u_interior = x_vec.getArray()

        # Build full solution
        U = np.zeros(n_nodes)
        U[boundary] = 0.0
        U[interior] = u_interior
        u_all[k] = U.reshape((s, s))

        time_residual_hist.append(np.asarray(tr_hist))

        if ksp.getConvergedReason() <= 0:
            print(f"Warning: PETSc KSP did not converge for sample {k}")

    total_time = time.time() - total_time_start
    return u_all, total_time, time_residual_hist


def Darcy2D_FEM_pc_solve(coefs, u0=None, rtol=1e-5, pc_type="icc"):
    """
    FEM + PETSc solver for -div(a grad u) = f on [0,1]^2 with u=0 on boundary.

    New argument:
      pc_type : str
        Preconditioner choice (case-insensitive). Supported options:
          'jacobi'   -> PETSc PC JACOBI
          'ssor'     -> PETSc PC SOR  (use symmetric SOR via options below)
          'icc'      -> PETSc PC ICC  (Incomplete Cholesky)
          'ilu'      -> PETSc PC ILU  (Incomplete LU)
          'amg'      -> PETSc PC GAMG (Algebraic multigrid)
          'gmg'      -> PETSc PC MG   (Geometric/Multigrid)
          'deflation'-> PETSc PC DEFLATION
          'none'     -> PETSc PC NONE (no preconditioner)
        Default: 'icc'

    Notes:
      - This function uses PETSc/petsc4py. For more advanced tuning (levels, drop tol,
        hypre options, deflation spaces, etc.) set PETSc options via the options
        database (e.g. PETSc.Options()) or extend the code where noted.
    """
    # local import so function is self-contained
    from petsc4py import PETSc
    import numpy as np
    import time
    from scipy import sparse
    import numpy.linalg as la

    n, s, _ = coefs.shape
    assert s == coefs.shape[2], "coefs must be square grids"

    # Prepare initial guesses (interior only)
    u0_interior = None
    if u0 is not None:
        if u0.shape == (n, s, s):
            u0_interior = u0[:, 1:-1, 1:-1].reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, s - 2, s - 2):
            u0_interior = u0.reshape(n, (s - 2) * (s - 2))
        elif u0.shape == (n, (s - 2) * (s - 2)):
            u0_interior = u0
        else:
            raise ValueError("u0 shape not recognized. Expect (N,s,s) or (N,s-2,s-2) or (N,(s-2)*(s-2)).")

    # Mesh
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    node_id = lambda i, j: i * s + j

    # Connectivity
    elements = []
    for i in range(s - 1):
        for j in range(s - 1):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    n_nodes = nodes.shape[0]
    boundary = np.where((np.isclose(nodes[:, 0], 0)) |
                        (np.isclose(nodes[:, 0], 1)) |
                        (np.isclose(nodes[:, 1], 0)) |
                        (np.isclose(nodes[:, 1], 1)))[0]
    interior = np.setdiff1d(np.arange(n_nodes), boundary)

    f_fun = lambda x, y: 1.0

    def make_nearest_coef_func(coef_grid):
        def coef_func(xq, yq):
            ix = min(s - 1, max(0, int(round(xq * (s - 1)))))
            iy = min(s - 1, max(0, int(round(yq * (s - 1)))))
            return coef_grid[ix, iy]
        return coef_func

    u_all = np.zeros((n, s, s))
    time_residual_hist = []

    total_time_start = time.time()

    # normalize user choice
    pc_choice = (pc_type or "icc").strip().lower()

    # mapping input names -> PETSc PC.Type attributes
    pc_map = {
        "jacobi": PETSc.PC.Type.JACOBI,
        "ssor": PETSc.PC.Type.SOR,        # use PCSOR; symmetric variant can be set via options
        "icc": PETSc.PC.Type.ICC,         # incomplete cholesky
        "ilu": PETSc.PC.Type.ILU,         # incomplete LU
        "amg": PETSc.PC.Type.GAMG,        # algebraic multigrid (GAMG)
        "gmg": PETSc.PC.Type.MG,          # multigrid (MG)
        "deflation": PETSc.PC.Type.DEFLATION,
        "none": PETSc.PC.Type.NONE
    }

    for k in range(n):
        coef_grid = coefs[k]
        coef_func = make_nearest_coef_func(coef_grid)

        # Assemble stiffness A and load b
        A = sparse.lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        for tri in elements:
            pts = nodes[tri]
            x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            # Gradients of P1 shape functions
            C = np.array([
                [x1 * y2 - x2 * y1, y1 - y2, x2 - x1],
                [x2 * y0 - x0 * y2, y2 - y0, x0 - x2],
                [x0 * y1 - x1 * y0, y0 - y1, x1 - x0]
            ]) / (2 * area)
            grads = C[:, 1:]

            centroid = pts.mean(axis=0)
            a_val = coef_func(centroid[0], centroid[1])
            f_val = f_fun(centroid[0], centroid[1])

            Ke = a_val * area * (grads @ grads.T)
            Fe = f_val * area / 3.0 * np.ones(3)

            for i_local, i_global in enumerate(tri):
                b[i_global] += Fe[i_local]
                for j_local, j_global in enumerate(tri):
                    A[i_global, j_global] += Ke[i_local, j_local]

        # Dirichlet BC: u = 0 on boundary
        A = A.tocsr()
        Aii = A[interior, :][:, interior].copy()
        bi = b[interior]  # no subtraction needed, since u_boundary = 0

        # Convert to PETSc
        A_petsc = PETSc.Mat().createAIJ(size=Aii.shape,
                                        csr=(Aii.indptr, Aii.indices, Aii.data))
        b_vec = PETSc.Vec().createWithArray(bi, comm=PETSc.COMM_SELF)
        norm_b = np.linalg.norm(bi)

        if u0_interior is None:
            x_vec = PETSc.Vec().createSeq(len(interior), comm=PETSc.COMM_SELF)
        else:
            x_vec = PETSc.Vec().createWithArray(np.copy(u0_interior[k]), comm=PETSc.COMM_SELF)

        # PETSc KSP setup (CG for SPD)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()

        # choose PC based on user choice -> map to PETSc PC.Type
        if pc_choice in pc_map:
            try:
                pc.setType(pc_map[pc_choice])
            except Exception as e:
                # if requested PC not available on this build, warn and fall back to NONE
                print(f"Warning: requested PETSc PC type '{pc_choice}' not available ({e}). Falling back to NONE.")
                pc.setType(PETSc.PC.Type.NONE)
        else:
            # unknown string: warn and use ICC as sensible default for SPD problems
            print(f"Warning: unknown pc_type='{pc_type}'. Using ICC (Incomplete Cholesky) by default.")
            pc.setType(PETSc.PC.Type.ICC)

        # Small extra settings for some PCs (optional / conservative defaults)
        if pc_choice == "ssor":
            # enable symmetric variant of SOR (SSOR)
            # You can tune -pc_sor_omega (relaxation), -pc_sor_iterations, etc via PETSc options.
            try:
                PETSc.Options().setValue("-pc_sor_symmetric", None)
            except Exception:
                # options may already be set or not supported; ignore
                pass

        # Example: if using GAMG you can tune via PETSc options (left commented for now)
        # if pc_choice == "amg":
        #     PETSc.Options().setValue("-pc_gamg_type", "agg")  # example option

        # set tolerances and norm
        ksp.setTolerances(rtol=rtol)
        ksp.setNormType(PETSc.KSP.NormType.NORM_NATURAL)

        # Monitor true residuals
        tr_hist = []
        t0 = time.perf_counter()

        def monitor(ksp_obj, its, rnorm_unused):
            # compute true residual r = b - A*x and store relative norm
            r_vec = b_vec.duplicate()
            A_petsc.mult(x_vec, r_vec)   # r = A*x
            r_vec.scale(-1)
            r_vec.axpy(1.0, b_vec)       # r = b - A*x
            rel_rnorm = r_vec.norm() / (norm_b if norm_b != 0.0 else 1.0)
            tr_hist.append((time.perf_counter() - t0, rel_rnorm))

        ksp.setMonitor(monitor)
        if u0_interior is not None:
            ksp.setInitialGuessNonzero(True)

        # Solve
        ksp.solve(b_vec, x_vec)
        u_interior = x_vec.getArray()

        # Build full solution
        U = np.zeros(n_nodes)
        U[boundary] = 0.0
        U[interior] = u_interior
        u_all[k] = U.reshape((s, s))

        time_residual_hist.append(np.asarray(tr_hist))

        if ksp.getConvergedReason() <= 0:
            print(f"Warning: PETSc KSP did not converge for sample {k} (reason={ksp.getConvergedReason()}).")

    total_time = time.time() - total_time_start
    return u_all, total_time, time_residual_hist


def Beam1D_solve(F_train):
    r"""
    Solves the deflection of a simply supported beam under a variable load for multiple samples.

    Parameters:
    - F_train: torch.Tensor or np.ndarray of shape (num_samples, 256, 1)
               A tensor containing load values for each sample, with each sample having 256 points.

    Returns:
    - y_samples: np.ndarray of shape (num_samples, 256)
                 Array containing the deflection values for each sample.
    """
    # Define beam parameters
    L = 1  # Total length of the beam (m)
    b = h = 1
    E = 1e1  # Young's modulus (Pa)
    I = 1 / 12 * b * h ** 3  # Moment of inertia (m^4)

    # Number of segments (nodes - 1)
    n = F_train.shape[1]  # n=64 for VINO Bram, n=256 for FNO bram
    h = L / (n - 1)  # Segment length

    # Convert F_train to a numpy array if it's a tensor and adjust its shape
    if isinstance(F_train, torch.Tensor):
        F_train_np = F_train.squeeze(-1).numpy()  # Convert to numpy and remove last dimension
    else:
        F_train_np = F_train.squeeze(-1)  # If it's already an np.ndarray

    # Initialize an array to store deflection results for each sample
    y_samples = np.zeros((F_train_np.shape[0], n))

    # Loop over each sample in F_train
    for sample_idx in range(F_train_np.shape[0]):
        # Use the current sample's load values
        W = F_train_np[sample_idx]  # This is a 256-point load vector

        # Matrix for the moments (Mi-1 - 2Mi + Mi+1)
        A = np.zeros((n, n))
        b_moments = np.zeros(n)

        # Apply FDM for moments equation: M_{i-1} - 2M_i + M_{i+1} = -h^2 * W_n
        for i in range(1, n - 1):  # Loop should run from 1 to n-2
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1
            b_moments[i] = -h ** 2 * W[i]

        # Boundary conditions: M_0 = M_n = 0 for simply supported beam
        A[0, 0] = 1
        A[-1, -1] = 1

        # Solve for moments M
        M = np.linalg.solve(A, b_moments)

        # Now, apply FDM to deflection using the moments
        # Deflection equation: y_{i-1} - 2*y_i + y_{i+1} = h^2 * M_i / (E * I)
        A_deflection = np.zeros((n, n))
        b_deflection = np.zeros(n)

        for i in range(1, n - 1):  # Loop should run from 1 to n-2
            A_deflection[i, i - 1] = 1
            A_deflection[i, i] = -2
            A_deflection[i, i + 1] = 1
            b_deflection[i] = h ** 2 * M[i] / (E * I)

        # Boundary conditions: y_0 = y_n = 0 for simply supported beam
        A_deflection[0, 0] = 1
        A_deflection[-1, -1] = 1

        # Solve for deflections y for the current sample
        y = np.linalg.solve(A_deflection, b_deflection)
        y_samples[sample_idx] = y  # Store the result in the array

    return y_samples

# import matplotlib.pyplot as plt

# # Testing
# num_x = 32
# num_y = 32

# x = np.linspace(0, 1, num_x)
# y = np.linspace(0, 1, num_y)

# [X, Y] = np.meshgrid(x,y)
# #F = -2*(Y**2-Y + X**2 - X)
# F = -2*np.sin(2*np.pi*Y)+4*np.pi**2*np.sin(2*np.pi*Y)*(X**2-X)
# coef = np.ones((num_y, num_x))
# coef[:num_x//2, num_x//2:] = 1.

# U_comp = solve_gwf(coef, F)
# #U_exact = (X**2-X)*(Y**2-Y)
# U_exact = X*(X-1)*np.sin(2*np.pi*Y)
# plt.contourf(X, Y, U_comp)
# plt.colorbar()
# plt.title('Computed U')
# plt.show()

# U_error = U_exact - U_comp
# plt.contourf(X, Y, U_error)
# plt.colorbar()
# plt.title('Error U_exact - U_comp')
# plt.show()
