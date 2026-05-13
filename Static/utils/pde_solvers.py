import time
import random
import pyamg

import numpy as np
from petsc4py import PETSc
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.fftpack import idct
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve, cg
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.linalg import cholesky, solve_triangular

#import scipy.sparse as sp
#import scipy.sparse.linalg as spla

from utils.IGA.Geom_examples import Quadrilateral, PlateWHoleQuadrant
from utils.IGA.IGA import IGAMesh2D
from utils.IGA.assembly import gen_gauss_pts, stiff_elast_FGM_2D, stiff_elast_2D
from utils.IGA.boundary import boundary2D, applyBCElast2D
from utils.IGA.materials import MaterialElast2D_RandomFGM, MaterialElast2D
from utils.IGA.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from utils.IGA.postprocessing import comp_measurement_values, get_measurements_vector, get_measurement_stresses, \
    plot_fields_2D
from utils.IGA.materials import MaterialElast2D_Hole
from utils.IGA.postprocessing import get_measurement_stresses_FGM


def PlateHole_IGA_solver(model_data, GRF_data):
    """
    Solves a 2D elasticity problem on a Plate with random hole under tensions and creates a database.

    Fixed BC
    """

    def GRF(_alpha, _tau, s1, s2):
        xi = np.random.randn(s1, s2)
        k1, k2 = np.meshgrid(np.arange(s1), np.arange(s2), indexing='ij')
        coef = _tau ** (_alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + _tau ** 2) ** (-_alpha / 2)
        l = np.sqrt(s1 * s2) * coef * xi
        l[0, 0] = 0
        u = idct(idct(l, axis=0, norm='ortho'), axis=1, norm='ortho')
        return u

    def generate_inputs(s1, s2, _alpha, _tau):
        pad_n1 = s1 // 5
        pad_n2 = s2 // 5
        sensor_pts1 = np.linspace(0, 1, s1 - 2 * pad_n1 + 1, endpoint=False)[1:]
        sensor_pts2 = np.linspace(0, 1, s2 - 2 * pad_n2 + 1, endpoint=False)[1:]
        x_new_grid, y_new_grid = np.meshgrid(sensor_pts1, sensor_pts2, indexing='ij')

        in_data = GRF(_alpha, _tau, s1 - 2 * pad_n1, s2 - 2 * pad_n2)
        #in_data = in_data * np.sin(x_new_grid * y_new_grid * (1 - x_new_grid) * (1 - y_new_grid))
        in_data = in_data * x_new_grid * y_new_grid * (1 - x_new_grid) * (1 - y_new_grid)
        in_data = in_data + np.abs(in_data.min()) - 0.5 * in_data.std()
        in_data = np.pad(in_data, ((pad_n1, pad_n1), (pad_n2, pad_n2)), mode='constant', constant_values=1)
        in_data = np.where(in_data >= 0, 1., -1.)

        # plt.figure()
        # plt.imshow(in_data, cmap='jet', origin='lower')
        # plt.colorbar(label='Predicted u(x, y)')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Predicted Random Field (2D)')
        # plt.axis('equal')
        # plt.show()
        return in_data

    # Approximation parameters
    p = q = 3  # Polynomial degree of the approximation
    num_refinements = model_data["num_refinements"]  # Number of uniform refinements

    # Step 0: Generate the geometry
    num_pts_u = model_data["numPtsU"]
    num_pts_v = model_data["numPtsV"]
    beam_length = model_data["length"]
    beam_width = model_data["width"]
    traction = model_data["traction"]
    e_mod = model_data["E"]
    nu = model_data["nu"]

    alpha = GRF_data["alpha"]
    tau = GRF_data["tau"]

    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    grid = patch1.getUnifIntPts(num_pts_u, num_pts_v, [1, 1, 1, 1])
    patch_list = [patch1]

    # Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x, y):
        return [0., 0.]

    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    _Y = np.linspace(0, beam_width, num_pts_v)[:, None].flatten()

    def u_bound_neu(x, y, nx, ny):
        return [np.interp(y, _Y, traction), 0.]

    bound_right = boundary2D("Neumann", 0, "right", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)

    bound_all = [bound_right, bound_left]

    # Step 1: Define the material properties
    x_2d = np.reshape(grid[0], (num_pts_v, num_pts_u))
    y_2d = np.reshape(grid[1], (num_pts_v, num_pts_u))

    x_1d = np.linspace(0, beam_length, num_pts_u)
    y_1d = np.linspace(0, beam_width, num_pts_v)

    mask = generate_inputs(num_pts_v, num_pts_u, alpha, tau)
    mask = np.where(mask > 0, False, True)

    _x = np.where((0 <= x_2d) & (x_2d <= beam_length), x_2d, np.where(x_2d < 0, 0, beam_length))
    _y = np.where((0 <= y_2d) & (y_2d <= beam_width), y_2d, np.where(y_2d < 0, 0, beam_width))
    e_mod_mat = e_mod * (1e-9 * mask + 1 * (~mask))
    e_mod_fun = RGI((x_1d, y_1d), e_mod_mat.T, method='linear', bounds_error=False)

    material = MaterialElast2D_Hole(Emod=e_mod, nu=nu, vertices=vertices, elasticity_fun=e_mod_fun, plane_type="stress")

    # Step 2: Degree elevate and refine the geometry
    # t = time.time()
    for patch in patch_list:
        patch.degreeElev(p - 1, q - 1)
    # elapsed = time.time() - t
    # print("Degree elevation took ", elapsed, " seconds")

    # t = time.time()
    # Refine the mesh in the horizontal direction first two times to get square elements
    for i in range(2):
        for patch in patch_list:
            patch.refineKnotVectors(True, False)

    for i in range(num_refinements):
        for patch in patch_list:
            patch.refineKnotVectors(True, True)
    # elapsed = time.time() - t
    # print("Knot insertion took ", elapsed, " seconds")

    # t = time.time()
    mesh_list = []
    for patch in patch_list:
        mesh_list.append(IGAMesh2D(patch))
    # elapsed = time.time() - t
    # print("Mesh initialization took ", elapsed, " seconds")

    for mesh in mesh_list:
        mesh.classify_boundary()

    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)

    # Step 3. Assemble linear system
    gauss_quad_u = gen_gauss_pts(p + 1)
    gauss_quad_v = gen_gauss_pts(q + 1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    # t = time.time()
    stiff, e_modulus, coord = stiff_elast_FGM_2D(mesh_list, material, gauss_rule)
    stiff = stiff.tocsr()
    # elapsed = time.time() - t
    # print("Stiffness assembly took ", elapsed, " seconds")

    # Step 4. Apply boundary conditions
    # t = time.time()
    # Assume no volume force
    rhs = np.zeros(2 * size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)
    # elapsed = time.time() - t
    # print("Applying B.C.s took ", elapsed, " seconds")

    # Step 6. Solve the linear system
    # t = time.time()
    sol0 = spsolve(stiff, rhs)
    # elapsed = time.time() - t
    # print("Linear sparse solver took ", elapsed, " seconds")

    # Step 7a. Plot the solution in matplotlib
    # t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = num_pts_u
    num_pts_eta = num_pts_v
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = (
        comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurements_vector, num_fields))
    # elapsed = time.time() - t
    # print("Computing the displacement values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(
        num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurement_stresses_FGM, num_output_fields, material)
    # elapsed = time.time() - t
    # print("Computing the stress values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    disp2d = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    # elapsed = time.time() - t
    # print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

    field_names = ['stress_xx', 'stress_yy', 'stress_xy']
    stress = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)

    # elapsed = time.time() - t
    # print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

    # return Emod_mat_gen.reshape(numPtsV, numPtsU), disp2D, stress
    return mask, disp2d.transpose((1, 0, 2)), stress.transpose((1, 0, 2))


def PlotMesh(cells, points):
    plt.figure(figsize=(8, 6))
    for cell in cells:
        # Extract node coordinates for the cell.
        pts = points[cell, :]
        # Close the quadrilateral.
        pts = np.vstack((pts, pts[0]))
        plt.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.5)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=10)
    plt.title("Uniform Rectangular Mesh (Quad Elements) with a Circular Hole")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal')
    plt.show()


def PlateWithHole_solver(mask_matrix, solver="spsolve", u0=None, x_max=5, y_max=5, E=1e3, nu=0.3):
    """
    Finite Element Method solver for 2D elasticity given a mask matrix.
    Inputs:
        - mask_matrix: 2D boolean numpy array (True = included, False = excluded)
        - x_max, y_max: Dimensions of the rectangular domain (default 5x5)
        - E: Young's modulus
        - nu: Poisson's ratio
    Outputs:
        - node coordinates, displacements (ux, uy), stress components (sigma_xx, sigma_yy, sigma_xy)
    """
    # =============================================================================
    # Timing: Step 1 - Mesh Generation (Quad Elements with a Circular Hole)
    # =============================================================================
    t0 = time.perf_counter()

    # Generate a structured grid of nodes
    num_points_y, num_points_x = mask_matrix.shape
    x_coords = np.linspace(0, x_max, num_points_x)
    y_coords = np.linspace(0, y_max, num_points_y)
    nx, ny = num_points_x, num_points_y
    X, Y = np.meshgrid(x_coords, y_coords)
    points_all = np.column_stack((X.flatten(), Y.flatten()))

    # Generate elements (quad connectivity)
    cells = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if mask_matrix[j, i] and mask_matrix[j, i + 1] and mask_matrix[j + 1, i + 1] and mask_matrix[j + 1, i]:
                n1 = i + j * nx
                n2 = (i + 1) + j * nx
                n3 = (i + 1) + (j + 1) * nx
                n4 = i + (j + 1) * nx
                cells.append([n1, n2, n3, n4])
    cells = np.array(cells)

    # Determine which nodes are actually used.
    used_nodes = np.unique(cells.flatten())
    # Build a mapping from old node indices to new ones.
    node_map = {old: new for new, old in enumerate(used_nodes)}
    # Re-index points.
    points = points_all[used_nodes]
    # Update cell connectivity using new node numbering.
    cells = np.array([[node_map[n] for n in cell] for cell in cells])

    t1 = time.perf_counter()
    print(f"Step 1 (Mesh Generation): {t1 - t0:.4f} seconds")

    # =============================================================================
    # Timing: Step 2 - Plot the Rectangular Mesh
    # =============================================================================
    t_mesh_plot = time.perf_counter()

    # PlotMesh(cells, points)

    t2 = time.perf_counter()
    print(f"Step 2 (Mesh Plotting): {t2 - t_mesh_plot:.4f} seconds")

    # =============================================================================
    # Timing: Step 3 - Material Properties & Element Routines Setup
    # =============================================================================
    t3 = time.perf_counter()

    # Elasticity matrix
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                      [nu, 1, 0],
                                      [0, 0, (1 - nu) / 2]])

    def element_stiffness_quad(xe, ye):
        """Compute 8x8 element stiffness matrix"""
        ke = np.zeros((8, 8))
        gauss_pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        for xi in gauss_pts:
            for eta in gauss_pts:
                dN_dxi = np.array([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
                dN_deta = np.array([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
                J = np.array([[np.sum(dN_dxi * xe), np.sum(dN_dxi * ye)],
                              [np.sum(dN_deta * xe), np.sum(dN_deta * ye)]])
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2 * i] = dN_dx[i]
                    B[1, 2 * i + 1] = dN_dy[i]
                    B[2, 2 * i] = dN_dy[i]
                    B[2, 2 * i + 1] = dN_dx[i]
                ke += B.T @ D @ B * detJ
        return ke

    t3_end = time.perf_counter()
    print(f"Step 3 (Material & Element Setup): {t3_end - t3:.4f} seconds")

    # =============================================================================
    # Timing: Step 4 - Global Assembly for the Quad Mesh
    # =============================================================================
    t4 = time.perf_counter()

    num_nodes = len(points)
    num_dofs = 2 * num_nodes
    K_data, K_row, K_col = [], [], []
    f = np.zeros(num_dofs)

    for cell in cells:
        xe = points[cell, 0]
        ye = points[cell, 1]
        ke = element_stiffness_quad(xe, ye)
        dofs = [2 * n + i for n in cell for i in range(2)]
        for i in range(8):
            for j in range(8):
                K_row.append(dofs[i])
                K_col.append(dofs[j])
                K_data.append(ke[i, j])

    K = sparse.coo_matrix((K_data, (K_row, K_col)), shape=(num_dofs, num_dofs)).tocsr()

    t4_end = time.perf_counter()
    print(f"Step 4 (Global Assembly): {t4_end - t4:.4f} seconds")

    # =============================================================================
    # Timing: Step 5 - Apply Boundary Conditions
    # =============================================================================
    t5 = time.perf_counter()

    tol = 1e-3
    right_nodes = np.where(np.abs(points[:, 0] - x_max) < tol)[0]
    left_nodes = np.where(np.abs(points[:, 0]) < tol)[0]
    constrained_dofs = np.union1d(2 * left_nodes, 2 * left_nodes + 1)

    for n1, n2 in zip(right_nodes[:-1], right_nodes[1:]):
        edge_length = np.linalg.norm(points[n1] - points[n2])
        f[2 * n1] += edge_length / 2
        f[2 * n2] += edge_length / 2

    K = K.tolil()
    for dof in constrained_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        f[dof] = 0
    K = K.tocsr()

    t5_end = time.perf_counter()
    print(f"Step 5 (Boundary Conditions): {t5_end - t5:.4f} seconds")

    # =============================================================================
    # Timing: Step 6 - Solve the Linear System
    # =============================================================================
    if u0 is not None:
        _u0 = np.empty(2 * u0.shape[0], dtype=u0.dtype)
        _u0[0::2] = u0[:, 0]
        _u0[1::2] = u0[:, 1]
    else:
        _u0 = u0

    t6 = time.perf_counter()

    if solver == "spsolve":
        u = spsolve(K, f)

    elif solver == "cg":
        if u0 is None:
            u, info = cg(K, f, rtol=1e-05)
        else:
            u, info = cg(K, f, x0=_u0, rtol=1e-05)
        if info != 0:
            print(f"Warning: CG did not converge")

    elif solver == "pyagm":
        ml = pyamg.ruge_stuben_solver(K)
        if u0 is None:
            u = ml.solve(f, tol=1e-5)
        else:
            u = ml.solve(f, x0=_u0, tol=1e-5)

    elif solver == "petsc":
        K = PETSc.Mat().createAIJ(size=(2 * used_nodes.shape[0], 2 * used_nodes.shape[0]),
                                  csr=(K.indptr, K.indices, K.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(K)
        ksp.setType(PETSc.KSP.Type.CG)

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ICC)
        if u0 is None:
            x = PETSc.Vec().createWithArray(f, comm=PETSc.COMM_SELF)
            ksp.solve(x, x)
            u = x.getArray()
        else:
            f_petsc = PETSc.Vec().createWithArray(f, comm=PETSc.COMM_SELF)
            x = PETSc.Vec().createWithArray(np.copy(_u0), comm=PETSc.COMM_SELF)
            ksp.setInitialGuessNonzero(True)
            ksp.solve(f_petsc, x)
            u = x.getArray()
    else:
        raise Exception("Solver is wrong!")

    t6_end = time.perf_counter()
    print(f"Step 6 (Solve Linear System): {t6_end - t6:.4f} seconds")

    ux, uy = u[0::2], u[1::2]

    ux_grid = np.full((ny * nx,), np.nan)
    uy_grid = np.full((ny * nx,), np.nan)
    ux_grid[used_nodes] = ux
    uy_grid[used_nodes] = uy

    ux_grid = ux_grid.reshape((nx, ny))
    uy_grid = uy_grid.reshape((nx, ny))
    u = np.stack((ux_grid, uy_grid), axis=-1)

    return u


def PlateWithHole_IGA_solver(mask_matrix, model_data, solver="spsolve", u0=None, rtol=1e-6,
                             return_space=False, return_history=False):
    """
    IGA solver for 2D elasticity problem on a Plate with hole under tensions given a mask matrix.
    Inputs:
        - mask_matrix: 2D boolean numpy array (True = included, False = excluded)
        - model_data: problem data including Young's modulus, Poisson's ratio, refinements, etc.
        - solver: which solver to use ("spsolve", "cg", "pyamg", "petsc")
        - u0: optional initial guess sampled on a uniform grid with shape (Mu, Mv, 2)
        - return_space: if True, also return the list of meshes defining the spline space
        - return_history: if True, return convergence trajectory (time, residual norm)
    Outputs:
        - displacements (ux, uy), stress components (sigma_xx, sigma_yy, sigma_xy)
        - optionally, the mesh list describing the spline space
        - optionally, tr_hist: list of (time, residual) pairs
    """
    # Approximation parameters
    p = q = 3  # Polynomial degree of the approximation
    num_refinements = model_data["num_refinements"]  # Number of uniform refinements

    # Step 0: Generate the geometry
    num_pts_u = model_data["numPtsU"]
    num_pts_v = model_data["numPtsV"]
    beam_length = model_data["length"]
    beam_width = model_data["width"]
    traction = model_data["traction"]
    e_mod = model_data["E"]
    nu = model_data["nu"]

    # --- build geometry & patches as before ---
    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    patch_list = [patch1]

    # Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x, y):
        return [0., 0.]

    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    _Y = np.linspace(0, beam_width, num_pts_v)[:, None].flatten()

    def u_bound_neu(x, y, nx, ny):
        return [np.interp(y, _Y, traction), 0.]

    bound_right = boundary2D("Neumann", 0, "right", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)
    bound_all = [bound_right, bound_left]

    # Step 1: Define the material properties
    x_1d = np.linspace(0, beam_length, num_pts_u)
    y_1d = np.linspace(0, beam_width, num_pts_v)
    mask = ~mask_matrix
    e_mod_mat = e_mod * (1e-9 * mask + 1 * (~mask))
    e_mod_fun = RGI((x_1d, y_1d), e_mod_mat.T, method='linear', bounds_error=False)

    material = MaterialElast2D_Hole(Emod=e_mod, nu=nu, vertices=vertices,
                                    elasticity_fun=e_mod_fun, plane_type="stress")

    # Step 2: Degree elevate and refine the geometry
    # t = time.time()
    for patch in patch_list:
        patch.degreeElev(p - 1, q - 1)
    for i in range(2):
        for patch in patch_list:
            patch.refineKnotVectors(True, False)
    for i in range(num_refinements):
        for patch in patch_list:
            patch.refineKnotVectors(True, True)

    mesh_list = [IGAMesh2D(patch) for patch in patch_list]
    for mesh in mesh_list:
        mesh.classify_boundary()

    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)  # returns global size or modifies meshes

    # Step 3: assemble linear system
    gauss_quad_u = gen_gauss_pts(p + 1)
    gauss_quad_v = gen_gauss_pts(q + 1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    stiff, e_modulus, coord = stiff_elast_FGM_2D(mesh_list, material, gauss_rule)

    stiff = stiff.tocsr()
    rhs = np.zeros(2 * size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)

    # --- prepare initial guess x0 from u0 using projection (if u0 provided) ---
    x0 = None
    if u0 is not None:
        # u0 has shape (Mu, Mv, 2) on a uniform [0,1]x[0,1] grid
        # t_proj_start = time.time()
        x0 = project_disp_grid_to_bspline(u0, mesh_list)
        # print("Projection time is ", time.time() - t_proj_start)

    # --- Solve linear system using solver choice, passing x0 to iterative solvers ---
    t0 = time.time()
    if solver == "spsolve":
        sol = spsolve(stiff, rhs)
        # print("Condition number is", np.linalg.cond(stiff.todense()))

    elif solver == "cg":
        tr_hist = []
        t_cg = time.perf_counter()

        def cb(xk):
            r_norm = np.linalg.norm(rhs - stiff @ xk)
            tr_hist.append((time.perf_counter() - t_cg, r_norm))

        sol, info = cg(stiff, rhs, x0=x0, rtol=rtol, callback=cb)
        # if x0 is not None:
        #     rel_err = np.linalg.norm(sol - x0) / np.linalg.norm(sol)
        #     print("Relative error of initial guess is", rel_err)
        if info != 0:
            print("Warning: CG did not fully converge")
    elif solver == "pyamg":
        ml = pyamg.ruge_stuben_solver(stiff)
        sol = ml.solve(rhs, x0=x0, tol=rtol)
    elif solver == "petsc":
        K = PETSc.Mat().createAIJ(size=stiff.shape, csr=(stiff.indptr, stiff.indices, stiff.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(K)
        ksp.setType(PETSc.KSP.Type.CG)

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.ICC)
        f_petsc = PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_SELF)
        if x0 is None:
            x = PETSc.Vec().createSeq(stiff.shape[0])
            ksp.solve(f_petsc, x)
            sol = x.getArray()
        else:
            x = PETSc.Vec().createWithArray(np.copy(x0), comm=PETSc.COMM_SELF)
            ksp.setInitialGuessNonzero(True)
            ksp.solve(f_petsc, x)
            sol = x.getArray()
    else:
        raise ValueError("Unknown solver: " + str(solver))
    solver_time = time.time() - t0

    # --- post-processing (unchanged) ---
    num_pts_xi = model_data["numPtsU"]
    num_pts_eta = model_data["numPtsV"]
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = (
        comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol, get_measurements_vector, num_fields))
    # num_output_fields = 4
    # meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(
    #     num_pts_xi, num_pts_eta, mesh_list, sol, get_measurement_stresses_FGM, num_output_fields, material)

    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    disp2d = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    # field_names = ['stress_xx', 'stress_yy', 'stress_xy']
    # stress = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
    #                         meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)

    disp_grid = disp2d.transpose((1, 0, 2))
    outputs = [disp_grid, sol, solver_time]
    if return_space:
        outputs.append(mesh_list)
    if return_history:
        outputs.append(tr_hist)

    return tuple(outputs)
    # return disp_grid, stress.transpose((1, 0, 2)), sol


# -------------------------
# B-spline utilities
# -------------------------
def make_open_uniform_knots(n_ctrl, degree, domain=(0.0, 1.0)):
    """
    Open-uniform knot vector on [a,b] for given number of control points and degree.
    Length = n_ctrl + degree + 1
    """
    a, b = domain
    p = int(degree)
    m = n_ctrl + p + 1  # total number of knots
    knots = np.empty(m, dtype=float)

    # clamped ends
    knots[:p + 1] = a
    knots[-(p + 1):] = b

    # required number of interior knots
    n_interior = n_ctrl - p - 1  # may be 0 or more

    if n_interior > 0:
        # produce exactly n_interior interior knots strictly between a and b
        # we create n_interior+2 samples including endpoints, then drop endpoints
        interior = np.linspace(a, b, n_interior + 2, endpoint=True)[1:-1]
        knots[p + 1: m - (p + 1)] = interior
    # else: no interior knots, slice is empty and left as-is

    return knots


def find_span(n_ctrl, degree, u, knots):
    """
    Find the span index i such that u in [knots[i], knots[i+1]).
    For u == knots[-1], return n_ctrl-1 (last span).
    """
    p = degree
    if u >= knots[-1] - 1e-14:  # protect against float roundoff at right end
        return n_ctrl - 1
    # searchsorted returns index j such that knots[j-1] <= u < knots[j]
    j = np.searchsorted(knots, u, side='right') - 1
    # clamp to valid span range [p, n_ctrl-1]
    return int(min(max(j, p), n_ctrl - 1))


def basis_funs(span, u, degree, knots):
    """
    Piegl & Tiller Algorithm A2.2: non-recursive computation of the p+1 nonzero basis functions.
    Returns array N[0..p] corresponding to indices span-p .. span.
    """
    p = degree
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(0, j):
            denom = right[r + 1] + left[j - r]
            if denom == 0.0:
                term = 0.0
            else:
                term = N[r] / denom
            temp = term * right[r + 1]
            N[r] = saved + temp
            saved = term * left[j - r]
        N[j] = saved
    return N


def bspline_basis_matrix(params, n_ctrl, degree, knots=None, domain=(0.0, 1.0)):
    """
    Build dense evaluation matrix B of shape (len(params), n_ctrl),
    where B[i, a] = N_a^{(degree)}(params[i]).
    Uses local support to fill only (degree+1) entries per row.
    """
    if knots is None:
        knots = make_open_uniform_knots(n_ctrl, degree, domain=domain)
    p = degree
    mu = len(params)
    B = np.zeros((mu, n_ctrl), dtype=float)
    for i, u in enumerate(params):
        span = find_span(n_ctrl, p, u, knots)
        Nloc = basis_funs(span, u, p, knots)  # length p+1
        a0 = span - p
        B[i, a0:a0 + p + 1] = Nloc
    return B, knots


# -------------------------
# Weights and helpers
# -------------------------

def uniform_weights(n, domain=(0.0, 1.0)):
    """
    Simple rectangle-rule weights for equally spaced samples across [a,b].
    """
    a, b = domain
    return np.full(n, (b - a) / n, dtype=float)


def apply_row_weights(A, w):
    """
    Return (diag(w) @ A) without forming the diagonal: scales each row i by w[i].
    """
    return (w[:, None] * A)


def apply_col_weights(A, w):
    """
    Return (A @ diag(w)) without forming the diagonal: scales each column j by w[j].
    """
    return (A * w[None, :])


# -------------------------
# Core L2 projection (scalar component)
# -------------------------

def project_scalar_l2(F, Bu, Bv, wu=None, wv=None, reg_u=0.0, reg_v=0.0):
    """
    L2 (weighted least-squares) projection for a single scalar field sampled on a grid.

    Minimize || Wu^{1/2} (F - Bu C Bv^T) Wv^{1/2} ||_F^2
    over control matrix C (nu x nv), where:
      - F: (Mu x Mv) sample values
      - Bu: (Mu x nu) basis matrix in u
      - Bv: (Mv x nv) basis matrix in v
      - wu: length-Mu nonnegative weights (default: ones)
      - wv: length-Mv nonnegative weights (default: ones)
    reg_u, reg_v: small Tikhonov regularization (adds reg * I to 1D Grams)
    """
    Mu, Mv = F.shape
    nu = Bu.shape[1]
    nv = Bv.shape[1]

    if wu is None:
        wu = np.ones(Mu, dtype=float)
    if wv is None:
        wv = np.ones(Mv, dtype=float)

    # 1D Gram matrices (SPD, banded if Bu/Bv are banded)
    Gu = Bu.T @ apply_row_weights(Bu, wu)  # nu x nu
    Gv = Bv.T @ apply_row_weights(Bv, wv)  # nv x nv

    if reg_u > 0:
        Gu.flat[::nu + 1] += reg_u
    if reg_v > 0:
        Gv.flat[::nv + 1] += reg_v

    # Cholesky factors
    Lu = cholesky(Gu, lower=True, overwrite_a=False, check_finite=True)
    Lv = cholesky(Gv, lower=True, overwrite_a=False, check_finite=True)

    # RHS: R = Bu^T * Wu * F * Wv * Bv
    FW = apply_col_weights(F, wv)  # F * diag(wv)   (Mu x Mv)
    WF = apply_row_weights(FW, wu)  # diag(wu) * F * diag(wv)
    R = Bu.T @ WF @ Bv  # (nu x nv)

    # Solve Gu C Gv^T = R using triangular solves only.
    # Step 1: Lu X = R
    X = solve_triangular(Lu, R, lower=True)

    # Step 2: Z Lv^T = X  ->  (Lv) Z^T = X^T
    Zt = solve_triangular(Lv, X.T, lower=True)  # solves Lv * Zt = X^T
    Z = Zt.T

    # Step 3: Lu^T Y = Z
    Y = solve_triangular(Lu.T, Z, lower=False)

    # Step 4: C Lv = Y  ->  (Lv^T) C^T = Y^T
    Ct = solve_triangular(Lv.T, Y.T, lower=False)  # solves Lv^T * Ct = Y^T
    C = Ct.T

    return C


# -------------------------
# Vector-field wrapper
# -------------------------

def project_vector_field(u_grid, v_grid, Fx, Fy, nu=64, nv=256, deg_u=3, deg_v=3,
                         wu=None, wv=None, domain_u=(0.0, 1.0), domain_v=(0.0, 1.0),
                         reg_u=0.0, reg_v=0.0):
    """
    Project a 2D vector field (Fx, Fy) sampled on tensor grid (u_grid x v_grid)
    onto bicubic B-splines with (nu x nv) control points.

    Returns:
      (Cu_x, Cv_x, Cu_y, Cv_y, Bu, Bv)
      where Cu_x and Cu_y are the control coefficient matrices for Fx and Fy respectively,
      and (Bu, Bv) are the basis matrices you can reuse for reconstruction.
    """
    Mu = len(u_grid)
    Mv = len(v_grid)
    assert Fx.shape == (Mu, Mv)
    assert Fy.shape == (Mu, Mv)

    Bu, _ = bspline_basis_matrix(u_grid, nu, deg_u, domain=domain_u)
    Bv, _ = bspline_basis_matrix(v_grid, nv, deg_v, domain=domain_v)

    if wu is None:
        wu = np.ones(Mu, dtype=float)
    if wv is None:
        wv = np.ones(Mv, dtype=float)

    Cx = project_scalar_l2(Fx, Bu, Bv, wu=wu, wv=wv, reg_u=reg_u, reg_v=reg_v)
    Cy = project_scalar_l2(Fy, Bu, Bv, wu=wu, wv=wv, reg_u=reg_u, reg_v=reg_v)
    return Cx, Cy, Bu, Bv


def reconstruct(Bu, C, Bv):
    """
    Evaluate the spline surface given control matrix C on the sample grid defined by (Bu,Bv):
      F_hat = Bu @ C @ Bv^T
    """
    return Bu @ C @ Bv.T


def _local_to_global_map(mesh):
    """Return array mapping each local basis index to global index for a mesh."""
    mapping = np.full(mesh.num_basis, -1, dtype=int)
    for local_nodes, global_nodes in zip(mesh.elem_node, mesh.elem_node_global):
        mapping[local_nodes] = global_nodes
    if np.any(mapping < 0):
        raise ValueError("Incomplete local-to-global map detected for mesh")
    return mapping


def project_disp_grid_to_bspline(grid_disp, mesh_list, domain_u=(0.0, 1.0), domain_v=(0.0, 1.0),
                                 reg_u=1e-12, reg_v=1e-12):
    """Project displacement grid onto the spline space described by ``mesh_list``.

    The incoming grid is typically stored with the first axis corresponding to the
    v-direction (as produced by :func:`plot_fields_2D`).  We detect the correct
    orientation by projecting both axis orderings onto the first mesh and selecting
    the one with the smaller reconstruction error before assembling the global
    coefficient vector.
    """
    grid_disp = np.asarray(grid_disp)
    if grid_disp.ndim != 3 or grid_disp.shape[-1] != 2:
        raise ValueError("grid_disp must have shape (Mu, Mv, 2)")

    if not mesh_list:
        raise ValueError("mesh_list must contain at least one mesh")

    def _extract_components(arr, orientation):
        if orientation == "uv":
            Fx = arr[..., 0]
            Fy = arr[..., 1]
        elif orientation == "vu":
            Fx = arr[..., 0].T
            Fy = arr[..., 1].T
        else:
            raise ValueError(f"Unknown orientation '{orientation}'")
        return Fx, Fy

    def _evaluate_orientation(mesh, orientation):
        Fx, Fy = _extract_components(grid_disp, orientation)
        Mu, Mv = Fx.shape
        u_grid = np.linspace(domain_u[0], domain_u[1], Mu)
        v_grid = np.linspace(domain_v[0], domain_v[1], Mv)
        wu = uniform_weights(Mu, domain_u)
        wv = uniform_weights(Mv, domain_v)

        nu, nv = mesh.n
        deg_u, deg_v = mesh.deg.tolist()
        Cx, Cy, Bu, Bv = project_vector_field(
            u_grid, v_grid, Fx, Fy,
            nu=nu, nv=nv,
            deg_u=deg_u, deg_v=deg_v,
            wu=wu, wv=wv,
            domain_u=domain_u, domain_v=domain_v,
            reg_u=reg_u, reg_v=reg_v
        )

        recon_x = reconstruct(Bu, Cx, Bv)
        recon_y = reconstruct(Bu, Cy, Bv)
        target = np.stack((Fx, Fy))
        recon = np.stack((recon_x, recon_y))
        denom = np.linalg.norm(target)
        rel_err = 0.0 if denom == 0 else np.linalg.norm(recon - target) / denom
        return {
            "Fx": Fx,
            "Fy": Fy,
            "Mu": Mu,
            "Mv": Mv,
            "u_grid": u_grid,
            "v_grid": v_grid,
            "wu": wu,
            "wv": wv,
            "Cx": Cx,
            "Cy": Cy,
            "rel_err": rel_err,
        }

    first_mesh = mesh_list[0]
    orient_data = {
        orientation: _evaluate_orientation(first_mesh, orientation)
        for orientation in ("uv", "vu")
    }
    best_orientation = min(orient_data, key=lambda ori: orient_data[ori]["rel_err"])
    best_data = orient_data[best_orientation]

    Fx = best_data["Fx"]
    Fy = best_data["Fy"]
    u_grid = best_data["u_grid"]
    v_grid = best_data["v_grid"]
    wu = best_data["wu"]
    wv = best_data["wv"]

    max_global = -1
    for mesh in mesh_list:
        for g_nodes in mesh.elem_node_global:
            max_global = max(max_global, int(np.max(g_nodes)))
    num_global = max_global + 1
    sol_proj = np.zeros(2 * num_global)

    for mesh in mesh_list:
        nu, nv = mesh.n
        deg_u, deg_v = mesh.deg.tolist()
        if mesh is first_mesh:
            Cx, Cy = best_data["Cx"], best_data["Cy"]
        else:
            Cx, Cy, _, _ = project_vector_field(
                u_grid, v_grid, Fx, Fy,
                nu=nu, nv=nv,
                deg_u=deg_u, deg_v=deg_v,
                wu=wu, wv=wv,
                domain_u=domain_u, domain_v=domain_v,
                reg_u=reg_u, reg_v=reg_v
            )
        local_to_global = _local_to_global_map(mesh)
        Cx_flat = Cx.T.flatten()
        Cy_flat = Cy.T.flatten()
        sol_proj[2 * local_to_global] = Cx_flat
        sol_proj[2 * local_to_global + 1] = Cy_flat

    return sol_proj
