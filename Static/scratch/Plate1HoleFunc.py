import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io


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


def quad_B_at_center(xe, ye):
    """
    Compute the B matrix (and detJ) for a 4-node quad element at the
    natural coordinate (xi,eta) = (0,0) for stress evaluation.
    """
    xi = 0.0
    eta = 0.0
    dN_dxi = np.array([
        -0.25 * (1 - eta),
        0.25 * (1 - eta),
        0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
        0.25 * (1 + xi),
        0.25 * (1 - xi)
    ])
    J = np.zeros((2, 2))
    J[0, 0] = np.sum(dN_dxi * xe)
    J[0, 1] = np.sum(dN_dxi * ye)
    J[1, 0] = np.sum(dN_deta * xe)
    J[1, 1] = np.sum(dN_deta * ye)
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
    return B, detJ


def PlateWithHole_solver(mask_matrix, x_max=5, y_max=5, E=1e3, nu=0.3):
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

    PlotMesh(cells, points)

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

    K = sp.coo_matrix((K_data, (K_row, K_col)), shape=(num_dofs, num_dofs)).tocsr()

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
    t6 = time.perf_counter()

    u = spla.spsolve(K, f)
    ux, uy = u[0::2], u[1::2]

    t6_end = time.perf_counter()
    print(f"Step 6 (Solve Linear System): {t6_end - t6:.4f} seconds")

    return ux, uy, cells, points


# ------------------ Main Program ------------------

# Define domain limits.
x_min, x_max = 0, 5
y_min, y_max = 0, 5

E = 1e3
nu = 0.3
D = E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                  [nu, 1, 0],
                                  [0, 0, (1 - nu) / 2]])

# --- Modified Mask Generation for a 200x200 Mesh ---
num_points = 200  # Use 200 nodes in each direction
x_coords = np.linspace(x_min, x_max, num_points)
y_coords = np.linspace(y_min, y_max, num_points)
X, Y = np.meshgrid(x_coords, y_coords)
points_all = np.column_stack((X.flatten(), Y.flatten()))

# Define a mask for nodes: True if node is outside the hole, False if inside.
center = np.array([2.5, 2.5])
radius = 0.5
dist = np.linalg.norm(points_all - center, axis=1)
mask_nodes = dist >= radius  # Outside the hole: True
mask_nodes = mask_nodes.reshape((num_points, num_points))

# Solve the problem using the 200x200 mesh.
ux, uy, cells, points = PlateWithHole_solver(mask_nodes, x_max=x_max, y_max=y_max, E=E, nu=nu)

u = np.empty((ux.size + uy.size,), dtype=ux.dtype)
u[0::2] = ux
u[1::2] = uy

# =============================================================================
# Timing: Step 7 - Post-Processing (Stresses and Nodal Averages)
# =============================================================================
t7 = time.perf_counter()

num_nodes = points.shape[0]
sigma_xx = np.zeros(num_nodes)
sigma_yy = np.zeros(num_nodes)
sigma_xy = np.zeros(num_nodes)
count = np.zeros(num_nodes)

for cell in cells:
    dofs = []
    for nid in cell:
        dofs.extend([2 * int(nid), 2 * int(nid) + 1])
    u_e = u[np.array(dofs, dtype=int)]
    xe = points[cell, 0]
    ye = points[cell, 1]
    B, detJ = quad_B_at_center(xe, ye)
    strain = B @ u_e
    stress = D @ strain  # [σ_xx, σ_yy, σ_xy]
    for nid in cell:
        sigma_xx[nid] += stress[0]
        sigma_yy[nid] += stress[1]
        sigma_xy[nid] += stress[2]
        count[nid] += 1

sigma_xx /= count
sigma_yy /= count
sigma_xy /= count
von_mises = np.sqrt(sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3 * sigma_xy ** 2)

t7_end = time.perf_counter()
print(f"Step 7 (Post-Processing): {t7_end - t7:.4f} seconds")

# =============================================================================
# Timing: Step 8 - Plot Field Variables
# =============================================================================
t8 = time.perf_counter()

# For plotting field variables with matplotlib’s tripcolor,
# convert each quad element into two triangles.
triangles_plot = []
for cell in cells:
    triangles_plot.append([cell[0], cell[1], cell[2]])
    triangles_plot.append([cell[0], cell[2], cell[3]])
triangles_plot = np.array(triangles_plot)

triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles_plot)

plt.figure()
plt.tripcolor(triangulation, ux, shading='flat', cmap='jet')
plt.title('Horizontal Displacement (ux)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.figure()
plt.tripcolor(triangulation, uy, shading='flat', cmap='jet')
plt.title('Vertical Displacement (uy)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.figure()
plt.tripcolor(triangulation, sigma_xx, shading='flat', cmap='jet')
plt.title('σ_xx Stress')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.figure()
plt.tripcolor(triangulation, sigma_yy, shading='flat', cmap='jet')
plt.title('σ_yy Stress')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.figure()
plt.tripcolor(triangulation, sigma_xy, shading='flat', cmap='jet')
plt.title('σ_xy Stress')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.figure()
plt.tripcolor(triangulation, von_mises, shading='flat', cmap='jet')
plt.title('Von Mises Stress')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.colorbar()

plt.show()

t8_end = time.perf_counter()
print(f"Step 8 (Save & Plot): {t8_end - t8:.4f} seconds")
