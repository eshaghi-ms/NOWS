import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags


# Generate Gaussian Random Field (GRF) with periodic BCs
def generate_grf(N, gamma, tau, sigma):
    k = np.fft.fftfreq(N, d=1 / N) * 2 * np.pi
    eigs = sigma * ((k ** 2 + tau ** 2) ** (-gamma / 2))
    xi = np.random.randn(N) + 1j * np.random.randn(N)
    u_hat = eigs * xi
    u = np.real(ifft(u_hat)) * N
    return u


# Burgers solver (spectral semi-implicit with direct division in Fourier space)
def burgers_solver_iterative(init, tspan, s, visc):
    dt = tspan[1] - tspan[0]
    x = np.linspace(0, 1, s, endpoint=False)
    k = fftfreq(s, d=1 / s) * 2 * np.pi
    k2 = k ** 2

    u = init.copy()
    u_hist = [u.copy()]
    L = 1 + dt * visc * k2  # Spectral denominator

    for t in tspan[1:]:
        u_sq = u ** 2
        nonlinear = -0.5 * np.gradient(u_sq, x, edge_order=2)
        rhs = u + dt * nonlinear
        u_hat = fft(rhs)
        u = np.real(ifft(u_hat / L))
        u_hist.append(u.copy())

    return np.array(u_hist)


# Burgers solver (direct dense solve with finite differences)
def burgers_solver_direct(init, tspan, s, visc):
    dt = tspan[1] - tspan[0]
    dx = 1 / s
    x = np.linspace(0, 1, s, endpoint=False)

    # Construct second derivative matrix with periodic BCs
    main_diag = -2 * np.ones(s)
    off_diag = np.ones(s - 1)
    lap = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(s, s)).toarray()
    lap[0, -1] = lap[-1, 0] = 1  # periodic BCs
    lap /= dx ** 2

    A = np.eye(s) - dt * visc * lap  # Implicit matrix

    u = init.copy()
    u_hist = [u.copy()]

    for t in tspan[1:]:
        u_sq = u ** 2
        nonlinear = -0.5 * np.gradient(u_sq, dx, edge_order=2)
        rhs = u + dt * nonlinear
        u = np.linalg.solve(A, rhs)
        u_hist.append(u.copy())

    return np.array(u_hist)


# Parameters
N = 1
gamma = 2.5
tau = 7
sigma = 7 ** 2
visc = 1 / 1000
s = 1024
steps = 200
tspan = np.linspace(0, 1, steps + 1)

# Generate initial condition
u0 = generate_grf(s, gamma, tau, sigma)

# Solve using both methods
sol_iter = burgers_solver_iterative(u0, tspan, s, visc)
sol_direct = burgers_solver_direct(u0, tspan, s, visc)

# Plot final solutions and difference
x = np.linspace(0, 1, s, endpoint=False)

plt.figure(figsize=(10, 5))
plt.plot(x, sol_iter[-1], label='Iterative (spectral)', linewidth=2)
plt.plot(x, sol_direct[-1], '--', label='Direct (dense solve)', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title("Burgers' Equation Final Time Comparison")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 3))
plt.plot(x, sol_iter[-1] - sol_direct[-1], label='Difference', color='red')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Difference: Iterative - Direct')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
