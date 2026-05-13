import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1, ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)),
                        0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)),
                        0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def plot_test_sample_1d(index, x_test, y_test, pred, s):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create x-axis grid
    x_plot = np.linspace(0, 1, s)

    # Get data
    x_input = x_test[index, :, 0].cpu().numpy()
    y_exact = y_test[index, -1, :].cpu().numpy()
    y_pred = pred[index, -1, :].cpu().numpy()

    # Plot u(x): predicted vs exact
    plt.figure(figsize=(8, 3))
    plt.plot(x_plot, y_exact, label='Exact', linewidth=2)
    plt.plot(x_plot, y_pred, '--', label='Prediction', linewidth=2)
    plt.title(f'Burgers Equation Output - Test Sample #{index}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot a(x): input
    plt.figure(figsize=(8, 3))
    plt.plot(x_plot, x_input, color='purple', label='Input a(x)', linewidth=2)
    plt.title(f'Input Initial Condition - Test Sample #{index}')
    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()


# Burgers solver using semi-implicit spectral method
def burgers_solver_spectral(init, tspan, s, visc):
    from scipy.fft import fft, ifft, fftfreq
    dt = tspan[1] - tspan[0]
    x = np.linspace(0, 1, s, endpoint=False)
    k = fftfreq(s, d=1 / s) * 2 * np.pi
    k2 = k ** 2

    u = init.copy()
    u_hist = [u.copy()]
    L = 1 + dt * visc * k2

    for t in tspan[1:]:
        u_sq = u ** 2
        nonlinear = -0.5 * np.gradient(u_sq, x, edge_order=2)
        rhs = u + dt * nonlinear
        rhs_hat = fft(rhs)
        u_hat_new = rhs_hat / L
        u = np.real(ifft(u_hat_new))
        u_hist.append(u.copy())

    return np.array(u_hist)


def burgers_solver_direct(init, tspan, s, visc):
    from scipy.sparse import diags
    dt = tspan[1] - tspan[0]
    dx = 1 / s
    x = np.linspace(0, 1, s, endpoint=False)

    # Construct second derivative matrix with periodic BCs
    main_diag = -2 * np.ones(s)
    off_diag = np.ones(s - 1)
    lap = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(s, s)).toarray()
    lap[0, -1] = lap[-1, 0] = 1  # periodic BCs
    lap /= dx ** 2

    A = np.eye(s) - dt * visc * lap

    u = init.copy()
    u_hist = [u.copy()]

    for t in tspan[1:]:
        u_sq = u ** 2
        nonlinear = -0.5 * np.gradient(u_sq, dx, edge_order=2)
        rhs = u + dt * nonlinear
        u = np.linalg.solve(A, rhs)
        u_hist.append(u.copy())

    return np.array(u_hist)


def burgers_solver_cg(init, tspan, s, visc, x0=None, rtol=1e-8, maxiter=1000):
    import numpy as np
    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import cg

    dt = tspan[1] - tspan[0]
    dx = 1 / s
    x = np.linspace(0, 1, s, endpoint=False)

    # Construct periodic Laplacian matrix (second derivative)
    main_diag = -2 * np.ones(s)
    off_diag = np.ones(s - 1)
    lap = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(s, s)).toarray()
    lap[0, -1] = lap[-1, 0] = 1  # enforce periodic BCs
    lap /= dx ** 2

    # Convert to sparse matrix for CG
    A = csr_matrix(np.eye(s) - dt * visc * lap)

    u = init.copy()
    u_hist = [u.copy()]

    for i, t in enumerate(tspan[1:]):
        u_sq = u ** 2
        nonlinear = -0.5 * np.gradient(u_sq, dx, edge_order=2)
        rhs = u + dt * nonlinear

        # Solve A u_next = rhs using CG
        if x0 is None:
            u, info = cg(A, rhs, x0=u, rtol=rtol, atol=0., maxiter=maxiter)
        else:
            if i != 0:
                # u, info = cg(A, rhs, x0=x0[i + 1], rtol=rtol, atol=0., maxiter=maxiter)
                u, info = cg(A, rhs, x0=x0[i + 1], rtol=rtol, atol=0., maxiter=maxiter)
            else:
                u, info = cg(A, rhs, x0=u, rtol=rtol, atol=0., maxiter=maxiter)

        if info != 0:
            raise RuntimeError(f"CG did not converge at time {t}, info: {info}")

        u_hist.append(u.copy())

    return np.array(u_hist)


def NEW_burgers_solver_spectral(init, tspan, s, visc):
    """
    Strang-splitting spectral solver for viscous Burgers' equation on [0,1] with periodic BC.
    Matches a second-order accurate MATLAB spinop implementation.
    init: initial array of length s
    tspan: array of time points (length M+1)
    s: number of grid points
    visc: viscosity nu
    Returns: array of shape (M+1, s)
    """
    import numpy as np
    from scipy.fft import fft, ifft, fftfreq
    # spatial grid
    x = np.linspace(0, 1, s, endpoint=False)
    # wave numbers
    k = fftfreq(s, d=1 / s) * 2 * np.pi
    k2 = k ** 2
    # dealiasing mask (2/3 rule)
    kmax = np.max(np.abs(k))
    mask = np.abs(k) <= (2 / 3) * kmax

    dt = tspan[1] - tspan[0]
    # initial spectral state
    u_hat = fft(init)
    # record history
    u_hist = [init.copy()]

    # precompute linear propagator
    L_half = np.exp(-visc * k2 * dt / 2)

    for _ in tspan[1:]:
        # 1) half linear step
        u_hat = u_hat * L_half
        u = np.real(ifft(u_hat))

        # 2) full nonlinear step via RK2 in spectral form
        # compute nonlinear rhs in spectral: -0.5 d_x(u^2)
        u_sq = u * u
        non_hat = -0.5j * k * fft(u_sq)
        non_hat *= mask
        non = np.real(ifft(non_hat))
        # midpoint
        u_mid = u + dt * non
        u_mid_sq = u_mid * u_mid
        non_mid_hat = -0.5j * k * fft(u_mid_sq)
        non_mid_hat *= mask
        non_mid = np.real(ifft(non_mid_hat))
        # advance
        u = u + dt * 0.5 * (non + non_mid)

        # 3) half linear step
        u_hat = fft(u) * L_half
        u = np.real(ifft(u_hat))

        u_hist.append(u.copy())

    return np.array(u_hist)


def NEW_burgers_solver_direct(init, tspan, s, visc,
                              cfl_adv=0.5, cfl_diff=0.5):
    """
    Direct finite-difference solver for viscous Burgers' equation
    with periodic BC, using adaptive sub-stepping to enforce
    explicit stability limits.

    Parameters
    ----------
    init : array_like, shape (s,)
        Initial u at uniform grid on [0,1).
    tspan : array_like, shape (M+1,)
        Times to record solution, assumed uniform spacing dt.
    s : int
        Number of grid points.
    visc : float
        Viscosity nu.
    cfl_adv : float, optional
        Safety factor for advective CFL (<=1).
    cfl_diff : float, optional
        Safety factor for diffusive CFL (<=1).

    Returns
    -------
    u_hist : ndarray, shape (M+1, s)
        u(t_n, x_j) for each t_n in tspan.
    """
    x = np.linspace(0, 1, s, endpoint=False)
    dx = 1.0 / s
    dt = tspan[1] - tspan[0]

    def rhs(u):
        # periodic neighbors
        up = np.roll(u, -1)
        um = np.roll(u, +1)
        ux = (up - um) / (2 * dx)
        uxx = (up - 2 * u + um) / (dx * dx)
        return -u * ux + visc * uxx

    u = init.copy()
    u_hist = [u.copy()]

    for _ in tspan[1:]:
        # Estimate maximum stable sub‐step for this u
        umax = np.max(np.abs(u))
        if umax == 0:
            dt_adv = np.inf
        else:
            dt_adv = cfl_adv * dx / umax
        dt_diff = cfl_diff * dx * dx / visc if visc > 0 else np.inf
        dt_max = min(dt_adv, dt_diff)
        # Number of sub‐steps
        nsub = max(1, int(np.ceil(dt / dt_max)))
        small_dt = dt / nsub

        # sub‐step loop (RK2)
        for _sub in range(nsub):
            k1 = rhs(u)
            umid = u + small_dt * k1
            k2 = rhs(umid)
            u = u + 0.5 * small_dt * (k1 + k2)

        u_hist.append(u.copy())

    return np.array(u_hist)


def NEW_burgers_solver_cg(init, tspan, s, visc, X0=None, rtol=1e-10, maxiter=1000):
    """
    Burgers solver using Strang splitting and iterative CG for diffusion.

    Parameters
    ----------
    init : array_like, shape (s,)
        Initial condition.
    tspan : array_like, shape (M+1,)
        Time points.
    s : int
        Number of grid points.
    visc : float
        Viscosity nu.
    X0 : array_like or None, shape (M+1, s), optional
        Initial guesses for u at each time level. If provided, used as CG x0.
    tol : float
        CG tolerance.
    maxiter : int
        CG maximum iterations.

    Returns
    -------
    u_hist : ndarray, shape (M+1, s)
        Solution at each time in `tspan`.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg
    # grid and time
    x = np.linspace(0, 1, s, endpoint=False)
    dx = 1.0 / s
    dt = tspan[1] - tspan[0]
    M = len(tspan) - 1

    # 2nd-order periodic FD Laplacian
    e = np.ones(s)
    Dxx = diags([e, -2*e, e], [-1, 0, 1], shape=(s, s), format='lil')
    Dxx[0, -1] = 1
    Dxx[-1, 0] = 1
    Dxx = Dxx.tocsr() / dx**2

    # Linear operator for half-step
    I = diags([1.0], [0], shape=(s, s), format='csr')
    A_half = (I - 0.5*dt*visc*Dxx).tocsr()

    u = init.copy()
    u_hist = [u.copy()]

    for n in range(1, M+1):
        # choose CG initial guesses if provided
        x0_first = X0[n-1] if X0 is not None else u.copy()

        # 1) half linear step: (I - 0.5 dt visc Dxx) u = u_old
        u, info1 = cg(A_half, u, x0=x0_first, rtol=rtol, maxiter=maxiter)

        # 2) nonlinear RK2 step
        u_sq = u*u
        non = -0.5 * np.gradient(u_sq, dx)
        u_mid = u + dt*non
        mid_sq = u_mid*u_mid
        non_mid = -0.5 * np.gradient(mid_sq, dx)
        u = u + 0.5*dt*(non + non_mid)

        # 3) half linear step again
        x0_second = X0[n] if X0 is not None else u.copy()
        u, info2 = cg(A_half, u, x0=x0_second, rtol=rtol, maxiter=maxiter)

        u_hist.append(u.copy())

    return np.array(u_hist)
