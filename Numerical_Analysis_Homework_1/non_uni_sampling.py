
from prony import prony
import numpy as np
from scipy.interpolate import CubicSpline, BarycentricInterpolator

alpha = 1
M = 5
SEED = 0
ETA = [1e-9, 1e-6, 1e-3, 1e-1]

rng = np.random.default_rng(SEED)
c = rng.normal(0, 1, M)
w = rng.uniform(-alpha, 0, M) + 1j * rng.uniform(-np.pi / 2, np.pi / 2, M)


def signal(t, w, c):
    """Evaluate h(t) = sum_j c_j exp(w_j t) at every point of the array t."""
    h = np.zeros(len(t), dtype=complex)
    for j in range(len(w)):
        h += c[j] * np.exp(w[j] * t)
    return h




def generation():
    """Perturbed measurements h(k + eps_k), k = 0, ..., 2M-1, one row per eta."""
    h = np.zeros((len(ETA), 2 * M), dtype=complex)
    t = np.zeros((len(ETA), 2 * M))
    for i in range(len(ETA)):
        eps = rng.uniform(-ETA[i], ETA[i], 2 * M)
        t[i] = np.arange(2 * M) + eps          # sampling points k + eps_k
        h[i] = signal(t[i], w, c)
    return h,t




def error(h, N=500):
    """Continuous reconstruction error E on a uniform grid of N+1 points in [0, 2M-1]."""
    x = np.linspace(0, 2 * M - 1, N + 1)
    h_true = signal(x, w, c)
    errors = np.zeros(len(ETA))
    for i in range(len(ETA)):
        w_hat, c_hat = prony(h[i])
        h_rec = signal(x, w_hat, c_hat)
        errors[i] = np.mean(np.abs(h_true - h_rec) ** 2)
    return errors



def interpolant(h,t,method):
    k = np.arange(2 * M)
    h_hat = np.zeros_like(h)
    for i in range(len(ETA)):
        if method == "linear":       # piecewise linear 
            h_hat[i] = np.interp(k, t[i], h[i].real) + 1j * np.interp(k, t[i], h[i].imag)
        elif method == "spline":     # cubic spline and extrapolates at the boundaries
            h_hat[i] = CubicSpline(t[i], h[i])(k)
        elif method == "lagrange":   # global polynomial of degree 2M-1
            h_hat[i] = BarycentricInterpolator(t[i], h[i])(k)
    return h_hat

