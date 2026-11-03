import numpy as np
from prony import prony

M = 4
T = np.array([0.5, 1.3, 2.0, 3.2, 4.0])   # T_1 < ... < T_{M+1}
b = np.array([1.0, -0.5, 2.0, 0.8])       # plateau values

K = 2 * M + 2                             # minimal number of measurements (3.b)
dgamma = 0.8 * np.pi / T[-1]              # step size, dgamma 


def generation_fourier(T, b, K, dgamma):
    """Measurements h_hat_k = (i k dgamma) f_hat(k dgamma), k = 0, ..., K-1, from eq. (10)."""
    k = np.arange(K)
    h_hat = np.zeros(K, dtype=complex)
    for j in range(len(b)):
        h_hat += b[j] * (np.exp(-1j * k * dgamma * T[j]) - np.exp(-1j * k * dgamma * T[j + 1]))
    return h_hat


def reconstruction(omega_hat, c_hat, dgamma):
    """Recover T_j and b_j from the Prony parameters (3.a)."""
    M = len(omega_hat) - 1                 
    T_r = np.zeros(M + 1)
    b_r = np.zeros(M)

    for i in range(M + 1):
        T_r[i] = -omega_hat[i].imag / dgamma   

    order = np.argsort(T_r)                # Prony returns the pairs in arbitrary order
    T_r = T_r[order]
    c_sorted = c_hat[order]                # same permutation for the coefficients

    b_current = 0
    for i in range(M):
        b_current += c_sorted[i].real      # b_j = b_{j-1} + c_j
        b_r[i] = b_current

    return b_r, T_r


def recovery_error(T, b, T_r, b_r):
    """Maximum absolute error on the breakpoints and on the plateau values."""
    return np.max(np.abs(T - T_r)), np.max(np.abs(b - b_r))


x = np.linspace(0,5,1000)
def piecewise_constant(b, T, x):
    """Evaluate f(x) = sum_j b_j 1_[T_j, T_{j+1})(x) on the array x, eq. (9)."""
    f = np.zeros(len(x))
    for j in range(len(b)):
        indicator = (x >= T[j]) & (x < T[j + 1])   # 1 on [T_j, T_{j+1}), 0 elsewhere
        f += b[j] * indicator
    return f

K_ = [10, 20, 50, 100]


def inverse_fourier(x, T, b, h_hat, dgamma):
    """Trapezoidal approximation of the inverse Fourier transform of f_hat
    from the K measurements."""
    K = len(h_hat)

    # Fourier measurements f_hat(k dgamma)
    f_hat = np.zeros(K, dtype=complex)
    for i in range(K):
        if i == 0:
            for j in range(len(b)):
                f_hat[i] += b[j] * (T[j + 1] - T[j])      
        else:
            f_hat[i] = h_hat[i] / (1j * i * dgamma)       

    
    f = np.zeros_like(x)
    f += f_hat[0].real * dgamma / (2 * np.pi)
    for m in range(1, K - 1):
        f += ((dgamma / np.pi) * f_hat[m] * np.exp(1j * dgamma * m * x)).real
    f += ((dgamma / (2 * np.pi)) * f_hat[K - 1] * np.exp(1j * dgamma * (K - 1) * x)).real

    return f