import numpy as np
from numpy.polynomial import polynomial as P


def build_system(h):
    #input: a vector of measurements {h_k}_{k=0}^{2M−1}
    #output: matrix H(MxM) and vector b(Mx1) 
    """build the system Hp=b"""
    M = len(h) // 2
    H = np.zeros((M, M), dtype=complex)
    for m in range(M):
        for k in range(M):
            H[m, k] = h[m + k]
    b = np.zeros(M, dtype=complex)
    for m in range(M):
        b[m] = -h[M + m]                       # b[m] = -h_{M+m}
    return H, b


def prony(h):
    #input: a vector of measurements {h_k}_{k=0}^{2M−1}
    #output: the recovered vectors {ω_j}_{j=1}^M and  {c_j}_{j=1}^M

    h = np.asarray(h, dtype=complex)
    if len(h) % 2 != 0:
        raise ValueError("there must be an even number of measurements (2M).")
    M = len(h) // 2

    H, b = build_system(h)
    p = np.linalg.solve(H, b)

    
    z = P.polyroots(np.append(p, 1.0))#give the roots of the polynomiale see eq.(4) of the pdf 

    omega = np.log(z)

    V = np.vander(z, M, increasing=True).T   # V[k, j] = z_j^k where j=0,...,M-1 and k=0,...,M-1
    c = np.linalg.solve(V, h[:M]) #solve the linear system for M first measurements

    return omega, c


