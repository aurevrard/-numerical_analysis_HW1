import os
import numpy as np
import matplotlib.pyplot as plt
from prony import prony, build_system
from non_uni_sampling import generation,error,ETA, interpolant
from constant_function import generation_fourier,reconstruction,recovery_error,inverse_fourier,piecewise_constant

SEED = 0
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
TAB_DIR = "tables"
os.makedirs(TAB_DIR, exist_ok=True)


def sort_params(omega, c):
    order = np.lexsort((omega.real, omega.imag))
    return omega[order], c[order]



def question_1f(M_values=(2, 5, 10), alpha=1.0, seed=SEED):
    """(1.f) Test exact recovery on a known synthetic signal."""
    rng = np.random.default_rng(seed)
    rows = []
    print("\n=== (1.f) Exact recovery test ===")
    for M in M_values:
        omega = rng.uniform(-alpha, 0, M) + 1j * rng.uniform(-np.pi, np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)

        k = np.arange(2 * M)
        h = np.exp(np.outer(k, omega)) @ c        # h_k = sum_j c_j exp(omega_j k)

        omega_hat, c_hat = prony(h)

        omega, c = sort_params(omega, c)
        omega_hat, c_hat = sort_params(omega_hat, c_hat)

        err_omega = np.max(np.abs(omega - omega_hat))
        err_c = np.max(np.abs(c - c_hat))
        H, _ = build_system(h)                    # only to report the conditioning of H
        print(f"M = {M:2d}: max error on omega = {err_omega:.1e}, on c = {err_c:.1e}, "
              f"cond(H) = {np.linalg.cond(H):.1e}")
        rows.append(f"{M} & {err_omega:.1e} & {err_c:.1e} \\\\")
    with open(os.path.join(TAB_DIR, "tab_1f.tex"), "w") as file:
        file.write("\n".join(rows))

def question_2a(n_trials=50):
    """(2.a) Robustness of Prony's method under non-uniform sampling:
    reconstruction error E versus eta, over several trials."""
    etas = np.array(ETA)
    E = np.zeros((n_trials, len(ETA)))      # one row per trial, one column per eta
    for trial in range(n_trials):
        E[trial] = error(generation()[0])          # new sampling errors eps at each trial

    E_median = np.median(E, axis=0)
    E_q25, E_q75 = np.percentile(E, [25, 75], axis=0)

    # print the median errors (values for the report)
    for eta, e in zip(etas, E_median):
        print(f"eta = {eta:.0e}: median E = {e:.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for trial in range(n_trials):               # all trials, light points
        ax.loglog(etas, E[trial], "o", color="tab:blue", alpha=0.15, markersize=3)
    slope = np.polyfit(np.log10(etas), np.log10(E_median), 1)[0]   # log-log slope
    print(f"log-log slope of median E: {slope:.2f}")
    ax.loglog(etas, E_median, "o-", color="tab:blue",
              label=f"median over trials (slope ≈ {slope:.2f})")
    ax.fill_between(etas, E_q25, E_q75, color="tab:blue", alpha=0.2,
                    label="25th-75th percentile")
   
    ax.set_xlabel(r"sampling perturbation $\eta$")
    ax.set_ylabel(r"reconstruction error $\mathcal{E}$")
    ax.set_title(rf"Prony on non-uniform samples ($M=5$, {n_trials} trials)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_2a.pdf"))
    plt.close(fig)


def question_2b(n_trials=50):
    """(2.b) Interpolation preprocessing: comparison with the direct use of the samples (2.a)."""
    etas = np.array(ETA)
    methods = ["linear", "spline", "lagrange"]
    labels = {"direct": "no interpolation ", "linear": "piecewise linear",
              "spline": "cubic spline", "lagrange": "Lagrange (degree 2M-1)"}

    E = {name: np.zeros((n_trials, len(ETA))) for name in ["direct"] + methods}
    for trial in range(n_trials):
        h, t = generation()                          # same draws for all methods
        E["direct"][trial] = error(h)
        for method in methods:
            E[method][trial] = error(interpolant(h, t, method))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["direct"] + methods:
        E_median = np.median(E[name], axis=0)
        E_q25, E_q75 = np.percentile(E[name], [25, 75], axis=0)
        slope = np.polyfit(np.log10(etas), np.log10(E_median), 1)[0]
        print(f"{labels[name]:25s} median E = {E_median}, slope = {slope:.2f}")
        line, = ax.loglog(etas, E_median, "o-", label=f"{labels[name]} (slope ≈ {slope:.2f})")
        ax.fill_between(etas, E_q25, E_q75, color=line.get_color(), alpha=0.2)

    ax.set_xlabel(r"sampling perturbation $\eta$")
    ax.set_ylabel(r"reconstruction error $\mathcal{E}$")
    ax.set_title(rf"Effect of interpolation preprocessing ($M=5$, {n_trials} trials)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_2b.pdf"))
    plt.close(fig)


M3 = 4
T3 = np.array([0.5, 1.3, 2.0, 3.2, 4.0])   # breakpoints T_1 < ... < T_{M+1}
b3 = np.array([1.0, -0.5, 2.0, 0.8])       # plateau values, b_j != b_{j-1}
K3 = 2 * M3 + 2                            # minimal number of measurements (3.b)
DGAMMA3 = 0.8 * np.pi / T3[-1]             # step size, dgamma <= pi / T_{M+1} (3.b)


def question_3c():
    """(3.c) Recovery of a piecewise constant function with M = 4."""
    h_hat = generation_fourier(T3, b3, K3, DGAMMA3)
    omega_hat, c_hat = prony(h_hat)
    b_r, T_r = reconstruction(omega_hat, c_hat, DGAMMA3)
    err_T, err_b = recovery_error(T3, b3, T_r, b_r)

    print("\n=== (3.c) Recovery of a piecewise constant function (M = 4) ===")
    print("true T:     ", T3)
    print("recovered T:", T_r)
    print("true b:     ", b3)
    print("recovered b:", b_r)
    print(f"max error on T = {err_T:.1e}, max error on b = {err_b:.1e}")

    rows = []
    for j in range(len(T3)):
        if j < len(b3):
            rows.append(f"{j + 1} & {T3[j]:.4f} & {T_r[j]:.12f} & {b3[j]:.4f} & {b_r[j]:.12f} \\\\")
        else:                                     # T_{M+1} has no plateau value
            rows.append(f"{j + 1} & {T3[j]:.4f} & {T_r[j]:.12f} & -- & -- \\\\")
    with open(os.path.join(TAB_DIR, "tab_3c.tex"), "w") as file:
        file.write("\n".join(rows))
    return b_r, T_r




def question_3d(K_values=(10, 20, 50, 100), n_points=1000):
    """(3.d) Trapezoidal inverse Fourier transform versus Prony's reconstruction."""
    print("\n=== (3.d) Trapezoidal inverse Fourier transform versus Prony ===")
    x = np.linspace(0, 5, n_points)
    f_true = piecewise_constant(b3, T3, x)

    # Prony reconstruction with the minimal number of measurements K3 = 2M+2 
    omega_hat, c_hat = prony(generation_fourier(T3, b3, K3, DGAMMA3))
    b_r, T_r = reconstruction(omega_hat, c_hat, DGAMMA3)
    f_prony = piecewise_constant(b_r, T_r, x)
    print(f"Prony (K = {K3}): mean abs error = {np.mean(np.abs(f_prony - f_true)):.1e}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, K in zip(axes.flat, K_values):
        h_hat = generation_fourier(T3, b3, K, DGAMMA3)    # same step, K measurements
        f_trap = inverse_fourier(x, T3, b3, h_hat, DGAMMA3)
        err = np.abs(f_trap - f_true)
        print(f"trapezoidal K = {K:3d}: mean abs error = {np.mean(err):.2e}, max abs error = {np.max(err):.2f}")

        ax.plot(x, f_true, color="black", lw=1.5, label="exact $f$")
        ax.plot(x, f_trap, color="tab:orange", lw=1.2, label=f"trapezoidal, $K={K}$")
        ax.plot(x, f_prony, "--", color="tab:blue", lw=1.2, label=f"Prony, $K={K3}$")
        ax.set_title(rf"$K = {K}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$f(x)$")
    fig.suptitle(rf"Recovery of a piecewise constant function ($M=4$, $\Delta\gamma = 0.8\pi/T_5$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_3d.pdf"))
    plt.close(fig)

if __name__ == "__main__":
    question_1f()
    question_2a()
    question_2b()
    question_3c()
    question_3d()