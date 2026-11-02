import os
import numpy as np
import matplotlib.pyplot as plt
from prony import prony
import non_uni_sampling
from non_uni_sampling import generation,error,ETA, interpolant

SEED = 0
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def sort_params(omega, c):
    order = np.lexsort((omega.real, omega.imag))
    return omega[order], c[order]


def question_1f(M_values=(2, 5, 10), alpha=1.0, seed=SEED):
    """(1.f) Test exact recovery on a known synthetic signal."""
    rng = np.random.default_rng(seed)
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
        print(f"M = {M:2d}: max error on omega = {err_omega:.2e}, on c = {err_c:.2e}")

def question_2a(n_trials=50):
    """(2.a) Robustness of Prony's method under non-uniform sampling:
    reconstruction error E versus eta, over several trials."""
    etas = np.array(ETA)
    E = np.zeros((n_trials, len(ETA)))      # one row per trial, one column per eta
    for t in range(n_trials):
        E[t] = error(generation()[0])          # new sampling errors eps at each trial

    E_median = np.median(E, axis=0)
    E_q25, E_q75 = np.percentile(E, [25, 75], axis=0)

    # print the median errors (values for the report)
    for eta, e in zip(etas, E_median):
        print(f"eta = {eta:.0e}: median E = {e:.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for t in range(n_trials):               # all trials, light points
        ax.loglog(etas, E[t], "o", color="tab:blue", alpha=0.15, markersize=3)
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


def question_3c():
    # M = 4, choix des b_j et T_j, K et Delta_gamma d'après 3.b
    # vérification de la reconstruction
    ...


def question_3d():
    # trapèzes pour K = 10, 20, 50, 100 vs Prony -> figures/fig_3d.pdf
    ...


if __name__ == "__main__":
    question_1f()
    question_2a()
    question_2b()
    question_3c()
    question_3d()