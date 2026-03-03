import numpy as np

def analytical_shrinkage(X, k=None):
    n, p = X.shape

    # === Handle demeaning ===
    if k is None or (isinstance(k, float) and np.isnan(k)):
        X = X - X.mean(axis=0)
        k = 1

    n_eff = n - k # This is 'n' in the MATLAB code

    # === Sample covariance ===
    sample = (X.T @ X) / n_eff

    # === Eigen-decomposition ===
    # lambda_ are sorted ascending by default in eigh
    lambda_all, u = np.linalg.eigh(sample)
    lambda_all = np.maximum(lambda_all, 1e-12)

    # === Truncate eigenvalues for kernel (Matching MATLAB line 28) ===
    # Only use the min(p, n_eff) non-zero eigenvalues for calculations
    lambda_reduced = lambda_all[max(0, p - n_eff):p]
    
    # === Bandwidth and Kernel constants ===
    h = n_eff ** (-1 / 3)
    L = np.tile(lambda_reduced, (len(lambda_reduced), 1)).T
    H = h * L.T
    x = (L - L.T) / H

    # === Compute ftilde and Hftilde ===
    # Using np.maximum for the (1 - x^2/5) part to match max(..., 0)
    ftemp = (3 / (4 * np.sqrt(5))) * np.maximum(1 - x**2 / 5, 0)
    ftilde = np.mean(ftemp / H, axis=1) / np.pi

    Hftemp = (
        (-3 / (10 * np.pi)) * x
        + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - x**2 / 5)
        * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x) + 1e-15)) # Small epsilon for log
    )
    # Handle the specific case where |x| == sqrt(5)
    mask = np.abs(x) == np.sqrt(5)
    Hftemp[mask] = (-3 / (10 * np.pi)) * x[mask]
    Hftilde = np.mean(Hftemp / H, axis=1)

    # === Shrink eigenvalues ===
    if p <= n_eff:
        dtilde = lambda_reduced / (
            (np.pi * (p / n_eff) * lambda_reduced * ftilde) ** 2
            + (1 - (p / n_eff) - np.pi * (p / n_eff) * lambda_reduced * Hftilde) ** 2
        )
    else:
        # Singular case: p > n_eff
        Hftilde0 = (
            (1 / np.pi)
            * (
                3 / (10 * h**2)
                + (3 / (4 * np.sqrt(5)) / h)
                * (1 - 1 / (5 * h**2))
                * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))
            )
            * np.mean(1 / lambda_reduced)
        )
        dtilde0 = 1 / (np.pi * (p - n_eff) / n_eff * Hftilde0)
        dtilde1 = lambda_reduced / (np.pi**2 * lambda_reduced**2 * (ftilde**2 + Hftilde**2))
        
        # Combine: p-n_eff null eigenvalues + n_eff shrunk non-null eigenvalues
        dtilde = np.concatenate([dtilde0 * np.ones(p - n_eff), dtilde1])

    # === Reconstruct covariance ===
    sigmatilde = u @ np.diag(dtilde) @ u.T

    return sigmatilde, dtilde