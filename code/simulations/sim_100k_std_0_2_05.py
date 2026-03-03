# For server not to crash!
import os
# Force each task to stay on 1 thread so they dont fight for the 32/128 slots
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import math
from joblib import Parallel, delayed
from tqdm import tqdm
#from sklearn.covariance import LedoitWolf #1
from sklearn.covariance import ledoit_wolf
from ledoit2004b import cov1Para #2
from ledoit2020 import analytical_shrinkage #3

# %%
np.set_printoptions(suppress=True, precision=6)
warnings.simplefilter("ignore", UserWarning)



def ledoit2020(X):
    Sigma_hat, _ = analytical_shrinkage(X)
    return Sigma_hat


def get_mu_vector_fast(m, m_star):
    indices = np.arange(1, m + 1) # Create indices 1, 2, ..., m
    
    mu = np.where(indices <= m_star, 0, 0.01 * (indices - m_star)) # If index <= m_star, mu=0. Else, 0.01 * (index - m_star)
    
    return mu



# V2 -- single factor + iid noise

def generate_data(n, m, m_star, s_common=0.2, s_idio=0.5):
    
    mu = get_mu_vector_fast(m, m_star)
    
    xi_common = np.random.normal(0, s_common, size=(n, 1)) # xi_i: common shock for obs (n, 1)
    xi_idio = np.random.normal(0, s_idio, size=(n, m)) # xi_ij: shock specific for every obs and model (n, m)
    
    # matrix broadcasting
    X = xi_common + xi_idio + mu
    
    return X, mu


def mcs_procedure(X, mu_hat, n, alpha=0.05, sigma_func=None):

    m0 = len(mu_hat) # num of models
    M = list(range(m0))  # initial list of models

    p_values = []
    
    while True: # keep going until a stopping rule is met
        m = len(M)
        if m == 1:
            # Only one model left
            p_values.append(1.0) # add '1' as a convention
            break
        
        # Subset cov matrix estimates for current M (but no re-calculation of it)
        mu_M = mu_hat[M] # is the mean(X, axis=0)

        X_M = X[:, M] # slice our data X
        
        if sigma_func is None:
            Sigma_hat = np.cov(X_M, rowvar=False, bias=False)  # sample covariance
        elif sigma_func == '2004a':
            #igma_hat = LedoitWolf().fit(X_M).covariance_     #ledoit_wolf_2004a(X)  # shrinkage 1
            Sigma_hat, _ = ledoit_wolf(X_M)   #ledoit_wolf_2004a(X)  # shrinkage 1 (this returns tuple but faster than above)
        elif sigma_func== '2004b':
            Sigma_hat = cov1Para(pd.DataFrame(X_M)).to_numpy()    #ledoit_wolf_2004b(X) # shrinkage 2
        elif sigma_func== '2020':
            Sigma_hat = ledoit2020(X_M)
        
        # Construct contrast matrix A: (m-1) × m (i.e. the difference operator)
        A = np.hstack([np.ones((m-1, 1)), -np.eye(m-1)]) # stacking horizontally vector of 1s and a negative Identity matrix
        
        # Wald statistic
        W =n * (A @ mu_M).T @ np.linalg.inv(A @ Sigma_hat @ A.T) @ (A @ mu_M)
        
        # get p-value from the chi^2(m-1) distrib
        pval = 1 - chi2.cdf(W, df=m-1) # I get global null on the whole (sub)set M
        p_values.append(pval)
        
        # Decision
        if pval >= alpha:
            break  # stop, current M is confidence set
        else:
            # Eliminate model with largest estimated mean
            worst_in_M = np.argmax(mu_M)  # index within M
            M.pop(worst_in_M) # removing worst m from models list "M"
    
    return set(M), p_values





# grid values
m_values = [10, 50, 250]
n_values = [20, 200, 2000]
p_list = [0.05, 0.10, 0.20] # ratio of model* sets
R = 100000
alpha_fixed = 0.05 # The significance level for the test
sigma_funcs = [None, '2004a', '2004b', '2020'] # shrinkage methods


# ### Parallel Computing with joblib

def run_single_replica(n, m, m_star_count, m_star_indices, non_best_indices, alpha, s):
    try:
        # Data generation
        #X, _ = generate_nonlinear_functions(n, m, m_star_count)
        X, _ = generate_data(n, m, m_star_count)
        mu_hat = np.mean(X, axis=0)
        
        # Run MCS
        final_set, _ = mcs_procedure(X, mu_hat, n, alpha=alpha, sigma_func=s)
        
        # Calculate metrics for this one run
        is_covered = 1 if m_star_indices.issubset(final_set) else 0
        size = len(final_set)
        fpr = 0
        if len(non_best_indices) > 0:
            fpr = len(final_set.intersection(non_best_indices)) / len(non_best_indices) # fp/fp+tn
        
        return is_covered, size, fpr, True # True to indicate success
    
    except Exception as e:
        return 0, 0, 0, False # False to indicate failed replica





# V2 with failure handling

results = []

for s in tqdm(sigma_funcs, desc="Cov Matrix Method"):
    for m in tqdm(m_values, desc="M models", leave=False):
        for n in tqdm(n_values, desc='N Runs', leave=False):
            for p in p_list:
                m_star_count = math.ceil(p * m)
                m_star_indices = set(range(m_star_count))
                non_best_indices = set(range(m)) - m_star_indices

                # Run R replicas in parallel
                parallel_results = Parallel(n_jobs=100)(  #n_jobs=-1 uses all available cpu cores      
                    delayed(run_single_replica)(n, m, m_star_count, m_star_indices, non_best_indices, alpha_fixed, s) 
                    for _ in range(R)
                )
                
                # Filter only successes
                success_results = [res for res in parallel_results if res[3] is True]
                num_successes = len(success_results)

                if num_successes > 0:
                    coverage_successes = sum(res[0] for res in success_results)
                    total_mcs_size = sum(res[1] for res in success_results)
                    total_fpr = sum(res[2] for res in success_results)

                    results.append({
                        'Method': "Sample Cov" if s is None else s,
                        'm': m, 'n': n, 'p': p,
                        'Coverage': coverage_successes / num_successes,
                        'MCS Size': total_mcs_size / num_successes,
                        'FPR': total_fpr / num_successes,
                        'Failures': R - num_successes # to monitor instability
                    })
                else:
                    # if all replicas fail (very unlikely)
                    results.append({
                        'Method': "Sample Cov" if s is None else s,
                        'm': m, 'n': n, 'p': p,
                        'Coverage': np.nan, 'MCS Size': np.nan, 'FPR': np.nan,
                        'Failures': R
                    })
                
                print(f"\n[FINITO] Metodo: {s} | m: {m} | n: {n} | p: {p}")
                print(f"Result -> Coverage: {results[-1]['Coverage']:.4f}, Failures: {results[-1]['Failures']}") # with [-1] i get last res

df_results_parall = pd.DataFrame(results)

df_results_parall.to_csv('df_results_run_100k.csv')

