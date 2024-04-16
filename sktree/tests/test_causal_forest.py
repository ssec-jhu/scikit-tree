import unittest
import pytest
from econml.dml import CausalForestDML
from econml.grf import CausalForest

import numpy as np
from scipy.stats import beta


def get_base_config():
    return {
            'model_y': 'forest',  # Use random forest for the outcome model
            'model_t': 'forest',  # Use random forest for the treatment model
            'discrete_treatment': False,
            'n_estimators': 2000,  # Number of trees in the forest
            'min_samples_leaf': 10,  # Minimum number of samples in leaf nodes
            'min_impurity_decrease': 0.001,  # Minimum impurity decrease for splitting nodes
            'verbose': 0,  # Verbosity level (0: no output, 1: progress bar, 2: detailed output)
            'n_jobs': None,  # Number of jobs to run in parallel (-1: use all available cores)
            'fit_intercept': False,
            'criterion': "het"
        }

def get_causal_data(confounding, heterogeneity, n_samples, n_features):
    X = np.random.uniform(0, 1, size=(n_samples, n_features))
    if confounding:
        beta24 = beta.pdf(X[:,2], 2, 4)
        e = 0.25 * (1 + beta24)
        m = 2 * X[:,2] - 1
    else:
        e = 0.5 * np.ones(n_samples, dtype=int)
        m = 0
    W = np.random.binomial(1, e, n_samples)
    if heterogeneity:
        Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
        Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
        T = Xi1 * Xi2
    else:
        T = np.zeros(n_samples, dtype=int)
    y = np.random.normal(m + (W - 0.5) * T, 1)
    return (X, W, y, e, m, T)


#@pytest.mark.parametrize("criterion", ["het", "mse"])
def test_grf_perf():
    """Testing accuracy of various GRFs"""
    # MSE truth data
    mse_truth = np.array([
        [1.37, 6.48, 0.85, 0.87],
        [0.63, 6.23, 0.58, 0.59],
        [2.05, 8.02, 0.92, 0.93],
        [0.71, 7.61, 0.52, 0.52],
        [0.81, 0.16, 1.12, 0.27],
        [0.68, 0.10, 0.80, 0.20],
        [0.90, 0.13, 1.17, 0.17],
        [0.77, 0.09, 0.95, 0.11],
        [4.51, 7.67, 1.92, 0.91],
        [2.45, 7.94, 1.51, 0.62],
        [5.93, 8.68, 1.92, 0.93],
        [3.54, 8.61, 1.55, 0.57]
    ])
    cntr1 = 0
    for confounding, heterogeneity in [(False, True), (True, False), (True, True)]:
        cntr2 = 0
        for p, n in [(10, 800), (10, 1600), (20, 800), (20, 1600)]:
            print("Confounding = ", confounding, ", Heterogeneity = ", heterogeneity, ", p = ", p, ", n = ", n)
            X, W, y, e, m, T = get_causal_data(confounding, heterogeneity, n, p)
            
            # Random Forest without centering
            forest = CausalForestDML(**get_base_config()).fit(y, T=W, X=X) # Fit the model to the data
            replications = 60
            GRF = np.zeros(replications)
            for cntr in range(0, replications):
                X_rep, W_rep, y_rep, e_rep, m_rep, T_rep = get_causal_data(confounding, heterogeneity, 1000, p)
                GRF[cntr] = forest.score(y_rep, T=W_rep, X=X_rep) # Calculate mean squared error
            GRF_mean = np.mean(GRF)    
            ground_truth = mse_truth[4*cntr1+cntr2, 2]
            print("Mean Squared Error: ", GRF_mean, " Truth value: ", ground_truth)
            #np.testing.assert_allclose(mse, ground_truth, rtol=0.5, atol=1.0)
            
            # Random Forest with centering
            y = y - (m + (e - 0.5) * T)
            W = W - e
            forest = CausalForestDML(**get_base_config()).fit(y, T=W, X=X) # Fit the model to the data
            GRF = np.zeros(replications)
            for cntr in range(0, replications):
                X_rep, W_rep, y_rep, e_rep, m_rep, T_rep = get_causal_data(confounding, heterogeneity, 1000, p)
                y_rep = y_rep - (m_rep + (e_rep - 0.5) * T_rep)
                W_rep = W_rep - e_rep
                GRF[cntr] = forest.score(y_rep, T=W_rep, X=X_rep) # Calculate mean squared error
            GRF_mean = np.mean(GRF)    
            ground_truth = mse_truth[4*cntr1+cntr2, 3]
            print("Mean Squared Error: ", GRF_mean, " Truth value: ", ground_truth)
            #np.testing.assert_allclose(mse, ground_truth, rtol=0.5, atol=1.0)
            cntr2 += 1
        cntr1 += 1
        
test_grf_perf()