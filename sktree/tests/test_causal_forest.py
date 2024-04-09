import unittest
import pytest
from econml.dml import CausalForestDML
import numpy as np
from scipy.stats import beta


def test_grf_perf(self):
        """Testing accuracy of various GRFs"""
        
        #import matplotlib.pyplot as plt
        ## Parameters for the beta distribution
        #alpha = 2
        #beta_param = 4
        ## Generate x values for plotting
        #x = np.linspace(0, 1, 1000)
        ## Compute the beta density function
        #pdf = beta.pdf(x, alpha, beta_param)
        ## Plot the beta density function
        #plt.figure(figsize=(8, 6))
        #plt.plot(x, pdf, label=f'Beta({alpha}, {beta_param}) Density Function')
        #plt.xlabel('x')
        #plt.ylabel('Probability Density')
        #plt.title('Beta Density Function')
        #plt.legend()
        #plt.grid(True)
        #plt.show()

        # Parameters of the beta distribution
        a, b = 2, 4  # Beta function parameters
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
                X = np.random.uniform(0, 1, size=(n,p))
                if confounding:
                    beta24 = beta.pdf(X[:,2], a, b)
                    e = 0.25 * (1 + beta24)
                    #W = bernoulli_distribution(e)
                    m = 2 * X[:,2] - 1
                else:
                    e = 0.5 * np.ones(n, dtype=int)
                    #W = bernoulli_distribution(e)
                    m = 0
                W = np.random.binomial(1, e, n)
                if heterogeneity:
                    Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
                    Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
                    T = Xi1 * Xi2
                else:
                    T = 0 * np.ones(n, dtype=int)
                y = np.random.normal(m + (W - 0.5) * T, 1)
                
                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
                #est = RegressionForest(
                #    max_depth=None,
                #    n_estimators=2000, # Number of trees in the forest
                #    min_samples_leaf=10,  # Minimum number of samples in leaf nodes
                #    min_impurity_decrease=0.001,  # Minimum impurity decrease for splitting nodes
                #    verbose=0,  # Verbosity level
                #    n_jobs=None,  # Number of jobs to run in parallel (-1: use all available cores)
                #    random_state=12345
                #)
                #est.fit(X_train, y_train) # Fit the model to your data
                #y_pred = est.predict(X_test) # Predict the target variable on the test set
                #mse = mean_squared_error(y_test, y_pred) # Calculate mean squared error
                #ground_truth = mse_truth[4*cntr1+cntr2, 0]
                #print("Mean Squared Error: ", mse, " Truth value: ", ground_truth)
                ##np.testing.assert_allclose(mse, ground_truth, rtol=0.5, atol=1.0)
                
                # Random Forest without centering    
                est = CausalForestDML(
                    model_y='forest',  # Use random forest for the outcome model
                    model_t='forest',  # Use random forest for the treatment model
                    #featurizer=NoFeaturizer(),  # Disable featurization (local centering)
                    discrete_treatment=False,
                    cv=2,
                    n_estimators=2000,  # Number of trees in the forest
                    min_samples_leaf=10,  # Minimum number of samples in leaf nodes
                    min_impurity_decrease=0.001,  # Minimum impurity decrease for splitting nodes
                    verbose=0,  # Verbosity level (0: no output, 1: progress bar, 2: detailed output)
                    n_jobs=None,  # Number of jobs to run in parallel (-1: use all available cores)
                    fit_intercept=False,
                    criterion="het"
                )
                est.fit(y, T=W, X=X) # Fit the model to your data
                
                GRF = np.zeros(60)
                for cntr in range(0, 60):
                    X = np.random.uniform(0, 1, size=(1000,p))
                    if confounding:
                        beta24 = beta.pdf(X[:,2], a, b)
                        e = 0.25 * (1 + beta24)
                        m = 2 * X[:,2] - 1
                    else:
                        e = 0.5 * np.ones(1000, dtype=int)
                        m = 0
                    W = np.random.binomial(1, e, 1000)
                    if heterogeneity:
                        Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
                        Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
                        T = Xi1 * Xi2
                    else:
                        T = 0 * np.ones(1000, dtype=int)
                    y = np.random.normal(m + (W - 0.5) * T, 1)
                    GRF[cntr] = est.score(y, T=W, X=X) # Calculate mean squared error
                    
                GRF_mean = np.mean(GRF)    
                ground_truth = mse_truth[4*cntr1+cntr2, 2]
                print("Mean Squared Error: ", GRF_mean, " Truth value: ", ground_truth)
                #np.testing.assert_allclose(mse, ground_truth, rtol=0.5, atol=1.0)
                
                # Random Forest with centering
                est = CausalForestDML(
                    model_y='forest',  # Use random forest for the outcome model
                    model_t='forest',  # Use random forest for the treatment model
                    #featurizer=None,  # Leave local centering enabled
                    discrete_treatment=False,
                    cv=2,
                    n_estimators=2000,  # Number of trees in the forest
                    min_samples_leaf=10,  # Minimum number of samples in leaf nodes
                    min_impurity_decrease=0.001,  # Minimum impurity decrease for splitting nodes
                    verbose=0,  # Verbosity level (0: no output, 1: progress bar, 2: detailed output)
                    n_jobs=None,  # Number of jobs to run in parallel (-1: use all available cores)
                    fit_intercept=False,
                    criterion="het"
                )
                y = y - (W - 0.5) * T
                W = W - e
                est.fit(y, T=W, X=X) # Fit the model to your data
                
                GRF = np.zeros(60)
                for cntr in range(0, 60):
                    X = np.random.uniform(0, 1, size=(1000,p))
                    if confounding:
                        beta24 = beta.pdf(X[:,2], a, b)
                        e = 0.25 * (1 + beta24)
                        m = 2 * X[:,2] - 1
                    else:
                        e = 0.5 * np.ones(1000, dtype=int)
                        m = 0
                    W = np.random.binomial(1, e, 1000)
                    if heterogeneity:
                        Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
                        Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
                        T = Xi1 * Xi2
                    else:
                        T = 0 * np.ones(1000, dtype=int)
                    y = np.random.normal(m + (W - 0.5) * T, 1)

                    y = y - (W - 0.5) * T
                    W = W - e
                    GRF[cntr] = est.score(y, T=W, X=X) # Calculate mean squared error
                    
                GRF_mean = np.mean(GRF)    
                ground_truth = mse_truth[4*cntr1+cntr2, 3]
                print("Mean Squared Error: ", GRF_mean, " Truth value: ", ground_truth)
                #np.testing.assert_allclose(mse, ground_truth, rtol=0.5, atol=1.0)
                
                cntr2 += 1
            cntr1 += 1
