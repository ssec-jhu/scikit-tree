import numpy as np
from scipy.stats import beta

# Parameters of the beta distribution
a = 2  # Shape parameter (a > 0)
b = 4  # Shape parameter (b > 0)

def multi_dimensional_uniform(min_value, max_value, size):
    outcomes = np.random.uniform(min_value, max_value, size)
    return outcomes

def bernoulli_distribution(success_probability):
    #outcomes = np.zeros(len(success_probability), dtype=int)
    #for i, prob in enumerate(success_probability):
    #    outcomes[i] = np.random.choice([0, 1], p=[1 - prob, prob])
    random_numbers = np.random.rand(len(success_probability))
    outcomes = (random_numbers < success_probability).astype(int)
    return outcomes

def normal_distribution(mean, standard_deviation):
    outcomes = np.random.normal(mean, standard_deviation)
    return outcomes

def make_dataset_without_confounding(num_samples, num_features):
    X = multi_dimensional_uniform(0, 1, (num_samples, num_features))
    e = 0.5 * np.ones(num_samples, dtype=int)
    W = bernoulli_distribution(e, num_samples)
    m = 0
    Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
    Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
    tau = Xi1 * Xi2
    mu = m + (W - 0.5) * tau
    sigma = 1
    Y = normal_distribution(mu, sigma)
    return X, Y, W

def make_dataset_without_heterogeneity(num_samples, num_features):
    X = multi_dimensional_uniform(0, 1, (num_samples, num_features))
    beta24 = beta.pdf(X[:,2], a, b)
    e = 0.25 * (1 + beta24)
    W = bernoulli_distribution(e, num_samples)
    m = 2 * X[:,2] - 1
    tau = 0
    mu = m + (W - 0.5) * tau
    sigma = 1
    Y = normal_distribution(mu, sigma)
    return X, Y, W

def make_dataset_with_heterogeneity_and_confounding(num_samples, num_features):
    X = multi_dimensional_uniform(0, 1, (num_samples, num_features))
    beta24 = beta.pdf(X[:,2], a, b)
    e = 0.25 * (1 + beta24)
    W = bernoulli_distribution(e)
    m = 2 * X[:,2] - 1
    Xi1 = 1 + 1 / (1 + np.exp(-20 * (X[:,0]-1/3)))
    Xi2 = 1 + 1 / (1 + np.exp(-20 * (X[:,1]-1/3)))
    tau = Xi1 * Xi2
    mu = m + (W - 0.5) * tau
    sigma = 1
    Y = normal_distribution(mu, sigma)
    return X, Y, W

## Usage example
#num_samples = 10
#num_features = 3
#result = make_dataset_with_heterogeneity_and_confounding(num_samples, num_features)
#print(result)
#result = make_dataset_with_heterogeneity_and_confounding(num_samples, num_features)
#print(result)
#result = make_dataset_with_heterogeneity_and_confounding(num_samples, num_features)
#print(result)