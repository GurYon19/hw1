###### Your ID ######
# ID1: 
# ID2: 
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 1 ###

def find_sample_size_binom(defective_percent=0.03, target_probability=0.85, min_defective=1):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    n = min_defective
    # 1 - stats.binom.cdf(k, n, p) = P(X >= k)

    while (1 - stats.binom.cdf(min_defective-1, n, defective_percent)) < target_probability:
        n += 1
    return n


def find_sample_size_nbinom(defective_percent=0.03, target_probability=0.85, min_defective= 1):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    n = min_defective
    #nbinom.cdf(k, r, p), we get P(X <= k) where X is the number of failures before getting the r-th success
    # in our case r == 1 , success if 
    # nbinom is the negative binomial distribution
    while stats.nbinom.cdf(n-min_defective, min_defective, defective_percent) < target_probability:
        n += 1
    return n

def compare_q1():
    res1= find_sample_size_nbinom(0.1, 0.9, 5)
    res2= find_sample_size_nbinom(0.3, 0.9, 15)
    return (res1, res2)


def same_prob(defective_percent1=0.1, defective_percent2=0.3, target_probability=0.9, min_defective1=5, min_defective2=15):
    # Start from the maximum of the two minimum defective counts
    # This ensures both probabilities are > 0
    n = max(min_defective1, min_defective2)
    max_iterations = 1000  # Safety limit to avoid infinite loop
    
    for _ in range(max_iterations):
        # Calculate probability for scenario 1: P(X1 >= min_defective1) where X1 ~ Binom(n, p1)
        prob1 = 1 - stats.binom.cdf(min_defective1 - 1, n, defective_percent1)
        
        # Calculate probability for scenario 2: P(X2 >= min_defective2) where X2 ~ Binom(n, p2)
        prob2 = 1 - stats.binom.cdf(min_defective2 - 1, n, defective_percent2)
        
        # Check if probabilities are approximately equal
        if np.isclose(prob1, prob2, atol=1e-2):
            return n
        
        n += 1
    
    # If no solution found within max_iterations
    return None

### Question 2 ###

def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None):
    """
    Create k experiments where X is sampled. Calculate the empirical centralized third moment of Y based 
    on your k experiments.
    """
    samples = stats.multinomial.rvs(n, p, size=k, random_state=seed)
    y_samples = samples[:, 1:4].sum(axis=1)
    centered = y_samples - y_samples.mean()
    empirical_moment = np.mean(centered ** 3)

    return empirical_moment

def class_moment():
    n = 20
    p = 0.1 + 0.1 + 0.1  # Sum of probabilities for X2, X3, X4
    # The centralized third moment for Binomial(n, p) is n * p * (1-p) * (1-2p)
    moment = n * p * (1 - p) * (1 - 2 * p)
    return moment

def plot_moments():
    moments = []
    for _ in range(1000):
        moments.append(empirical_centralized_third_moment())
    
    theoretical_val = class_moment()
    
    plt.figure(figsize=(10, 6))
    plt.hist(moments, bins=30, alpha=0.7, label='Empirical Moments')
    plt.axvline(theoretical_val, color='r', linestyle='dashed', linewidth=2, label='Theoretical Moment')
    plt.title('Histogram of Empirical Centralized Third Moments')
    plt.xlabel('Moment Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    dist_var = np.var(moments)
    return dist_var
    
def plot_moments_smaller_variance():
    # To reduce variance, we increase the sample size k in the experiment
    # Default k was 100, let's increase it to 1000
    k_new = 1000
    
    moments = []
    for _ in range(1000):
        moments.append(empirical_centralized_third_moment(k=k_new))
        
    theoretical_val = class_moment()
    
    plt.figure(figsize=(10, 6))
    plt.hist(moments, bins=30, alpha=0.7, color='green', label=f'Empirical Moments (k={k_new})')
    plt.axvline(theoretical_val, color='r', linestyle='dashed', linewidth=2, label='Theoretical Moment')
    plt.title(f'Histogram of Empirical Centralized Third Moments (Reduced Variance, k={k_new})')
    plt.xlabel('Moment Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    dist_var = np.var(moments)
    return dist_var
    
    
### Question 3 ###

def NFoldConv(P, n):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables, 
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """
    # Extract values and probabilities
    values = P[0]
    probs = P[1]
    
    # Current distribution Q starts as P
    # We'll represent it as a dictionary {value: prob} for easy merging
    current_dist = {val: prob for val, prob in zip(values, probs)}
    
    for _ in range(n - 1):
        new_dist = {}
        # Convolve current_dist with P
        for val_q, prob_q in current_dist.items():
            for val_p, prob_p in zip(values, probs):
                new_val = val_q + val_p
                new_prob = prob_q * prob_p
                
                if new_val in new_dist:
                    new_dist[new_val] += new_prob
                else:
                    new_dist[new_val] = new_prob
        current_dist = new_dist
    
    # Convert back to 2D numpy array [[values], [probabilities]]
    # Sort by values for cleaner output
    sorted_items = sorted(current_dist.items())
    Q_values = [item[0] for item in sorted_items]
    Q_probs = [item[1] for item in sorted_items]
    
    Q = np.array([Q_values, Q_probs])
    return Q
    
def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """
    values = P[0]
    probs = P[1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(values, probs, width=0.8, alpha=0.7, edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.show()


### Qeustion 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).
    
    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    # Create an array of even numbers from 0 to n
    k_values = np.arange(0, n + 1, 2)
    # Sum the PMF for all even k
    prob = stats.binom.pmf(k_values, n, p).sum()
    return prob

def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.
    
    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    prob = 0.5 * (1 + (1 - 2 * p) ** n)
    
    proof = """
    PROOF: P(X is even) = (1 + (1-2p)^n) / 2
    
    Step 1: Start with the Binomial Theorem
    For X ~ Binom(n, p), let q = 1-p. We know:
        sum(k=0 to n) of: C(n,k) * p^k * q^(n-k) = (p + q)^n = 1   ...(Equation 1)
    
    Step 2:  Trick - Replace p with -p
        sum(k=0 to n) of: C(n,k) * (-p)^k * q^(n-k) = (q - p)^n = (1-2p)^n   ...(Equation 2)
    
    Since (-p)^k = (-1)^k * p^k, we can rewrite Equation 2 as:
        sum(k=0 to n) of: C(n,k) * (-1)^k * p^k * q^(n-k) = (1-2p)^n
    
    Step 3: Add Equations 1 and 2
        sum(k=0 to n) of: C(n,k) * p^k * q^(n-k) * [1 + (-1)^k] = 1 + (1-2p)^n
    
    Step 4: The Magic of [1 + (-1)^k]
        - When k is EVEN: (-1)^k = 1, so [1 + (-1)^k] = 2
        - When k is ODD:  (-1)^k = -1, so [1 + (-1)^k] = 0
    
    Therefore, odd terms vanish and even terms are doubled:
        2 * sum(k even) of: C(n,k) * p^k * q^(n-k) = 1 + (1-2p)^n
        2 * P(X is even) = 1 + (1-2p)^n
    
    Step 5: Final Result
        P(X is even) = (1 + (1-2p)^n) / 2
    
    Notation: C(n,k) = binomial coefficient = "n choose k"
    """
    print(proof)
    return prob

### Question 5 ###

def three_RV(values, joint_probs):
    """
 
    Input:          
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z) 
    """
    # Collect all values and probabilities from the multidimensional arrays
    all_vals = []
    all_probs = []
    for idx in np.ndindex(values.shape):
        all_vals.append(values[idx])
        all_probs.append(joint_probs[idx])
    
    # Extract unique values for each RV
    x_vals = np.unique([val[0] for val in all_vals])
    y_vals = np.unique([val[1] for val in all_vals])
    z_vals = np.unique([val[2] for val in all_vals])
    
    # Calculate marginal probabilities 
    # P(X=x) = sum over all y,z of P(X=x, Y=y, Z=z)
    p_x = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[0] == x) for x in x_vals])
    p_y = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[1] == y) for y in y_vals])
    p_z = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[2] == z) for z in z_vals])
    
    # Calculate E[X], E[Y], E[Z]
    E_X = np.sum(x_vals * p_x)
    E_Y = np.sum(y_vals * p_y)
    E_Z = np.sum(z_vals * p_z)
    
    # Calculate E[X^2], E[Y^2], E[Z^2]
    E_X2 = np.sum(x_vals**2 * p_x)
    E_Y2 = np.sum(y_vals**2 * p_y)
    E_Z2 = np.sum(z_vals**2 * p_z)
    
    # Calculate Var(X), Var(Y), Var(Z)
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2
    
    # Calculate E[XY], E[XZ], E[YZ] from joint distribution
    E_XY = sum(val[0] * val[1] * prob for val, prob in zip(all_vals, all_probs))
    E_XZ = sum(val[0] * val[2] * prob for val, prob in zip(all_vals, all_probs))
    E_YZ = sum(val[1] * val[2] * prob for val, prob in zip(all_vals, all_probs))
    
    # Calculate covariances
    Cov_XY = E_XY - E_X * E_Y
    Cov_XZ = E_XZ - E_X * E_Z
    Cov_YZ = E_YZ - E_Y * E_Z
    
    # We saw the formula in class
    # Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z) + 2*Cov(X,Y) + 2*Cov(X,Z) + 2*Cov(Y,Z)

    v = Var_X + Var_Y + Var_Z + 2*Cov_XY + 2*Cov_XZ + 2*Cov_YZ
    
    return v

def three_RV_pairwise_independent(values, joint_probs):
    """
 
    Input:          
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z)
    """
    # For pairwise independent RVs: Cov(X,Y) = Cov(X,Z) = Cov(Y,Z) = 0
    # Therefore: Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z)
    
    # Collect all values and probabilities from the multidimensional arrays
    all_vals = []
    all_probs = []
    for idx in np.ndindex(values.shape):
        all_vals.append(values[idx])
        all_probs.append(joint_probs[idx])
    
    # Extract unique values for each RV
    x_vals = np.unique([val[0] for val in all_vals])
    y_vals = np.unique([val[1] for val in all_vals])
    z_vals = np.unique([val[2] for val in all_vals])
    
    # Calculate marginal probabilities
    p_x = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[0] == x) for x in x_vals])
    p_y = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[1] == y) for y in y_vals])
    p_z = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[2] == z) for z in z_vals])
    
    # Calculate E[X], E[Y], E[Z]
    E_X = np.sum(x_vals * p_x)
    E_Y = np.sum(y_vals * p_y)
    E_Z = np.sum(z_vals * p_z)
    
    # Calculate E[X^2], E[Y^2], E[Z^2]
    E_X2 = np.sum(x_vals**2 * p_x)
    E_Y2 = np.sum(y_vals**2 * p_y)
    E_Z2 = np.sum(z_vals**2 * p_z)
    
    # Calculate Var(X), Var(Y), Var(Z)
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2
    
    # For pairwise independent RVs, variance of sum is sum of variances
    v = Var_X + Var_Y + Var_Z
    
    return v

def is_pairwise_collectively(values, joint_probs):
    """

    Input:
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    TRUE or FALSE
    """
    # Collect all values and probabilities from the multidimensional arrays
    # values and joint_probs have the same shape
    all_vals = []
    all_probs = []
    for idx in np.ndindex(joint_probs.shape):  # Use joint_probs.shape
        all_vals.append(values[idx])
        all_probs.append(joint_probs[idx])
    
    # Extract unique values for each RV
    x_vals = np.unique([val[0] for val in all_vals])
    y_vals = np.unique([val[1] for val in all_vals])
    z_vals = np.unique([val[2] for val in all_vals])
    
    # Calculate marginal probabilities
    p_x = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[0] == x) for x in x_vals])
    p_y = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[1] == y) for y in y_vals])
    p_z = np.array([sum(prob for val, prob in zip(all_vals, all_probs) if val[2] == z) for z in z_vals])
    
    # Check if P(X=x, Y=y, Z=z) = P(X=x) * P(Y=y) * P(Z=z) for all combinations
    tolerance = 1e-10
    
    for val, joint_prob in zip(all_vals, all_probs):
        x, y, z = val
        
        # Find indices
        x_idx = np.where(x_vals == x)[0][0]
        y_idx = np.where(y_vals == y)[0][0]
        z_idx = np.where(z_vals == z)[0][0]
        
        # Calculate product of marginals
        product_of_marginals = p_x[x_idx] * p_y[y_idx] * p_z[z_idx]
        
        # Check if they're equal (within tolerance)
        if not np.isclose(joint_prob, product_of_marginals, atol=tolerance):
            return False
    
    return True


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """
    
    pass











    
    
    
    
    
    