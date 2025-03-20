import numpy as np
import pandas as pd
import os

# Function to calculate forward equations for EZ Diffusion
def forward_equations(a, v, t):
    """Compute predicted summary statistics."""
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * (1 - y) / (1 + y)
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
    return R_pred, M_pred, V_pred

# Function to calculate inverse equations for parameter estimation
def inverse_equations(R_obs, M_obs, V_obs):
    """Estimate parameters from observed statistics."""
    if R_obs <= 0 or R_obs >= 1:  # Prevent invalid log calculations
        return None, None, None
    
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    return a_est, v_est, t_est

# Function to generate observed noisy data
def simulate_observed_data(R_pred, M_pred, V_pred, N):
    """Simulate observed accuracy, mean RT, and variance."""
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))
    return R_obs, M_obs, V_obs

# Run the simulation
N_values = [10, 40, 4000]
results = []

for N in N_values:
    for _ in range(1000):
        # Generate random parameters
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t_true = np.random.uniform(0.1, 0.5)

        # Compute expected summary statistics
        R_pred, M_pred, V_pred = forward_equations(a_true, v_true, t_true)

        # Generate simulated observed data
        R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)

        # Recover parameters
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)

        # Store results
        results.append([N, a_true, v_true, t_true, a_est, v_est, t_est])

# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

# Save results
df = pd.DataFrame(results, columns=["N", "a_true", "v_true", "t_true", "a_est", "v_est", "t_est"])
df.to_csv("results/sim_results.csv", index=False)

print("Simulation complete. Results saved in results/sim_results.csv.")

