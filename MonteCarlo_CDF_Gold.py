
import numpy as np
import matplotlib.pyplot as plt

# === Manual gold parameters ===
S0 = 4000.34          # current gold price (USD per ounce)
mu_annual = 0.10      # expected annual return (10%)
sigma_annual = 0.18   # annual volatility (18%)

# --- Monte Carlo simulation ---
def gold_cdf(S0, mu, sigma, T_years=1, n_sims=100000):
    drift = (mu - 0.5 * sigma**2) * T_years
    diffusion = sigma * np.sqrt(T_years) * np.random.randn(n_sims)
    S_T = S0 * np.exp(drift + diffusion)
    return S_T

# --- Run simulation for today (1 trading session) ---
T_day = 1/252  # 1 trading day
S_T_day = gold_cdf(S0, mu_annual, sigma_annual, T_years=T_day) # type: ignore

# --- Compute daily returns ---
returns_day = S_T_day / S0 - 1  # daily returns in decimal

# --- Bell curve (PDF) for today ---
plt.figure(figsize=(10,6))
plt.hist(returns_day*100, bins=100, density=True, color="lightcoral", edgecolor="gray", alpha=0.7)
plt.title("Bell Curve (PDF) of Gold 1-Day Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Probability Density")
plt.axvline(np.mean(returns_day)*100, color="red", linestyle="--", label=f"Mean ≈ {np.mean(returns_day)*100:.3f}%")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- Threshold probabilities for today ---
thresholds_day = [0.5, 1, 2]  # daily % thresholds
print("\nGold 1-Day Threshold Probabilities:")
print(f"{'Threshold (%)':>12} | {'P(S_T > Threshold)':>18} | {'P(S_T <= Threshold)':>20}")
print("-"*55)
for t in thresholds_day:
    p_up = np.mean(S_T_day > S0*(1+t/100))
    p_down = 1 - p_up
    print(f"{t:12} | {p_up:18.4f} | {p_down:20.4f}")

# === Optional: reusable function for any threshold & horizon ===
def prob_above_threshold(S0, mu, sigma, threshold_pct, T_years=1, n_sims=100000):
    S_T_local = gold_cdf(S0, mu, sigma, T_years=T_years, n_sims=n_sims)
    threshold_val = S0 * (1 + threshold_pct/100)
    prob = np.mean(S_T_local > threshold_val)
    return prob

# --- Example usage ---
p1 = prob_above_threshold(S0, mu_annual, sigma_annual, 1, T_years=T_day) # type: ignore
print(f"\nProbability gold moves >1% today: {p1:.4f}")
