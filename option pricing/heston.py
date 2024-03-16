import numpy as np
from scipy.stats import norm

class HestonOptionPricing:
    def __init__(self, S, K, T, r, kappa, theta, sigma, rho, v0, num_steps=100, option_type='European'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.kappa = kappa  # Mean reversion rate of volatility
        self.theta = theta  # Long-term mean of volatility
        self.sigma = sigma  # Volatility of volatility (volatility of variance)
        self.rho = rho  # Correlation between asset price and volatility
        self.v0 = v0  # Initial variance
        self.num_steps = num_steps  # Number of steps in the Monte Carlo simulation
        self.option_type = option_type  # Type of option: 'European', 'American', or 'Asian'

    def simulate_paths(self):
        dt = self.T / self.num_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize arrays to store simulated paths
        S_paths = np.zeros((self.num_steps + 1, self.num_steps + 1))
        v_paths = np.zeros_like(S_paths)

        # Set initial conditions
        S_paths[0, 0] = self.S
        v_paths[0, 0] = self.v0

        for i in range(1, self.num_steps + 1):
            z_S = np.random.normal(size=(self.num_steps + 1,))
            z_v = self.rho * z_S + np.sqrt(1 - self.rho**2) * np.random.normal(size=(self.num_steps + 1,))

            for j in range(i + 1):
                v_paths[i, j] = np.maximum(0, v_paths[i-1, j] + self.kappa * (self.theta - v_paths[i-1, j]) * dt +
                                           self.sigma * np.sqrt(v_paths[i-1, j]) * sqrt_dt * z_v[j])
                S_paths[i, j] = S_paths[i-1, j] * np.exp((self.r - 0.5 * v_paths[i-1, j]) * dt +
                                                         np.sqrt(v_paths[i-1, j]) * sqrt_dt * z_S[j])

        return S_paths, v_paths

    def option_price(self):
        S_paths, v_paths = self.simulate_paths()

        if self.option_type == 'European':
            payoffs = np.maximum(0, S_paths[-1] - self.K)
        elif self.option_type == 'American':
            payoffs = np.maximum(0, np.maximum(S_paths - self.K, 0).max(axis=0))
        elif self.option_type == 'Asian':
            payoffs = np.maximum(0, np.mean(S_paths, axis=0) - self.K)

        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return option_price

    def calculate_greeks(self, epsilon=0.01):
        price = self.option_price()

        # Calculate delta
        delta = (self.option_price(S=self.S + epsilon) - price) / epsilon

        # Calculate vega
        vega = (self.option_price(sigma=self.sigma + epsilon) - price) / epsilon

        return {'delta': delta, 'vega': vega}

    def sensitivity_analysis(self, epsilon=0.01):
        greeks = {}
        perturbations = {'S': epsilon * self.S, 'K': epsilon * self.K, 'T': epsilon * self.T,
                         'r': epsilon * self.r, 'sigma': epsilon * self.sigma}

        for param, perturbation in perturbations.items():
            if param == 'S':
                option_price_up = self.option_price(S=self.S + perturbation)
                option_price_down = self.option_price(S=self.S - perturbation)
            elif param == 'K':
                option_price_up = self.option_price(K=self.K + perturbation)
                option_price_down = self.option_price(K=self.K - perturbation)
            elif param == 'T':
                option_price_up = self.option_price(T=self.T + perturbation)
                option_price_down = self.option_price(T=self.T - perturbation)
            elif param == 'r':
                option_price_up = self.option_price(r=self.r + perturbation)
                option_price_down = self.option_price(r=self.r - perturbation)
            elif param == 'sigma':
                option_price_up = self.option_price(sigma=self.sigma + perturbation)
                option_price_down = self.option_price(sigma=self.sigma - perturbation)

            sensitivity = (option_price_up - option_price_down) / (2 * perturbation)
            greeks[param] = sensitivity

        return greeks