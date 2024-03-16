import numpy as np
from scipy.stats import binom, norm

class BinomialOptionPricing:
    def __init__(self, S, K, T, r, sigma, num_steps=100, option_type='European'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.num_steps = num_steps  # Number of steps in the binomial tree
        self.option_type = option_type  # Type of option: 'European', 'American', or 'Asian'

    def binomial_tree(self):
        dt = self.T / self.num_steps  # Time step
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        q = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize stock price tree
        stock_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        stock_tree[0, 0] = self.S

        # Populate stock price tree
        for i in range(1, self.num_steps + 1):
            stock_tree[i, 0] = stock_tree[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_tree[i, j] = stock_tree[i - 1, j - 1] * d

        return stock_tree

    def option_price(self, q):
        dt = self.T / self.num_steps  # Time step
        stock_tree = self.binomial_tree()

        # Initialize option price tree
        option_tree = np.zeros_like(stock_tree)

        # Calculate option values at expiration
        if self.option_type == 'European':
            option_tree[:, -1] = np.maximum(0, stock_tree[:, -1] - self.K)
        elif self.option_type == 'American':
            option_tree[:, -1] = np.maximum(0, stock_tree[:, -1] - self.K)
        elif self.option_type == 'Asian':
            option_tree[:, -1] = np.maximum(0, np.mean(stock_tree[:, 1:], axis=1) - self.K)

        # Calculate option values at earlier nodes
        for j in range(self.num_steps - 1, -1, -1):
            for i in range(j + 1):
                option_tree[i, j] = np.exp(-self.r * dt) * (
                        (1 - q) * option_tree[i, j + 1] + q * option_tree[i + 1, j + 1])

                if self.option_type == 'American':
                    option_tree[i, j] = max(option_tree[i, j], stock_tree[i, j] - self.K)

        return option_tree[0, 0]

    def calculate_greeks(self, epsilon=0.01):
        price = self.option_price()

        # Calculate delta
        delta = (self.option_price(S=self.S + epsilon) - price) / epsilon

        # Calculate gamma
        gamma = (self.option_price(S=self.S + epsilon) - 2 * price + self.option_price(S=self.S - epsilon)) / epsilon ** 2

        # Calculate vega
        vega = (self.option_price(sigma=self.sigma + epsilon) - price) / epsilon

        # Calculate theta
        theta = -(self.option_price(T=self.T - epsilon) - price) / epsilon

        # Calculate rho
        rho = (self.option_price(r=self.r + epsilon) - price) / epsilon

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

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