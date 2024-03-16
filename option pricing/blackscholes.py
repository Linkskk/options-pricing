import numpy as np
from scipy.stats import norm

class BlackScholesOptionPricing:
    def __init__(self, S, K, T, r, sigma, option_type='European'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.option_type = option_type  # Type of option: 'European', 'American', or 'Asian'

    def option_price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'European':
            if self.option_type == 'call':
                option_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            elif self.option_type == 'put':
                option_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        elif self.option_type == 'American':
            # Implement American option pricing using analytical approximation
            # Here, we assume the European option price for simplicity
            option_price = self.option_price(option_type='European')
        elif self.option_type == 'Asian':
            # Implement Asian option pricing using analytical approximation
            # Here, we assume the European option price for simplicity
            option_price = self.option_price(option_type='European')
        else:
            raise ValueError("Invalid option type. Use 'European', 'American', or 'Asian'.")

        return option_price

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