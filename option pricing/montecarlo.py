import numpy as np

class MonteCarloOptionPricing:
    def __init__(self, S, K, T, r, sigma, num_simulations=10000, option_type='European'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.num_simulations = num_simulations  # Number of Monte Carlo simulations
        self.option_type = option_type  # Type of option: 'European', 'American', or 'Asian'

    def simulate_stock_prices(self):
        dt = self.T / 252  # Assuming 252 trading days in a year
        returns = (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * np.random.normal(size=(self.num_simulations, int(self.T * 252)))
        price_paths = np.exp(np.log(self.S) + np.cumsum(returns, axis=1))
        return price_paths

    def option_price(self):
        price_paths = self.simulate_stock_prices()

        if self.option_type == 'European':
            if self.T == 0:
                return np.maximum(0, self.S - self.K)
            else:
                discounted_payoffs = np.exp(-self.r * self.T) * np.maximum(price_paths[:, -1] - self.K, 0)
                option_price = np.mean(discounted_payoffs)
        elif self.option_type == 'American':
            discounted_payoffs = np.maximum(price_paths - self.K, 0)
            option_price = np.mean(np.exp(-self.r * self.T) * discounted_payoffs.max(axis=1))
        elif self.option_type == 'Asian':
            if self.T == 0:
                return np.maximum(0, np.mean(self.S) - self.K)
            else:
                average_prices = np.mean(price_paths, axis=1)
                discounted_payoffs = np.exp(-self.r * self.T) * np.maximum(average_prices - self.K, 0)
                option_price = np.mean(discounted_payoffs)
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