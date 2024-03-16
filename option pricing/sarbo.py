import numpy as np
from scipy.stats import norm

class SABROptionPricing:
    def __init__(self, S, K, T, r, alpha, beta, rho, nu, option_type='European'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.alpha = alpha  # SABR alpha parameter
        self.beta = beta  # SABR beta parameter
        self.rho = rho  # SABR rho parameter
        self.nu = nu  # SABR nu parameter
        self.option_type = option_type  # Type of option: 'European', 'American', or 'Asian'

    def sabr_volatility(self, F, K):
        if F == K:
            return self.alpha

        z = (self.nu / self.alpha) * ((F * K) ** ((1 - self.beta) / 2)) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

        A = self.alpha / ((F * K) ** ((1 - self.beta) / 2) * (1 + (((1 - self.beta) ** 2) / 24 * (np.log(F / K) ** 2)) +
                                                              (((1 - self.beta) ** 4) / 1920 * (np.log(F / K) ** 4))))

        B = 1 + ((1 - self.beta) ** 2 / 24 * self.alpha ** 2 / ((F * K) ** (1 - self.beta)) +
                 (1 / 4) * self.rho * self.beta * self.nu * self.alpha / ((F * K) ** ((1 - self.beta) / 2)) +
                 (2 - 3 * self.rho ** 2) / 24 * self.nu ** 2)

        return A * z / x * B

    def sabr_option_price(self):
        F = self.S * np.exp(self.r * self.T)
        K = self.K
        T = self.T

        if self.option_type == 'European':
            if K == 0:
                return np.maximum(0, F - self.K)
            else:
                iv = self.sabr_volatility(F, K)
                d1 = (np.log(F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
                d2 = d1 - iv * np.sqrt(T)
                if F > K:
                    option_price = F * norm.cdf(d1) - K * norm.cdf(d2)
                else:
                    option_price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
        elif self.option_type == 'Asian':
            if K == 0:
                return np.maximum(0, np.mean(F) - self.K)
            else:
                average_F = np.mean(F)
                iv = self.sabr_volatility(average_F, K)
                d1 = (np.log(average_F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
                d2 = d1 - iv * np.sqrt(T)
                if average_F > K:
                    option_price = average_F * norm.cdf(d1) - K * norm.cdf(d2)
                else:
                    option_price = K * norm.cdf(-d2) - average_F * norm.cdf(-d1)
        elif self.option_type == 'American':
            raise NotImplementedError("American option pricing with SABR model is not implemented yet.")
        else:
            raise ValueError("Invalid option type. Use 'European' or 'Asian'.")

        return option_price

    def calculate_greeks(self, epsilon=0.01):
        price = self.sabr_option_price()

        # Calculate delta
        delta = (self.sabr_option_price(S=self.S + epsilon) - price) / epsilon

        # Calculate gamma
        gamma = (self.sabr_option_price(S=self.S + epsilon) - 2 * price + self.sabr_option_price(S=self.S - epsilon)) / epsilon ** 2

        # Calculate vega
        vega = (self.sabr_option_price(alpha=self.alpha + epsilon) - price) / epsilon

        # Calculate theta
        theta = -(self.sabr_option_price(T=self.T - epsilon) - price) / epsilon

        # Calculate rho
        rho = (self.sabr_option_price(r=self.r + epsilon) - price) / epsilon

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def sensitivity_analysis(self, epsilon=0.01):
        greeks = {}
        perturbations = {'S': epsilon * self.S, 'K': epsilon * self.K, 'T': epsilon * self.T,
                         'r': epsilon * self.r, 'alpha': epsilon * self.alpha, 'beta': epsilon * self.beta,
                         'rho': epsilon * self.rho, 'nu': epsilon * self.nu}

        for param, perturbation in perturbations.items():
            if param == 'S':
                option_price_up = self.sabr_option_price(S=self.S + perturbation)
                option_price_down = self.sabr_option_price(S=self.S - perturbation)
            elif param == 'K':
                option_price_up = self.sabr_option_price(K=self.K + perturbation)
                option_price_down = self.sabr_option_price(K=self.K - perturbation)
            elif param == 'T':
                option_price_up = self.sabr_option_price(T=self.T + perturbation)
                option_price_down = self.sabr_option_price(T=self.T - perturbation)
            elif param == 'r':
                option_price_up = self.sabr_option_price(r=self.r + perturbation)
                option_price_down = self.sabr_option_price(r=self.r - perturbation)
            elif param == 'alpha':
                option_price_up = self.sabr_option_price(alpha=self.alpha + perturbation)
                option_price_down = self.sabr_option_price(alpha=self.alpha - perturbation)
            elif param == 'beta':
                option_price_up = self.sabr_option_price(beta=self.beta + perturbation)
                option_price_down = self.sabr_option_price(beta=self.beta - perturbation)
            elif param == 'rho':
                option_price_up = self.sabr_option_price(rho=self.rho + perturbation)
                option_price_down = self.sabr_option_price(rho=self.rho - perturbation)
            elif param == 'nu':
                option_price_up = self.sabr_option_price(nu=self.nu + perturbation)
                option_price_down = self.sabr_option_price(nu=self.nu - perturbation)

            sensitivity = (option_price_up - option_price_down) / (2 * perturbation)
            greeks[param] = sensitivity

        return greeks