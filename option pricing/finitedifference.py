import numpy as np

class FiniteDifferenceOptionPricing:
    def __init__(self, S, K, T, r, sigma, option_type='European', num_steps=100, num_grid_points=100):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.option_type = option_type  # Type of option: 'European', 'American'
        self.num_steps = num_steps  # Number of time steps
        self.num_grid_points = num_grid_points  # Number of grid points in the spatial direction

    def finite_difference_method(self):
        dt = self.T / self.num_steps  # Time step
        ds = 2 * self.S / self.num_grid_points  # Spatial step
        grid = np.zeros((self.num_steps + 1, self.num_grid_points + 1))

        # Initialize boundary conditions
        if self.option_type == 'European':
            grid[-1, :] = np.maximum(0, np.linspace(0, 2 * self.K, self.num_grid_points + 1) - self.K)
            grid[:, 0] = self.K
            grid[:, -1] = 0
        elif self.option_type == 'American':
            grid[-1, :] = np.maximum(0, np.linspace(0, 2 * self.K, self.num_grid_points + 1) - self.K)
            grid[:, 0] = self.K
            grid[:, -1] = 0

        # Finite difference method
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(1, self.num_grid_points):
                alpha = 0.5 * self.sigma**2 * j**2 * dt
                beta = 0.5 * self.r * j * dt
                gamma = -self.r * dt

                grid[i, j] = alpha * grid[i + 1, j - 1] + (1 - alpha - gamma) * grid[i + 1, j] + beta * grid[i + 1, j + 1]

                if self.option_type == 'American':
                    grid[i, j] = max(grid[i, j], np.maximum(0, j * ds - self.K))

        return grid[0, self.num_grid_points // 2]
