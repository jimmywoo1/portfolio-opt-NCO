from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from optimizers.base_optimizer import BaseOptimizer
from utils.utils import *


class PortfolioOptimizer(BaseOptimizer):
    '''
    optimal portfolio allocation using Markowitz mean-variance optimization

    attributes:
        _returns: daily returns
        _cov: covariance matrix of daily returns
        _num_assets: number of assets in investment universe
        _rf: risk free rate
        _W_init: initial portfolio allocation for optimization task, takes
            values of equal weight strategy
        _long_only: True if long positions only, false otherwise
        _bounds: lower and upper bounds of the weights of each asset
    '''
    def __init__(self,
                 returns,
                 cov,
                 risk_free,
                 long_only=True):
        '''
        constructor for the PortfolioOptimizer class

        args:
            returns: daily returns
            cov: covariance matrix of daily returns
            num_assets: number of assets in investment universe
            rf: risk free rate
            long_only: True if long positions only, false otherwise
        '''
        super().__init__(returns, cov, risk_free, long_only)
        wb = (0, 1) if self._long_only else (-1, 1)
        self._bounds = [wb for _ in range(self._num_assets)]

    def optimal_weights(self, objective) -> np.ndarray:
        '''
        computes the optimal weights using the provided objective function

        args:
            objective: "sharpe" for maximum sharpe ratio, and "variance"
                for minimum variance

        returns:
            optimal weight allocation
        '''
        constraints = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1})

        # define objective function
        if objective == 'sharpe':
            objective_fn = compute_sharpe
            args = (self._returns, self._cov, self._rf)
        elif objective == 'variance':
            objective_fn = compute_variance
            args = (self._cov)

        self._W  = minimize(objective_fn,
                            self._W_init,
                            args=args,
                            method='SLSQP',
                            bounds=self._bounds,
                            constraints=constraints).x

        return self._W

    def compute_efficient_frontier(self) -> List[Tuple[float, float]]:
        '''
        computes the efficient frontier of portfolios

        returns:
            list of volatilities and returns corresponding to the efficient
                frontier
        '''
        # compute over min/max of historical returns
        target_returns = np.linspace(np.min(self._returns), np.max(self._returns), 250)
        efficient_frontier = []

        for target in target_returns:
            constraints = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1},
                           {'type': 'eq', 'fun': lambda W: W @ self._returns - target})

            res = minimize(compute_variance,
                           self._W_init,
                           args=(self._cov),
                           method='SLSQP',
                           bounds=self._bounds,
                           constraints=constraints).x

            volatility = np.sqrt(res.T @ self._cov @ res)
            efficient_frontier.append((target, volatility))

        return efficient_frontier