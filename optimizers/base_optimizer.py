from abc import ABC, abstractmethod

import numpy as np

class BaseOptimizer(ABC):
    '''
    abstract base class for optimal portfolio allocation

    attributes:
        _returns: daily returns
        _cov: covariance matrix of daily returns
        _num_assets: number of assets in investment universe
        _rf: risk free rate
        _W_init: initial portfolio allocation for optimization task, takes
            values of equal weight strategy
        _long_only: True if long positions only, false otherwise
    '''

    def __init__(self,
                 returns,
                 cov,
                 risk_free,
                 long_only=True):
        '''
        constructor for the BaseOptimizer class

        args:
            returns: daily returns
            cov: covariance matrix of daily returns
            num_assets: number of assets in investment universe
            rf: risk free rate
            long_only: True if long positions only, false otherwise
        '''
        self._returns = returns
        self._cov = cov
        self._rf = risk_free
        self._long_only = long_only
        self._num_assets = self._returns.shape[0]
        self._W_init = np.ones(self._num_assets) / self._num_assets

    @abstractmethod
    def optimal_weights(self, objective):
        pass