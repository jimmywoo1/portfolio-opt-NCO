from typing import Tuple, Optional

import numpy as np
import pandas as pd

def compute_mean_returns(W: np.ndarray, returns: np.ndarray) -> float:
    '''
    computes the mean returns based on portfolio allocations

    args:
        W: weights of portfolio allocations
        returns: daily returns

    returns:
        mean returns
    '''
    return W @ returns

def compute_variance(W: np.ndarray, cov: np.ndarray) -> np.ndarray:
    '''
    computes the variance of returns based on portfolio allocations

    args:
        W: weights of portfolio allocations
        cov: covariance matrix of daily returns

    returns:
        variance of returns
    '''
    return W.T @ cov @ W


def compute_corr(cov: np.ndarray) -> np.ndarray:
    '''
    normalize covariance matrix into a correlation matrix

    args:
        cov: covariance matrix of daily returns

    returns:
        correlation of returns
    '''
    cov = pd.DataFrame(cov)
    std = np.sqrt(np.diag(cov))

    return cov / np.outer(std, std)

def historical_returns(prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    computes the annualized returns and covariance from historical prices

    args:
        prices: daily adjusted closing prices

    returns:
        annualized expected returns and covariance matrix
    '''
    returns = prices.pct_change().dropna()

    # expected returns
    expected_returns = returns.mean()
    expected_returns = (1 + expected_returns.values) ** 252 - 1

    # cumulative returns
    cum_returns = np.cumprod(1 + returns) - 1

    # covariance matrix
    cov = returns.cov()
    cov *= 252

    return expected_returns, cov.values, cum_returns

def compute_sharpe(W: np.ndarray,
                   returns: np.ndarray,
                   cov: np.ndarray,
                   rf: float) -> float:
    '''
    computes the (negative) sharpe ratio of  provided portfolio allocation

    args:
        W: weights of portfolio allocations
        returns: daily returns
        cov: covariance matrix of daily returns
        rf: risk free rate

    returns:
        (negative) sharpe ratio
    '''
    mu = compute_mean_returns(W, returns)
    var = compute_variance(W, cov)
    sharpe = (mu - rf) / np.sqrt(var)

    return -1 * sharpe

def convex_opt(cov: np.ndarray, mu: Optional[np.ndarray]=None) -> None:
    """
    """
    # pseudoinverse for numerically unstable matrices
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)

    ones = np.ones(shape=(len(inv), 1))
    mu = ones if mu is None else mu

    w = np.dot(inv, mu)
    return w / np.dot(ones.T, w)
