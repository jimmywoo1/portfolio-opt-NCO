from datetime import datetime, timedelta

import yfinance as yf
import matplotlib.pyplot as plt

from utils.utils import *
from optimizers.portfolio_optimizer import PortfolioOptimizer
from optimizers.nco import NestedClusteredOptimizer

if __name__ == "__main__":
    # load data
    num_years = 10
    rf = 0.04
    end = datetime(2022, 8, 31)
    start = end - timedelta(days=365 * num_years)
    tickers = ['XOM', 'AAPL', 'MSFT', "GOOG", "META", "AMZN", "AMD"]
    prices = yf.download(tickers, start=start, end=end)['Adj Close']

    mu, cov, cum_returns = historical_returns(prices)
    opt = PortfolioOptimizer(mu, cov, rf)

    # sharpe based portfolio optimization
    tan_w = opt.optimal_weights("sharpe")
    mu_t = compute_mean_returns(tan_w, mu)
    vol_t = np.sqrt(compute_variance(tan_w, cov))

    # global minimum variance
    min_var_w = opt.optimal_weights("variance")
    mu_v = compute_mean_returns(min_var_w, mu)
    vol_v = np.sqrt(compute_variance(min_var_w, cov))

    # NCO
    ncopt = NestedClusteredOptimizer(mu, cov, rf)
    nco_min_var_w = ncopt.optimal_weights("variance", max_num_clusters=None)
    mu_nco_min_var = compute_mean_returns(nco_min_var_w, mu)
    vol_nco_min_var = np.sqrt(compute_variance(nco_min_var_w, cov))
    nco_sharpe_w = ncopt.optimal_weights("sharpe", max_num_clusters=None)
    mu_nco_sharpe = compute_mean_returns(nco_sharpe_w, mu)
    vol_nco_sharpe = np.sqrt(compute_variance(nco_sharpe_w, cov))

    # efficient frontier
    ef = opt.compute_efficient_frontier()
    ef = np.array(ef)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f1, axes = plt.subplots(1, 2, figsize=(13, 7))

    # generate capital market line
    x_r = np.arange(0, 0.5, 0.01)
    A = np.vstack([[0, vol_t], np.ones(2)]).T
    k, b = np.linalg.lstsq(A, [rf, mu_t], rcond=None)[0]
    axes[0].plot(x_r, k * x_r + b, color=colors[2], label="Capital Market Line")

    # generate efficient frontier plot
    axes[0].plot(ef[:, 1], ef[:, 0], color=colors[1], label="Efficient Frontier")
    axes[0].scatter(vol_t, mu_t, color='g', marker='*', s=100, label="Maximum Sharpe")
    axes[0].scatter(vol_v, mu_v, color='b', marker='*', s=100, label="Global Minimum Variance")
    axes[0].scatter(vol_nco_min_var, mu_nco_min_var, color='m', marker='*', s=100, label="NCO (Min. Variance)")
    axes[0].scatter(vol_nco_sharpe, mu_nco_sharpe, color='c', marker='*', s=100, label="NCO (Max. Sharpe)")
    axes[0].scatter([cov[i, i] ** 0.5 for i in range(len(tickers))], mu, color='k', label='Individual Assets')
    axes[0].scatter(0, rf, color=colors[0], label="Risk Free Rate")

    for i in range(len(tickers)):
        axes[0].text(cov[i, i] ** 0.5 + 0.01, mu[i], tickers[i])

    axes[0].set(xlabel="Volatility", ylabel="Return", title="Efficient Frontier")
    axes[0].legend()

    bar_width = 0.1
    x = np.arange(len(tickers))

    # grouped bar chart of allocations
    axes[1].bar(x - 1.5 * bar_width, tan_w, bar_width, label='Maximum Sharpe')
    axes[1].bar(x - 0.5 * bar_width, min_var_w, bar_width, label='Global Minimum Variance')
    axes[1].bar(x + 0.5 * bar_width, nco_min_var_w, bar_width, label='NCO (Min. Variance)')
    axes[1].bar(x + 1.5 * bar_width, nco_sharpe_w, bar_width, label='NCO (Max. Sharpe)')
    axes[1].set(xlabel='Assets', ylabel="Weights", title="Weight Allocation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers)
    axes[1].legend()

    # in-sample test
    sharpe_returns = cum_returns @ tan_w
    min_var_returns = cum_returns @ min_var_w
    nco_min_var_returns = cum_returns @ nco_min_var_w
    nco_sharpe_returns = cum_returns @ nco_sharpe_w
    equal_w_returns = cum_returns @ (np.ones(cum_returns.shape[1]) / cum_returns.shape[1])

    f2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(sharpe_returns, label="Maximum Sharpe")
    ax2.plot(min_var_returns, label='Global Minimum Variance')
    ax2.plot(nco_min_var_returns, label='NCO (Min. Variance)')
    ax2.plot(nco_sharpe_returns, label='NCO (Max. Sharpe)')
    ax2.plot(equal_w_returns, label='Equal Weight')
    ax2.set(xlabel='Time', ylabel="Returns", title="In-Sample: Cumulative Returns Over Time")
    ax2.legend()

    # load out-of-sample data
    num_years_test = 1
    end_test = datetime(2023, 8, 31)
    start_test = end_test - timedelta(days=365 * num_years_test)
    prices_test = yf.download(tickers, start=start, end=end)['Adj Close']

    mu_test, cov_test, cum_returns_test = historical_returns(prices_test)

    mu_test, cov_test, cum_returns_test = historical_returns(prices_test)

    # out-of-sample test
    sharpe_returns_test = cum_returns_test @ tan_w
    min_var_returns_test = cum_returns_test @ min_var_w
    nco_min_var_returns_test = cum_returns_test @ nco_min_var_w
    nco_sharpe_returns_test = cum_returns_test @ nco_sharpe_w
    equal_w_returns_test = cum_returns_test @ (np.ones(cum_returns_test.shape[1]) / cum_returns_test.shape[1])

    f3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.plot(sharpe_returns_test, label="Maximum Sharpe")
    ax3.plot(min_var_returns_test, label='Global Minimum Variance')
    ax3.plot(nco_min_var_returns_test, label='NCO (Min. Variance)')
    ax3.plot(nco_sharpe_returns_test, label='NCO (Max. Sharpe)')
    ax3.plot(equal_w_returns_test, label='Equal Weight')
    ax3.set(xlabel='Time', ylabel="Returns", title="Out-of-Sample: Cumulative Returns over Time")
    ax3.legend()

    plt.show()