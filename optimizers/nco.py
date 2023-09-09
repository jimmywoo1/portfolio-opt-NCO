from typing import Dict, Any

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from optimizers.base_optimizer import BaseOptimizer
from utils.utils import *

class NestedClusteredOptimizer(BaseOptimizer):
    '''
    optimal portfolio allocation using Nested Clustered Optimization algorithm
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961)

    attributes:
        _returns: daily returns
        _cov: covariance matrix of daily returns
        _num_assets: number of assets in investment universe
        _rf: risk free rate
        _W_init: initial portfolio allocation for optimization task, takes
            values of equal weight strategy
        _long_only: True if long positions only, false otherwise
        _corr: correlation matrix of daily returns
    '''
    def __init__(self,
                 returns,
                 cov,
                 risk_free,
                 long_only=True):
        '''
        constructor for the NestedClusteredOptimizer class

        args:
            returns: daily returns
            cov: covariance matrix of daily returns
            num_assets: number of assets in investment universe
            rf: risk free rate
            long_only: True if long positions only, false otherwise
        '''
        super().__init__(returns, cov, risk_free, long_only)
        self._corr = compute_corr(cov)


    def _cluster_assets(self, max_num_clusters: int=None) -> None:
        '''
        groups assets into clusters using k means, using silhoueete scores
        to find the optimal number of clusters

        args:
            max_num_clusters: maximum number of clusters allowed
        '''
        # distance matrix for silhouette scores
        dist = ((1 - self._corr/ 2) ** 0.5).fillna(0)
        silhouette_scores = pd.Series(dtype=object)
        kmeans_obj = None

        max_num_clusters = self._corr.shape[0] // 2 if max_num_clusters is None else max_num_clusters

        # check num clusters
        for i in range(2, max_num_clusters):
            kmeans = KMeans(n_clusters=i, n_init=10).fit(dist)
            curr_scores = silhouette_samples(dist, kmeans.labels_)
            curr_metric = curr_scores.mean() / curr_scores.std()
            best_metric = silhouette_scores.mean() / silhouette_scores.std()

            # current silhouette scores better
            if np.isnan(best_metric) or curr_metric > best_metric:
                silhouette_scores = curr_scores
                kmeans_obj = kmeans

        # assign clusters using best cluster sizes
        self._clusters = {i: self._corr.columns[np.where(kmeans.labels_ == i)[0]].tolist()
                          for i in np.unique(kmeans.labels_)}
        self._num_clusters = len(self._clusters.keys())


    def optimal_weights(self,
                        objective: str,
                        **kwargs: Dict[str, Any]) -> np.ndarray:
        '''
        computes the optimal weights using the NCO algorithm

        args:
            objective: "sharpe" for maximum sharpe ratio, and "variance"
                for minimum variance

        returns:
            optimal weight allocation
        '''
        max_num_clusters = kwargs.pop("max_num_clusters", None)
        self._cluster_assets(max_num_clusters)

        # optimization parameter
        constraints = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1})
        wb = (0, 1) if self._long_only else (-1, 1)
        intra_weights = np.zeros((self._num_assets, self._num_clusters))

        # within-cluster weights
        for idx, cluster in self._clusters.items():
            curr_cov = self._cov[cluster][:, cluster]
            curr_mu = self._returns[cluster].reshape(-1, 1) \
                      if objective == "sharpe" else None
            intra_weights[cluster, idx] = convex_opt(curr_cov, curr_mu).flatten()

        # cluster weights
        cluster_cov = intra_weights.T @ self._cov @ intra_weights
        cluster_mu = intra_weights.T @ self._returns \
                     if objective == "sharpe" else None
        inter_weights = convex_opt(cluster_cov, cluster_mu).flatten()

        # final asset allocation
        self._W = np.multiply(intra_weights, inter_weights).sum(axis=1)

        return self._W