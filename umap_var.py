import numpy as np

import utils as u
import time


class UMAP:
    def __init__(self,
                 n_components=2,
                 n_neighbors=5,
                 num_trees=5,
                 spread=1.,
                 min_dist=0.1,
                 learning_rate=1.0,
                 repulsion_strength=1.0,
                 negative_sample_rate=5,
                 init='random'  # only
                 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.num_trees = num_trees
        self.spread = spread
        self.min_dist = min_dist
        self._initial_alpha = learning_rate
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.init = init

        self._a, self._b = None, None
        # self.r_forest = None
        self.sigmas = None
        self.rhos = None
        self.graph = None
        self.embedding = None

    def fit(self, X, counts=None):
        t0 = time.time()
        self._a, self._b = u.find_ab_params(self.spread, self.min_dist)
        print(time.time() - t0)
        t0 = time.time()
        self.graph, self.sigmas, self.rhos = u.build_graph_nocoo(X, self.n_neighbors, counts)

        print(time.time() - t0)
        t0 = time.time()
        self.embedding = u.embed_graph(
            self.graph,
            X.shape[0],
            self.n_components,
            self._initial_alpha,  # self.learning_rate
            self._a,
            self._b,
            self.repulsion_strength,  # repulsive strength
            self.negative_sample_rate,
            n_epochs=0,
            init=self.init
        )
        print(time.time() - t0)

    def fit_transform(self, X, counts=None):
        self.fit(X, counts)
        return self.embedding

    def transform(self):
        pass
