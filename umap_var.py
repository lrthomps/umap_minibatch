import utils as u


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
        self.r_forest = None
        self.sigmas = None
        self.rhos = None
        self.graph = None
        self.embedding = None

    def fit(self, X):
        self._a, self._b = u.find_ab_params(self.spread, self.min_dist)
        self.graph, self.sigmas, self.rhos = u.build_graph(X, self.n_neighbors)
        self.embedding = u.embed_graph(
            X,  # needed for spectral embedding
            self.graph,
            self.n_components,
            self._initial_alpha,  # self.learning_rate
            self._a,
            self._b,
            self.repulsion_strength,  # repulsive strength
            self.negative_sample_rate,
            n_epochs=0,
            init=self.init
        )

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding

    def transform(self):
        pass
