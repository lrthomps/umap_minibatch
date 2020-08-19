from annoy import AnnoyIndex
import heapq

import numpy as np
# import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist, euclidean, sqeuclidean

TOLERANCE = 1e-5
MIN_DIST_SCALE = 1e-3


def embed_graph(
        data,  # needed for spectral embedding
        graph,
        n_components,
        initial_alpha,  # self.learning_rate
        a,
        b,
        gamma,  # repulsive strength
        negative_sample_rate,
        n_epochs=0,
        init='random'
):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    # TODO: implement spectral?
    if init == 'random':
        # embedding = 10 * np.random.randn(graph.shape[0], n_components)
        # embedding = embedding - np.min(embedding, 0)
        embedding = np.random.uniform(0, 10, size=(graph.shape[0], n_components))
    else:
        raise NotImplementedError('Only random initialization of embedding is implemented')

    # all the graph probabilities are encoded in the sampling opportunity over epochs per sample
    weights = graph.data
    epochs_per_sample = -np.ones(weights.shape[0])
    n_samples = n_epochs * (weights / weights.max())
    # set how often (eg. long) a given sample is sampled according to the weight of its edges
    epochs_per_sample[n_samples > 1/(n_epochs+1)] = n_epochs / n_samples[n_samples > 1/(n_epochs+1)]

    embedding = optimize_layout(
        embedding,
        None,
        graph.col, graph.row,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a, b,
        gamma,
        initial_alpha,
        negative_sample_rate
    )
    return embedding


def optimize_layout(
        head_embedding,
        tail_embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a, b,
        gamma=1,
        initial_alpha=1,
        negative_sample_rate=5,
):
    same_embs = tail_embedding is None
    if same_embs:
        tail_embedding = head_embedding
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    # monitor backlog of due negative sampling
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        _optimize_one_epoch(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a, b,
            gamma,
            same_embs,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n)

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return head_embedding


def _optimize_one_epoch(
        head_embedding,
        tail_embedding,
        head,
        tail,
        n_vertices,
        epochs_per_sample,
        a, b,
        gamma,
        same_embs,
        alpha,
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        epoch):

    def clip(val):
        return np.minimum(4, np.maximum(-4, val))

    def dpos_dy(current, other):
        d_sq = sqeuclidean(current, other)
        if d_sq > 0.0:
            grad_coeff = -2.0 * a * b * pow(d_sq, b - 1.0)
            grad_coeff /= a * pow(d_sq, b) + 1.0
            return clip(grad_coeff * (current - other))
        return np.zeros_like(current)

    def dneg_dy(current, others):
        d_sq = cdist(current[np.newaxis, :], others,
                     'sqeuclidean').reshape((-1, 1))
        grad_coeff = 2.0 * gamma * b / (
                (0.001 + d_sq) * (a * pow(d_sq, b) + 1))
        grad_coeff[d_sq <= 0] = 0

        grad_d = clip(grad_coeff * (current[np.newaxis, :] - others))
        grad_d[grad_coeff[:, 0] <= 0, :] = 4
        return grad_d

    for i, max_e in enumerate(epoch_of_next_sample):
        if max_e > epoch:
            continue
        j, k = head[i], tail[i]
        current, other = head_embedding[j], tail_embedding[k]

        dpos = dpos_dy(current, other)
        current += alpha * dpos
        if same_embs:
            other += -alpha * dpos

        epoch_of_next_sample[i] += epochs_per_sample[i]

        n_neg_samples = int(
            (epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
        )

        if n_neg_samples <= 0:
            continue

        k = np.random.randint(0, n_vertices - 1, n_neg_samples)
        while any(k == j):
            k[k == j] = np.random.randint(0, n_vertices - 1, np.sum(k == j))

        others = tail_embedding[k]
        dneg = dneg_dy(current, others)
        current += alpha * np.sum(dneg, axis=0)
        if same_embs:
            others += -alpha * dneg

        epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
        )


def build_graph(X, n_neighbors):
    knn_indices, knn_dists = nearer_neighbours(X, n_neighbors)

    sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)
    rows, cols, vals = compute_graph_weights(knn_indices, knn_dists, sigmas, rhos)

    s_pigj = coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    s_pigj.eliminate_zeros()

    s_pjgi = s_pigj.transpose()
    prod_matrix = s_pigj.multiply(s_pjgi)
    s_pij = s_pigj + s_pjgi - prod_matrix
    s_pij.eliminate_zeros()

    return s_pij, sigmas, rhos


def random_nn_trees(X, num_trees):
    t = AnnoyIndex(X.shape[1], 'euclidean')
    for i in range(X.shape[0]):
        t.add_item(i, X[i, :])
    t.build(num_trees)
    return t


# def nearest_neighbours(X, k, num_trees=5, num_iters=1):
#     r_forest = random_nn_trees(X, num_trees)
#     knn = [r_forest.get_nns_by_item(i, k, include_distances=True) for i in range(X.shape[0])]
#     for i in range(X.shape[0]):
#         knn[i] = (-np.array(knn[i][1][1:]), np.array(knn[i][0][1:]))
#
#     for _ in range(num_iters):
#         old_knn = knn
#         for i in range(X.shape[0]):
#             h = list(zip(*old_knn[i]))
#             heapq.heapify(h)
#             for j in old_knn[i][1]:
#                 if i == j:
#                     continue
#                 for l in old_knn[j][1]:
#                     if (i == l) or (l in list(zip(*h))[1]):
#                         continue
#                     d_il = euclidean(X[i, :], X[l, :])
#                     heapq.heappush(h, (-d_il, l))
#                     if len(h) > k:
#                         heapq.heappop(h)
#             knn[i] = list(zip(*h))
#     knn_dists, knn_indices = zip(*knn)
#
#     knn_indices = np.array(knn_indices)
#     knn_dists = -np.array(knn_dists)
#
#     return knn_indices, knn_dists


def nearer_neighbours(X, k, num_trees=5, num_iters=2):
    r_forest = random_nn_trees(X, num_trees)
    knn = [r_forest.get_nns_by_item(i, k, include_distances=True) for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        knn[i] = (np.array(knn[i][0][1:]), np.array(knn[i][1][1:]))
    for _ in range(num_iters):
        old_knn = knn
        for i in range(X.shape[0]):
            ind, dist = old_knn[i]
            nn_ind = np.unique([k for j in ind for k in old_knn[j][0]
                                if (k != i) and (k not in ind)])
            ind = np.append(ind, nn_ind)
            dist = np.append(dist, cdist(X[[i], :], X[nn_ind, :]))
            keep = np.argsort(dist)[:k]
            knn[i] = (ind[keep], dist[keep])
    knn_indices, knn_dists = zip(*knn)
    return np.array(knn_indices), np.array(knn_dists)


def binary_search(f, target, lo=0., mid=1., hi=np.inf, n_iter=64):
    for _ in range(n_iter):
        f_mid = f(mid)
        if np.abs(f_mid - target) < TOLERANCE:
            break

        if f_mid > target:
            hi = mid
            mid = 0.5 * (lo + hi)
        else:
            lo = mid
            mid = mid*2 if np.isinf(hi) else 0.5 *(lo + hi)
    return mid


def smooth_knn_dist(knn_dists, k, bandwidth=1):
    target = np.log2(k) * bandwidth
    rho = np.zeros(knn_dists.shape[0])
    sigmas = np.zeros(knn_dists.shape[0])

    for i in range(knn_dists.shape[0]):
        rho[i] = np.min(knn_dists[i, knn_dists[i, :] > 0])
        d = knn_dists[i, :] - rho[i]
        psum = lambda sigma: np.sum(np.exp(-d / sigma))
        sigmas[i] = binary_search(psum, target)

    rho0 = rho == 0
    mean_distances = np.mean(knn_dists, axis=1)
    sig0 = sigmas < MIN_DIST_SCALE * mean_distances
    sigmas[~rho0 & sig0] = MIN_DIST_SCALE * mean_distances[~rho0 & sig0]
    mean_distance = np.mean(mean_distances)
    sig00 = sigmas < MIN_DIST_SCALE * mean_distance
    sigmas[rho0 & sig00] = MIN_DIST_SCALE * mean_distance
    return sigmas, rho


def compute_graph_weights(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.repeat(np.arange(n_samples), (n_neighbors))
    cols = knn_indices.reshape((-1))

    rho_m = rhos[:, np.newaxis]
    sig_m = sigmas[:, np.newaxis]
    ok = (knn_dists - rho_m > 0.0) & (sig_m > 0.0)

    vals = np.exp(- (knn_dists - rho_m) / sig_m)
    vals[~ok] = 1

    vals = vals.reshape((-1))

    return rows, cols, vals


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]