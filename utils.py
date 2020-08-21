from annoy import AnnoyIndex
import copy
import numpy as np
import random
from scipy.optimize import curve_fit


TOLERANCE = 1e-5
MAX_GRAD = 3.0
MIN_DIST_SCALE = 1e-3


def embed_graph(
        knn,
        n_vertices,
        n_components,
        initial_alpha,  # self.learning_rate
        a,
        b,
        gamma,  # repulsive strength
        negative_sample_rate,
        n_epochs=0,
        init='random'
):
    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if n_vertices <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    # TODO: implement spectral?
    if init == 'random':
        # embedding = 10 * np.random.randn(n_vertices, n_components)
        # embedding = embedding - np.min(embedding, 0)
        embedding = np.random.uniform(0, 10, size=(n_vertices, n_components))
    else:
        raise NotImplementedError('Only random initialization of embedding is implemented')

    # print(optimize_nothing(
    #     embedding,
    #     knn,
    #     n_epochs,
    #     negative_sample_rate))

    embedding = optimize_layout(
        embedding,
        knn,
        n_epochs,
        n_vertices,
        a, b,
        gamma,
        initial_alpha,
        negative_sample_rate
    )
    return embedding


def optimize_nothing(
        embedding,
        knn,
        n_epochs,
        negative_sample_rate
):
    summed = np.zeros(2)
    for n in range(n_epochs):
        for j, (ks, weight) in enumerate(knn):
            summed += embedding[j, :]/10 + np.sum(embedding[ks.astype(int), :], axis=0)/100
            ks = np.arange(0, negative_sample_rate * len(ks), 3)
            summed += np.sum(embedding[ks, :], axis=0)/100
    return summed


def optimize_layout(
        embedding,
        knn,
        n_epochs,
        n_vertices,
        a, b,
        gamma=1,
        initial_alpha=1,
        negative_sample_rate=5,
):
    def get_next(k, i_last):
        next = np.arange(i_last, i_last + k)
        i_last += k
        if i_last >= n_vertices:
            next[next >= n_vertices] -= n_vertices
            i_last -= n_vertices
        return next, i_last

    i_last = 1

    alpha = 2.0 * b * initial_alpha
    da = alpha / n_epochs

    for n in range(n_epochs):
        for j, (kpos, weight) in enumerate(knn):
            keep = random.random() <= weight
            kpos = kpos[keep].astype(int)
            if len(kpos) == 0:
                continue

            dpos = dpos_dy(embedding[[j], :], embedding[kpos, :], a, b)
            embedding[kpos, :] += -alpha * dpos

            kneg, i_last = get_next(negative_sample_rate * len(kpos), i_last)
            while np.any(kneg == j):
                kneg[kneg == j], i_last = get_next(np.sum(kneg == j), i_last)

            dneg = dneg_dy(embedding[[j], :], embedding[kneg], gamma, a, b)
            embedding[j, :] += alpha * (np.sum(dneg, axis=0) + np.sum(dpos, axis=0))
            embedding[kneg] += -alpha * dneg
        alpha -= da

    return embedding


def clip(val):
    return np.minimum(MAX_GRAD, np.maximum(-MAX_GRAD, val))


def dpos_dy(current, others, a, b):
    d_sq = l2_sq(current, others)
    grad_coeff = -a * (np.power(d_sq, b - 1.0)
                       / (a * np.power(d_sq, b) + 1.0))
    return clip(grad_coeff * (current - others))


def dneg_dy(current, others, gamma, a, b):
    d_sq = l2_sq(current, others)
    grad_coeff = gamma / (
            (0.001 + d_sq) * (a * np.power(d_sq, b) + 1))
    grad_coeff[d_sq <= 0] = 0

    grad_d = clip(grad_coeff * (current - others))
    grad_d[grad_coeff[:, 0] <= 0, :] = MAX_GRAD
    return grad_d


def build_graph_nocoo(X, n_neighbors):
    knn_d = nearer_neighbours(X, n_neighbors)

    sigmas, rhos = smooth_knn_dist(knn_d, n_neighbors)
    knn_w = compute_graph_weights(np.array(knn_d), sigmas, rhos)

    knn_list = []
    for i in range(knn_w.shape[0]):
        new, new_w = np.where(knn_w[:, 0, :] == i)
        dup = np.isin(new, knn_w[i, 0, :])
        for ii, j in enumerate(knn_w[i, 0, :]):
            if j not in new[dup]:
                continue
            jj = np.where(j == new[dup])[0][0]
            knn_w[i, 1, ii] = knn_w[i, 1, ii] + knn_w[j.astype(int), 1, jj] - knn_w[i, 1, ii]*knn_w[j.astype(int), 1, jj]
            knn_w[j.astype(int), 1, jj] = 0

        knn_list.append(np.concatenate(
            [knn_w[i, :, :], np.row_stack([new[~dup], knn_w[new[~dup], 1, new_w[~dup]]])], axis=1))

    return knn_list, sigmas, rhos


# def build_graph(X, n_neighbors):
#     knn_indices, knn_dists = nearer_neighbours(X, n_neighbors)
#
#     sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)
#     rows, cols, vals = compute_graph_weights(knn_indices, knn_dists, sigmas, rhos)
#
#     s_pigj = coo_matrix(
#         (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
#     )
#     s_pigj.eliminate_zeros()
#
#     s_pjgi = s_pigj.transpose()
#     prod_matrix = s_pigj.multiply(s_pjgi)
#     s_pij = s_pigj + s_pjgi - prod_matrix
#     s_pij.eliminate_zeros()
#
#     return s_pij, sigmas, rhos


def random_nn_trees(X, num_trees):
    t = AnnoyIndex(X.shape[1], 'euclidean')
    for i in range(X.shape[0]):
        t.add_item(i, X[i, :])
    t.build(num_trees)
    return t


def nearer_neighbours(X, k, num_trees=5, num_iters=2):
    r_forest = random_nn_trees(X, num_trees)
    knn = [r_forest.get_nns_by_item(i, k, include_distances=True) for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        knn[i] = (np.array(knn[i][0][1:]), np.array(knn[i][1][1:]))
    for _ in range(num_iters):
        old_knn = copy.deepcopy(knn)
        for i in range(X.shape[0]):
            ind, dist = old_knn[i]
            nn_ind = np.unique([k for j in ind for k in old_knn[j][0]
                                if (k != i) and (k not in ind)])
            ind = np.append(ind, nn_ind)
            dist = np.append(dist, l2_sq(X[[i], :], X[nn_ind, :]))
            keep = np.argsort(dist)[:k]
            knn[i] = (ind[keep], dist[keep])
    return knn


def l2_sq(x, y):
    return np.sum(np.square(x - y), axis=1, keepdims=True)


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


def smooth_knn_dist(knn, k, bandwidth=1):
    target = np.log2(k) * bandwidth
    rho = np.zeros(len(knn))
    sigmas = np.zeros(len(knn))

    means = []
    for i, (_, dist) in enumerate(knn):
        rho[i] = np.min(dist[dist > 0])
        d = dist - rho[i]
        psum = lambda sigma: np.sum(np.exp(-d / sigma))
        sigmas[i] = binary_search(psum, target)
        means.append(np.mean(dist))
        if rho[i] == 0:
            sigmas[i] = np.max(MIN_DIST_SCALE * means[-1], sigmas[i])

    mean_distance = np.mean(means)
    rho_0 = rho == 0
    if np.any(rho_0):
        sigmas[rho_0] = np.max(MIN_DIST_SCALE * mean_distance, sigmas[rho_0])
    return sigmas, rho


def compute_graph_weights(knn, sigmas, rhos):
    rho_m = rhos[:, np.newaxis]
    sig_m = sigmas[:, np.newaxis]

    vals = np.exp(- (knn[:, 1, :] - rho_m) / sig_m)
    ok = (knn[:, 1, :] - rho_m > 0.0) & (sig_m > 0.0)
    vals[~ok] = 1

    knn[:, 1, :] = vals
    return knn


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
