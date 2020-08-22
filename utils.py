from annoy import AnnoyIndex
import copy
import numpy as np
import random
from scipy.optimize import curve_fit


BOXMIN = np.array([10, 2])
TOLERANCE = 1e-5
MAX_GRAD = 4.0
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
        embedding = np.random.uniform(0, 4, size=(n_vertices, n_components))
    else:
        raise NotImplementedError('Only random initialization of embedding is implemented')

    embedding = optimize_layout(
        embedding,
        knn,
        n_epochs,
        knn.shape[1] // 4,
        n_vertices,
        a, b,
        gamma,
        initial_alpha,
        negative_sample_rate
    )
    return embedding


def optimize_layout(
        embedding,
        knn,
        n_epochs,
        minibatch,
        n_vertices,
        a, b,
        gamma=1,
        initial_alpha=1,
        negative_sample_rate=5,
):
    def batch(knn, k=minibatch):
        ir = np.random.choice(knn.shape[1], k)
        return knn[:, ir]

    ir = minibatch * 2

    def neg_samples(k):
        nonlocal ir
        next = np.arange(ir, ir + k)
        ir += k
        while ir >= n_vertices:
            next[next >= n_vertices] -= n_vertices
            ir -= n_vertices
        return next

    alpha = initial_alpha
    n_eff = n_epochs * knn.shape[1] // minibatch
    da = alpha / n_eff

    for _ in range(n_eff):
        js, kpos, weight = batch(knn)
        keep = random.random() <= weight
        kpos = kpos[keep].astype(int)
        if len(kpos) == 0:
            continue
        js = js[keep].astype(int)

        dpos = dpos_dy(embedding[js], embedding[kpos], a, b)
        embedding[kpos, :] += -alpha * dpos

        for _ in range(negative_sample_rate):
            kneg = neg_samples(len(kpos))
            while True:
                drop = (kneg == js) | (kneg == kpos)
                if ~np.any(drop):
                    break
                kneg[drop] = neg_samples(np.sum(drop))

            dneg = dneg_dy(embedding[js], embedding[kneg], gamma, a, b)
            dpos += dneg
            embedding[kneg] += -alpha * dneg
        embedding[js, :] += alpha * (dpos + dneg)

        alpha -= da

    return embedding


def clip(val):
    return np.minimum(MAX_GRAD, np.maximum(-MAX_GRAD, val))


def dpos_dy(current, others, a, b, boxmin=BOXMIN):
    del_x = current - others
    d_sq = dist(del_x)
    if boxmin is not None:
        d = np.sqrt(d_sq)

        hat_x = del_x / d
        case = np.argmin(boxmin * hat_x, axis=1)
        dbox = boxmin[case] - np.abs(del_x[np.arange(len(case)), case])
        ob = dbox > 0

    grad_coeff = -2.0 * a * b * (np.power(d_sq, b - 1.0)
                                 / (a * np.power(d_sq, b) + 1.0))
    grad = clip(grad_coeff * del_x)
    if boxmin is not None:
        grad[ob, case[ob]] = -np.sign(grad[ob, case[ob]]) * dbox[ob]
    return grad


def dneg_dy(current, others, gamma, a, b, boxmin=BOXMIN):
    del_x = current - others
    d_sq = dist(del_x)
    if boxmin is not None:
        d = np.sqrt(d_sq)

        hat_x = del_x / d
        case = np.argmin(boxmin * hat_x, axis=1)
        dbox = boxmin[case] - np.abs(del_x[np.arange(len(case)), case])
        ob = dbox > 0

    grad_coeff = 2.0 * b * gamma / (
            (0.001 + d_sq) * (a * np.power(d_sq, b) + 1))
    grad_coeff[d_sq <= 0] = 0
    grad_d = clip(grad_coeff * del_x)
    grad_d[grad_coeff[:, 0] <= 0, :] = MAX_GRAD
    if boxmin is not None:
        grad_d[ob, case[ob]] = np.sign(grad_d[ob, case[ob]]) * dbox[ob]
    return grad_d


def build_graph_nocoo(X, n_neighbors, counts=None):
    num_iters = max(0, 3 - int(np.log10(X.shape[0])))
    knn_d = nearer_neighbours(X, n_neighbors, num_iters=num_iters)

    sigmas, rhos = smooth_knn_dist(knn_d, n_neighbors)
    knn_w = compute_graph_weights(np.array(knn_d), sigmas, rhos)

    if counts is not None:
        knn_w[:, 1, :] *= counts[:, np.newaxis] / np.max(counts)

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
        knn_list[-1] = np.concatenate([i*np.ones((1, knn_list[-1].shape[1])), knn_list[-1]], axis=0)

    knn_m = np.concatenate(knn_list, axis=1)
    return knn_m, sigmas, rhos


def random_nn_trees(X, num_trees):
    t = AnnoyIndex(X.shape[1], 'euclidean')
    for i in range(X.shape[0]):
        t.add_item(i, X[i, :])
    t.build(num_trees)
    return t


def nearer_neighbours(X, k, num_trees=5, num_iters=0):
    r_forest = random_nn_trees(X, num_trees)
    knn = [r_forest.get_nns_by_item(i, k, include_distances=True) for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        knn[i] = (np.array(knn[i][0][1:]), np.array(knn[i][1][1:]))
    for _ in range(num_iters):
        old_knn = copy.deepcopy(knn)
        for i in range(X.shape[0]):
            ind, d = old_knn[i]
            nn_ind = np.unique([k for j in ind for k in old_knn[j][0]
                                if (k != i) and (k not in ind)])
            ind = np.append(ind, nn_ind)
            d = np.append(d, dist(X[[i], :] - X[nn_ind, :]))
            keep = np.argsort(d)[:k]
            knn[i] = (ind[keep], d[keep])
    return knn


def dist(x_y, metric='sqeuclidean'):
    d_sq = np.sum(np.square(x_y), axis=1, keepdims=True)
    if metric == 'euclidean':
        return np.sqrt(d_sq)
    return d_sq
    # return cdist(x, y, metric=metric)


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
