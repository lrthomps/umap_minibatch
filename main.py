# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

import umap_var as umapp
import umap

import time


if __name__ == '__main__':
    # X, y = load_boston(return_X_y=True)
    # X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

    # Load data from https://www.openml.org/d/554
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    cdict, embs = pickle.load(open('/home/lara/code/cloud/flightwc.p', 'br'))
    words, counts = np.array(list(cdict.keys())), np.array(list(cdict.values()))
    words = words[counts > 3]
    counts = counts[counts > 3]

    ic = np.argsort(counts)
    words = words[ic]
    counts = counts[ic]
    X = embs.loc[words].values

    # X = pickle.load(open('/home/lara/code/cloud/embeddings.p', 'br')).values
    print(X.shape)
    np.random.seed(1879)
    # ir = np.random.choice(X.shape[0], 1000, replace=False)

    reducer = umapp.UMAP(init='random', n_neighbors=25)

    t0 = time.time()
    embedding = reducer.fit_transform(X, counts)
    print(time.time() - t0)

    f = plt.figure(figsize=[10, 5])
    ax = f.add_subplot(111)
    ax.plot([np.min(embedding[:, 0]), np.max(embedding[:, 0] + 6)],
            [np.min(embedding[:, 1]), np.max(embedding[:, 1])], alpha=0)
    # plt.plot(embedding[:, 0], embedding[:, 1], '.', c=y)
    # sns.scatterplot(, embedding[:, 1]) #, hue=y[ir], palette='bright')
    for i_w, (w, num) in enumerate(zip(words, counts)):
        t = ax.text(embedding[i_w, 0], embedding[i_w, 1], w, fontsize=10 + 5 * num ** 0.25,
                    alpha=0.001 * (i_w + 100 - len(words)) ** 2, color='black', family='monospace')
    # plt.axis("off")
    plt.show()

