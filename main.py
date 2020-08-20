from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import umap_var as umapp
import umap
import time


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    reducer = umapp.UMAP(init='random',
                         n_neighbors=15)

    t0 = time.time()
    embedding = reducer.fit_transform(X)
    print(time.time() - t0)

    plt.plot(embedding[:, 0], embedding[:, 1], 'o')
    plt.show()

