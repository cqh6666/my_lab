# encoding=gbk

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import load_digits


if __name__ == '__main__':
    X, _ = load_digits(return_X_y=True)
    lle = LocallyLinearEmbedding(n_components=2)
    lle.fit(X)
    tra = lle.transform(X[0:2])
    print(tra)


