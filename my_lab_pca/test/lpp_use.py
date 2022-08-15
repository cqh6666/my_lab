# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lpp_use
   Description:   ...
   Author:        cqh
   date:          2022/8/10 19:42
-------------------------------------------------
   Change Activity:
                  2022/8/10:
-------------------------------------------------
"""
__author__ = 'cqh'

from lpproj import LocalityPreservingProjection as LPP
from sklearn.datasets import make_blobs
import time
from sklearn.decomposition import PCA


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res

    return inner


@get_time
def run(reduction):
    x_new = reduction.fit_transform(X)
    return x_new


if __name__ == '__main__':
    X, y = make_blobs(100000, n_features=3000, centers=4,
                      cluster_std=8, random_state=42)
    lpp = LPP(n_components=2)
    run(lpp)
    print("========")
    pca = PCA(n_components=2)
    run(pca)
