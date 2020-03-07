#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import random
import math

def get_dummy(matrix):
    item_list = sorted(list(set(matrix[:, 1])))
    user_list = sorted(list(set(matrix[:, 0])))
    item_dummies = [i for i in range(len(item_list))]
    user_dummies = [i for i in range(len(user_list))]
    item_dict = dict(zip(item_list, item_dummies))
    user_dict = dict(zip(user_list, user_dummies))
    for i in range(matrix.shape[0]):
        matrix[i, 0] = user_dict[matrix[i, 0]]
        matrix[i, 1] = item_dict[matrix[i, 1]]

    return matrix

if __name__ == "__main__":
    anime = pd.read_csv("~/Data/anime.csv")
    rating = pd.read_csv("~/Data/clean_rating2.csv")

    users = list(set(rating.user_id.tolist()))
    items = [i for i in range(len(set(rating.anime_id.tolist())))]
    
    user_item_matrix = rating[rating.rating > 0].loc[:, ["user_id", "anime_id"]].values
    user_item_matrix = get_dummy(user_item_matrix)
    rating_matrix = rating[rating.rating > 0].rating.values

    F, alpha, lam_bda = 100, 0.02, 0.01
    max_iter, n, k = 30, rating_matrix.shape[0], 1 / math.sqrt(F)
    P = np.array([[random.random() * k for i in range(F)] for j in range(len(set(user_item_matrix[:, 0])))])
    Q = np.array([[random.random() * k for i in range(len(set(user_item_matrix[:, 1])))] for j in range(F)])

    print(user_item_matrix)
    print(n)
    #gradient descent
    for step in range(max_iter):
        print(step + 1)
        for i in range(n):
            user = user_item_matrix[i, 0]
            item = user_item_matrix[i, 1]
            rui = rating_matrix[i]

            eui = rui - P[user, :].dot(Q[:, item])
            for f in range(F):
                P[user, f] += alpha * (eui * Q[f, item] - lam_bda * P[user, f])
                Q[f, item] += alpha * (eui * P[user, f] - lam_bda * Q[f, item])

        alpha *= 0.9
    
    print(P, Q)
