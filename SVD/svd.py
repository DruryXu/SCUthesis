#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd

def compress(data):
    user_num = len(set(data[:, 0]))
    item_num = len(set(data[:, 1]))
    user_dict = dict(zip(sorted(list(set(data[:, 0]))), [i for i in range(user_num)]))
    item_dict = dict(zip(sorted(list(set(data[:, 1]))), [i for i in range(item_num)]))

    for i in data:
        i[0] = user_dict[i[0]]
        i[1] = item_dict[i[1]]

    return data

if __name__ == "__main__":
    rating = pd.read_csv("~/Data/clean_rating2.csv")

    data = compress(rating[rating.rating > 0].values)
    n_users = rating.user_id.nunique()
    n_items = rating.anime_id.nunique()

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size = 0.3)

    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data:
        train_data_matrix[line[0], line[1]] = line[2]

    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    
    print(train_data_matrix.shape)
    u, s, vt = svds(train_data_matrix, k = 20)
    s_diag_matrix = np.diag(s)
    svd_prediction = np.dot(np.dot(u, s_diag_matrix), vt)

    svd_prediction[svd_prediction < 0] = 0
    svd_prediction[svd_prediction > 10] = 10
    print(svd_prediction)
