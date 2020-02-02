#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd

def percent(df):
    return df[df.rating == -1].count() / df.count()

if __name__ == "__main__":
    rating = pd.read_csv("~/Data/clean_rating2.csv")

    data = rating.groupby("user_id").count()
    data = data[data.anime_id > 300]

#    n_data = rating.loc[:, ["user_id", "rating"]].groupby("user_id").apply(percent)
#    n_data = n_data[n_data.rating < 0.5]
#    indexs = list(set(n_data.index) & set(data.index))
    indexs = list(set(data.index))

    user_dict = dict(zip(sorted(indexs), [i for i in range(1, len(indexs) + 1)]))
    matrix = []
    df = pd.DataFrame()
    for user in indexs:
        print(indexs.index(user), len(indexs))
#        if indexs.index(user) == 1000:
#            break
        series = rating[rating.user_id == user]["user_id"].map(lambda x: user_dict[user])
        frame = rating[rating.user_id == user]
        frame["user_id"] = series
        df = pd.concat((df, frame), axis = 0, join = 'outer')
        
    df.to_csv('~/Data/clean_rating3.csv', index = False)
    print(df)

