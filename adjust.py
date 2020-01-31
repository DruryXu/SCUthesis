#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd

def percent(df):
    return df[df.rating == -1].count() / df.count()

if __name__ == "__main__":
    rating = pd.read_csv("~/Data/rating.csv")

    data = rating.groupby("user_id").count()
    data = data[data.anime_id > 50]

    n_data = rating.loc[:, ["user_id", "rating"]].groupby("user_id").apply(percent)
    n_data = n_data[n_data.rating < 0.5]
    indexs = list(set(n_data.index) & set(data.index))

    matrix = []
    df = pd.concat((rating[rating.user_id == indexs[0]], rating[rating.user_id == indexs[1]]), axis = 0, join = 'outer')
    for user in range(3, len(indexs)):
        print(user, len(indexs))
        df = pd.concat((df, rating[rating.user_id == indexs[user]]), axis = 0, join = 'outer')
        
    df.to_csv('~/Data/clean_rating.csv', index = False)
    print(df)

