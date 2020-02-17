#!/usr/bin/env python
# coding=utf-8
from gensim.models import Word2Vec        
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#def getData():
#    rating = pd.read_csv("~/Data/rating.csv")
#    df_rating_train, df_rating_test = train_test_split(rating,
#                                                       stratify = rating["user_id"],
#                                                       random_state = 15688,
#                                                       test_size = 0.3)
#    return df_rating_train, df_rating_test


def rating_splitter(df):
    df['liked'] = np.where(df["rating"] > 5, 1, 0)
    df['anime_id'] = df['anime_id'].astype('str')
    gp_user_like = df.groupby(['liked', 'user_id'])

    return [gp_user_like.get_group(gp)['anime_id'].tolist() for gp in gp_user_like.groups]

if __name__ == "__main__":
#    df_train, df_test = getData()    
    rating = pd.read_csv("~/Data/rating.csv")
    splitted_animes = rating_splitter(rating)

    print(len(splitted_animes))
    import random
    for anime_list in splitted_animes:
        random.shuffle(anime_list)
    
    import datetime
    start = datetime.datetime.now()
    model = Word2Vec(sentences = splitted_animes,
                     iter = 5,
                     min_count = 10,
                     size = 200,
                     workers = 4,
                     sg = 1,
                     hs = 0,
                     negative = 5,
                     window = 9999999)
    print("Time passed:" + str(datetime.datetime.now() - start))
    model.save('item2Vec_20200216')


