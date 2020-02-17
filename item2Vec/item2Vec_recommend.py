#!/usr/bin/env python
# coding=utf-8
from gensim.models import Word2Vec
import pandas as pd

def recommender(positive_list = None, negative_list = None, topn = 5):
    model = Word2Vec.load("item2Vec_20200216")
    recommend_anime_ls = ["241"]
    for animeId, prob in model.wv.most_similar_cosmul(positive = positive_list, negative = negative_list, topn = topn):
        recommend_anime_ls.append(animeId)

    return recommend_anime_ls

if __name__ == "__main__":
    anime = pd.read_csv("~/Data/anime.csv")
    ls = recommender(positive_list = ["241"], topn = 5)
    print(anime[anime["anime_id"].isin(ls)])
