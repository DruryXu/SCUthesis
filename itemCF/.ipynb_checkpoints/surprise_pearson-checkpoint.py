#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNWithMeans
if __name__ == "__main__":

    rating = pd.read_csv("~/Data/clean_rating2.csv")
    reader = Reader(rating_scale=(-1, 10))
    data = Dataset.load_from_df(rating, reader)

    sim_options = {'name': 'pearson',
                   'user_based': False}
    algo = KNNWithMeans(sim_options = sim_options)
    print(cross_validate(algo, data, cv = 2))
    
    
