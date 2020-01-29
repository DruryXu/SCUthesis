#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd

import os
import json
from collections import defaultdict
if __name__ == "__main__":
    anime = pd.read_csv("~/Data/anime.csv")
    rating = pd.read_csv("~/Data/rating.csv")

    with open("positive.json", "r", encoding = "utf-8") as f:
        positive = json.loads(f.read())

    targetUser = 19
#    targetItem = list(positive.items())[0][0]
    targetAnime = rating[(rating.user_id == targetUser) & (rating.rating > 5)].anime_id.tolist()
    targetRating = rating[(rating.user_id == targetUser) & (rating.rating > 5)].rating.tolist()
    recList = []
    for index, a in enumerate(targetAnime):
        similarity = defaultdict(int)    
        for i in positive.items():
            bothLike = len(set(i[1]) & set(positive[str(a)]))
            if bothLike != 0:
                similarity[i[0]] = bothLike / (len(i[1]) * len(positive[str(a)])) ** 0.5

        sort = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
        recList += [(a, sort[i][0], sort[i][1] * targetRating[index]) for i in range(1, 4)]

    sortRec = sorted(recList, key = lambda x: x[2], reverse = True)
    print(sortRec)
