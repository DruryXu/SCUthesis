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

#    positive = {}
#    sort = sorted(set(rating.anime_id.tolist()))
#    for i in sort:
#        new_rating = rating[rating.anime_id == i]
#        user = new_rating[new_rating.rating > 5].user_id.tolist()
#        if len(user) > 0:
#            positive[i] = user
#
#    with open('positive.json', "w", encoding='utf-8') as f:
#        f.write(json.dumps(positive))



#    with open('positive.json', 'r', encoding='utf-8') as f:
#        positive = json.loads(f.read())
#    
#    similarity, c  = defaultdict(int), defaultdict(int)
#    target_user = 19
#    targetNum = 0
#    for k,v in positive.items():
#        if target_user in v:
#            targetNum += 1
#            for i in v:
#                c[i] += 1
#
#    print("hello")    
#    for i in c.keys():
#        new_rating = rating[rating.user_id == i]
#        tmp = len(new_rating[new_rating.rating > 5])
##        print(i, tmp)
#        similarity[i] = c[i] / (targetNum * tmp) ** 0.5
#    
#    print("hello")
##    sort = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
#    with open('similarity.json', 'w', encoding='utf-8') as f:
#        f.write(json.dumps(similarity))
    with open('similarity.json', 'r', encoding='utf-8') as f:
        similarity = json.loads(f.read())

    sort = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
    similist = [sort[i][0] for i in range(1, 6)]
    
    likeAnime = []
    target_user = 19
    for u in similist:
        new_rating = rating[rating.user_id == int(u)]
        tmp = new_rating[new_rating.rating > 5].anime_id.tolist()
        likeAnime += tmp        

    targetAnime = set(rating[rating.user_id == target_user].anime_id.tolist())
    reclist = set(likeAnime) & (targetAnime ^ set(likeAnime))
    
    recName = []
    for i in reclist:
        recName += anime[anime.anime_id == i].name.tolist()

    print(recName)
