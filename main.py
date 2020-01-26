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
    with open('positive.json', 'r', encoding='utf-8') as f:
        positive = json.loads(f.read())
    
    similarity, c  = defaultdict(int), defaultdict(int)
    target_user = 19
    targetNum = 0
    for k,v in positive.items():
        if target_user in v:
            targetNum += 1
            for i in range(len(v)):
                c[(target_user, i)] += 1
    
    for i in c.keys():
        tmp = 0
        for j in c.keys():
            if i[1] in j:
                tmp += c[j]
        
        similarity[i] = c[i] / (targetNum * tmp) ** 0.5

    print(len(set(similarity.values())))
#    target_user = 19
#    all_users = rating.user_id.unique().tolist()
#    for user in all_users[:100]:
#        cuser, ctarget_user, both = 0, 0, 0
#        for k,v in positive.items():
#            if user in v:
#                cuser += 1
#            if target_user in v:
#                ctarget_user += 1
#            if user in v and target_user in v:
#                both += 1
#
#        if cuser != 0 and ctarget_user != 0:    
#            similarity[(target_user, user)] = both / (cuser * ctarget_user) ** 0.5
#
#    print([i[0][1] for i in sorted(similarity.items(), key = lambda x: x[1], reverse = True)])
