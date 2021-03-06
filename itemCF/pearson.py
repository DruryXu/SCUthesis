#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np

import os
import json
from collections import defaultdict
if __name__ == "__main__":
    anime = pd.read_csv("~/Data/anime.csv")
    rating = pd.read_csv("~/Data/clean_rating2.csv")

#    anime_id = list(set(rating.anime_id.tolist()))
#    user_anime_matrix = defaultdict(list)
#    for aid in anime_id:
#        user_anime_matrix[aid] = rating[(rating.anime_id == aid) & (rating.rating > 0)].user_id.tolist()
#
#    with open("user_anime_matrix2.json", "w", encoding = "utf-8") as f:
#        f.write(json.dumps(user_anime_matrix))
    with open("user_anime_matrix2.json", "r", encoding="utf-8") as f:
        user_anime_matrix = json.loads(f.read())

    target_user = 19
    target_item = rating[rating.user_id == target_user].anime_id.tolist()
    all_item = list(set(rating.anime_id.tolist()))
    similarity_matrix = dict()

    print(len(target_item), len(all_item))
    for aid in target_item:
        for item in all_item:
            item_set = user_anime_matrix[str(item)]
            aid_set = user_anime_matrix[str(aid)]
            user_id = list(set(item_set) & set(aid_set))
            Sum, rui2, ruj2 = 0, 0, 0
            print(aid, item, len(user_id))
            if item in target_item or len(user_id) == 0:
                continue
            for uid in user_id:
#                print(rating[(rating.anime_id == aid) & (rating.user_id == uid)])
#                print(rating[(rating.anime_id == item) & (rating.user_id == uid)])
#                print(np.mean(rating[(rating.anime_id == aid) & (rating.rating > 0)].rating.tolist()))
#                print(rating[(rating.anime_id == item) & (rating.rating > 0)])
                rui = rating[(rating.anime_id == aid) & (rating.user_id == uid)].rating.tolist()[0] - np.mean(rating[(rating.anime_id == aid) & (rating.rating > 0)].rating.tolist())
                ruj = rating[(rating.anime_id == item) & (rating.user_id == uid)].rating.tolist()[0] - np.mean(rating[(rating.anime_id == item) & (rating.rating > 0)].rating.tolist())
                Sum += rui * ruj
                rui2 += rui ** 2
                ruj2 += ruj ** 2

            print(Sum, rui2, ruj2)
            if rui2 != 0 and ruj2 != 0:
                similarity_matrix[(aid, item)] = Sum / ((rui2 ** 0.5) * (ruj2 ** 0.5))
               
                
            
