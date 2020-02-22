#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import random
import collections
import math
import sys
import time
class item2VecDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        super(item2VecDataset, self).__init__()
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask = None):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction = 'none', weight = mask)
        return res.mean(dim = 1)

#class CBOW(nn.Module):
#    def __init__(self):
#        super(CBOW, self).__init__()
#        self.embedding1 = nn.Embedding(num_embeddings = size, embedding_dim = embed_size)
#        self.embedding2 = nn.Embedding(num_embeddings = size, embedding_dim = embed_size)
#
#    def forward(self, context, center_negative):
#        v = self.embedding1(context)
#        v = v.mean(dim = 0)
#        u = self.embedding2(center_negative)
#        pred = torch.bmm(v, u.permute(0, 2, 1))
#        return pred

def skip_gram(center, contexts_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

def train_item2Vec(net, lr, num_epochs, loss, data_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('train on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
#            print(center.shape, context_negative.shape)
            pred = skip_gram(center, context_negative, net[0], net[1])
#            print(pred.shape, mask.shape, label.shape)
            l = (loss(pred.view(label.shape), label, mask.view(label.shape)) * mask.shape[1] / mask.float().sum(dim = 1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1

        print('epoch %d, loss %f, time %.2fs' % (epoch + 1, l_sum / n, time.time() - start))

def get_corpus(path):
    ratings = pd.read_csv(path)
    positive_rating = ratings[ratings.rating >= 5]
    anime_list = positive_rating['anime_id'].tolist() 
    counter = collections.Counter(anime_list)

    idx_to_anime = list(set(anime_list))
    anime_to_idx = {anime_id: idx for idx, anime_id in enumerate(idx_to_anime)}
    
    gp = positive_rating.groupby("user_id")
    corpus = [list(map(lambda x: anime_to_idx[x], gp.get_group(user_id)['anime_id'].tolist())) for user_id, _ in gp]
    corpus = [[aid for aid in st if subsampling(aid, counter, len(anime_list), idx_to_anime)] for st in corpus]
    return corpus, anime_to_idx, idx_to_anime, counter

def subsampling(aid, counter, size, idx_to_anime):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 * size / counter[idx_to_anime[aid]])

def get_centers_and_contexts(corpus):
    centers, contexts = [], []
    for animes in corpus:
        if len(animes) < 2:
            continue
        centers += animes
        for idx, _ in enumerate(animes):
            contexts.append(animes[:idx] + animes[idx + 1:])

    return centers, contexts

def negative_sampling(contexts, weights, K):
    negatives, neg_candidates = [], []
    all_animes = list(range(len(weights)))
    for context in contexts:
#        print(contexts.index(context))
        negs, i = [], 0
        neg_candidates = random.choices(all_animes, weights, k = int(1e5))
        while len(negs) < K:
            if neg_candidates[i] in context:
                i += 1
                continue
            else:
                negs.append(neg_candidates[i])

        negatives.append(negs)

    return negatives

def select_batch(data):
    max_len = max([len(context) + len(negative) for _, context, negative in data])
    context_negatives, masks, labels, centers = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        context_negatives.append(context + negative + [0] * (max_len - cur_len))
        centers.append(center)
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        labels.append([1] * len(context) + [0] * (max_len - len(context)))

    return (torch.tensor(centers).view(-1, 1), torch.tensor(context_negatives), torch.tensor(masks), torch.tensor(labels))


if __name__ == "__main__":

    anime = pd.read_csv('~/Data/anime.csv')
    corpus, anime_to_idx, idx_to_anime, counter = get_corpus('~/Data/clean_rating3.csv')
    print(sum([len(st) for st in corpus]), len(corpus), len(idx_to_anime))
    centers, contexts = get_centers_and_contexts(corpus)
    weights = [counter[aid] ** 0.75 for aid in idx_to_anime]
    print(len(centers), len(contexts))
    negatives = negative_sampling(contexts, weights, 5)
    print(len(centers), len(contexts), len(negatives))
    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4

    dataset = item2VecDataset(centers, contexts, negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle = True, collate_fn = select_batch, num_workers = num_workers)

    embed_size = 100
#    for batch in data_iter:
#        for name, data in zip(['centers', 'context_negatives', 'masks', 'labels'], batch):
#            print(name, data.shape)
#        break
    net = nn.Sequential(
        nn.Embedding(num_embeddings = len(idx_to_anime), embedding_dim = embed_size),
        nn.Embedding(num_embeddings = len(idx_to_anime), embedding_dim = embed_size) 
    )
    train_item2Vec(net, 0.1, 10, BinaryCrossEntropyLoss(), data_iter)

    torch.save(net.state_dict(), 'skip_gram_complete.pt')
    W = net[0].weight.data
    x = W[10]
    print(anime[anime['anime_id'] == 5114])
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim = 1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k = 6)
    topk = topk.cpu().numpy()
    for i in topk[1:]:
        print('cosine sim = %.3f' % (cos[i]))
        print(anime[anime['anime_id'] == idx_to_anime[i]])

