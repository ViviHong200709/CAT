import json 
import torch
import numpy as np
import pyswarms as ps
dataset ='junyi'
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
theta = utrait['0']
tmp = itrait['0']
item_n = len(itrait.keys())

def fisher(item):
    if item.sum()!=1:
        return 300
    alpha = itrait[0]
    beta = itrait[1]
    pred = alpha * theta + beta
    pred = torch.sigmoid(torch.tensor(pred))
    q = 1 - pred
    fisher_info = float((q*pred*(alpha ** 2)).numpy())
    return 1

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':5, 'p':2}
optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=item_n, options=options)
cost, pos = optimizer.optimize(fisher, iters=2000)