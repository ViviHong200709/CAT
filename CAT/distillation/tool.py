import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
def split_data(utrait,label,k_fisher,rate=1):
    max_len=int(rate*len(utrait.keys()))
    u_train={}
    u_test={}
    for u,trait in utrait.items():
        if int(u)>=max_len:
            u_test[u]=trait
        else:
            u_train[u]=trait
    return ((u_train,label[:max_len],k_fisher[:max_len]),(u_test,label[max_len:],k_fisher[max_len:]))
    
def get_label_and_k(user_trait,item_trait,k):
    labels=[]
    k_fishers=[]
    for theta in tqdm(user_trait.values(),f'get top {k} items'):
        k_fisher, label = get_k_fisher(k, theta, item_trait)
        labels.append(label)
        k_fishers.append(k_fisher)
    return labels,k_fishers

def transform(item_trait, user_trait,labels,k_fishers):
    for theta, label, k_fisher in zip(user_trait.values(),labels,k_fishers):
        itrait = list(item_trait.values())
        yield pack_batch([[
            theta,
            itrait,
            label,
            k_fisher
            # topitrait,
            # topkitems,
            # tailitrait,
            # tailkitems    
        ]])
            # batch=[]
    # if batch:
    #     yield pack_batch(batch)

def pack_batch(batch):
    theta, itrait, label, k_fisher= zip(*batch)
    return (
        Tensor(theta), Tensor(itrait), Tensor(label), k_fisher
    )

def get_k_fisher(k,theta,items):
    fisher_arr = []
    for qid,(alpha,beta) in items.items():
        pred = alpha * theta + beta
        pred = torch.sigmoid(torch.tensor(pred))
        # pred = 1 / (1 + np.exp(-pred))
        q = 1 - pred
        fisher_info = float((q*pred*(alpha ** 2)).numpy())
        fisher_arr.append((fisher_info,qid))
    fisher_arr_sorted = sorted(fisher_arr, reverse=True)
    return [i[1] for i in fisher_arr_sorted[:k]],[i[0]for i in fisher_arr]