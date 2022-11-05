import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
import vegas
from scipy import integrate
import random
import functools

def split_data(utrait,label,k_fisher,rate=1,tested_info=None):
    max_len=int(rate*len(utrait.keys()))
    u_train={}
    u_test={}
    for u,trait in utrait.items():
        if int(u)>=max_len:
            u_test[u]=trait
        else:
            u_train[u]=trait
    if tested_info:
        return ((u_train,label[:max_len],k_fisher[:max_len],tested_info[:max_len]),(u_test,label[max_len:],k_fisher[max_len:],tested_info[max_len:]))
    else:
        return ((u_train,label[:max_len],k_fisher[:max_len]),(u_test,label[max_len:],k_fisher[max_len:]))
    
def get_label_and_k(user_trait,item_trait,k,stg="MFI"):
    labels=[]
    k_infos=[]
    tested_ns=[]
    for theta in tqdm(user_trait.values(),f'get top {k} items'):
        if stg=='MFI':
            k_info,label = get_k_fisher(k, theta, item_trait)
        elif stg=='KLI':
            k_info,label,tested_n= get_k_kli(k, theta, item_trait)
            tested_ns.append(tested_n)
        k_infos.append(k_info)
        labels.append(label)
    return labels,k_infos,tested_ns

def transform(item_trait, user_trait,labels,k_fishers,tested_infos=None):
    if tested_infos:
        for theta, label, k_fisher,tested_info in zip(user_trait.values(),labels,k_fishers,tested_infos):
            itrait = list(item_trait.values())
            item_n = len(itrait)
            yield pack_batch([
                torch.tensor(list(zip([theta]*item_n,tested_info))),
                itrait,
                label,
                k_fisher
            ])
    else:
        for theta, label, k_fisher in zip(user_trait.values(),labels,k_fishers):
            itrait = list(item_trait.values())
            item_n = len(itrait)
            yield pack_batch([
                torch.tensor([theta]*item_n).unsqueeze(-1),
                itrait,
                label,
                k_fisher
            ])

def pack_batch(batch):
    theta, itrait, label, k_fisher= batch
    return (
        theta, Tensor(itrait), Tensor(label), k_fisher
    )

def get_k_fisher(k, theta, items):
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


           
def get_k_kli(k, theta, items):
    items_n=len(items.keys())
    ns = [random.randint(1,20) for i in range(items_n)]
    dim = 1
    res_arr = []
    for (qid,(alpha, beta)),n in zip(items.items(),ns):
        if type(alpha) == float:
            alpha = np.array([alpha])
        if type(theta) == float:
            theta = np.array([theta])
        pred_estimate = np.matmul(alpha.T, theta) + beta
        pred_estimate = 1 / (1 + np.exp(-pred_estimate))
        def kli(x):
            if type(x) == float:
                x = np.array([x])
            pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1 - pred
            q = 1 - pred
            return pred_estimate * np.log(pred_estimate / pred) + \
                q_estimate * np.log((q_estimate / q)) 
        c = 3
        boundaries = [
            [theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        if len(boundaries) == 1:
            # KLI
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            res_arr.append((v,qid))
        else:
            # MKLI
            integ = vegas.Integrator(boundaries)
            result = integ(kli, nitn=10, neval=1000)
            res_arr.append((result.mean,qid))
    res_arr_sorted = sorted(res_arr, reverse=True)
    return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],ns
