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
    
def get_label_and_k(user_trait,item_trait,k,stg="MFI", model=None):
    labels=[]
    k_infos=[]
    tested_infos=[]
    # thetas=[0.,0.2,0.4,0.6,0.8,1.0]
    # for theta in thetas:
    #     k_info, label, tested_info = get_k_fisher(20, theta, item_trait)
    #     tested_infos.append(tested_info)
    #     # print('\n',theta,'\n',k_info)
    #     # print("===============")
    #     k_infos.append(k_info)
    #     labels.append(label)
    # print(k_infos)
    # return labels,k_infos,tested_infos
    for sid, theta in tqdm(user_trait.items(),f'get top {k} items'):
        # print(theta)
        if stg=='MFI':
            k_info, label, tested_info = get_k_fisher(k, theta, item_trait)
            tested_infos.append(tested_info)
            # print('\n',theta,'\n',k_info)
            # print("===============")
        elif stg=='KLI':
            k_info,label,tested_info= get_k_kli(k, theta, item_trait)
            tested_infos.append(tested_info)
        elif stg=='MAAT':
            get_k_emc(k, sid,theta, item_trait, model)
        k_infos.append(k_info)
        labels.append(label)
    return labels,k_infos,tested_infos

def transform(item_trait, user_trait,labels,k_fishers,tested_infos=None):
    if tested_infos:
        for theta, label, k_fisher,tested_info in zip(user_trait.values(),labels,k_fishers,tested_infos):
            itrait = list(item_trait.values())
            item_n = len(itrait)
            user_embs=[]
            for tmp in tested_info:
                user_emb = [theta]
                if type(tmp) == list:
                    user_emb.extend(tmp)
                else:
                    user_emb.append(tmp)
                user_embs.append(user_emb)
            yield pack_batch([
                torch.tensor(user_embs),
                # torch.tensor(list(zip([theta]*item_n,tested_info))),
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
    return [i[1] for i in fisher_arr_sorted[:k]],[i[0]for i in fisher_arr],[]

# def get_k_fisher(k, theta, items):
#     fisher_arr = []
#     items_n=len(items.keys())
#     ns = [random.randint(0,19) for i in range(items_n)]
#     tested_qids = [random.sample(list(range(0,20)),n) for n in ns]
#     avg_embs = np.array(list(items.values())).mean(axis=0)
#     p=0.002
#     # p=0.01
#     avg_tested_embs=[]
#     for tested_qid, (qid,(alpha,beta)) in zip(tested_qids,items.items()):
#         # tested_qid
#         if len(tested_qid)==0:
#             avg_tested_emb=np.array([0,0])
#         else:
#             avg_tested_emb = np.array([items[qid] for qid in tested_qid]).mean(axis=0)
#         item_emb=[alpha,beta]
#         pred = alpha * theta + beta
#         pred = torch.sigmoid(torch.tensor(pred))
#         # pred = 1 / (1 + np.exp(-pred))
#         q = 1 - pred
#         diff = ((item_emb-avg_tested_emb)**2).sum()
#         sim = ((item_emb-avg_embs)**2).sum()
#         fisher_info = float((q*pred*(alpha ** 2)).numpy()) + p*diff/sim
#         # print(float((q*pred*(alpha ** 2)).numpy()),0.01*diff/sim)
#         fisher_arr.append((fisher_info,qid,0.05*diff/sim))
#         avg_tested_embs.append(avg_tested_emb.tolist())
#     fisher_arr_sorted = sorted(fisher_arr, reverse=True)
#     tested_info=[]
#     for avg_tested_emb,n in zip(avg_tested_embs,ns):
#         avg_tested_emb.extend([n])
#         tested_info.append(avg_tested_emb)
#     # print([i[0] for i in fisher_arr_sorted[:k]],'\n',[i[2] for i in fisher_arr_sorted[:k]])
#     return [i[1] for i in fisher_arr_sorted[:k]],[i[0]for i in fisher_arr],tested_info

def get_k_emc(k,sid,theta,items,model):
    epochs = model.config['num_epochs']
    lr = model.config['learning_rate']
    device = model.config['device']
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
    res_arr = []
    for qid,(alpha,beta) in items.items():
        for name, param in model.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = model.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = model.model(student_id, question_id)
            loss = model._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = model.model.theta.weight.data.clone()
        model.model.theta.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = model.model(student_id, question_id)
            loss = model._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = model.model.theta.weight.data.clone()
        # model.model.theta.weight.data.copy_(original_weights)

        for param in model.model.parameters():
            param.requires_grad = True
        
        if type(alpha) == float:
            alpha = np.array([alpha])
        if type(theta) == float:
            theta = np.array([theta])
        pred = np.matmul(alpha.T, theta) + beta
        pred = 1 / (1 + np.exp(-pred))
        result = pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
        res_arr.append((result,qid))
    res_arr_sorted = sorted(res_arr, reverse=True)
    return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr]

           
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
