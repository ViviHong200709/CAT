import json
from CAT.mips.ball_tree import BallTree,search_metric_tree
from CAT.distillation.model import distillModel 
import numpy as np
import torch
import datetime
from tqdm import tqdm
import heapq
dataset='assistment'
stg="MFI"
cdm='irt'
postfix='_s'
path_prefix = f"/data/yutingh/CAT/data/{dataset}/{stg}/"
# with open(f"{path_prefix}ball_trait{postfix}.json", "w", encoding="utf-8") as f:
ball_trait = json.load(open(f"{path_prefix}ball_trait{postfix}.json", 'r'))
item_label = json.load(open(f"{path_prefix}item_label.json", 'r'))
# model = BallTree(dict(zip(range(len(ball_trait)),ball_trait)),item_label)

model = BallTree(dict(zip(range(len(ball_trait)),list(zip(item_label,ball_trait)))))
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
        
distill_k=50
embedding_dim=15
ctx='cuda:0'
user_dim=1
dMFI = distillModel(distill_k,embedding_dim,user_dim, device=ctx)
dMFI.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip{postfix}.pt')
# dMFI.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_ip{postfix}.pt')

k=20
efficient = False
starttime= datetime.datetime.now()
for _,q in tqdm(utrait.items()):
    u_emb = dMFI.model.utn(torch.tensor([q]).to(ctx)).tolist()
    if efficient:
        candidates=dict(zip(list(range(k)),[0]*k))
        search_metric_tree(candidates,np.array(u_emb),model)
    else:
        if k==1:
            qid = np.argmax(np.array([np.dot(np.array(u_emb),i_emb) for i_emb in np.array(ball_trait)]))
        else:
            tmp = [np.dot(np.array(u_emb),i_emb) for i_emb in np.array(ball_trait)]
            qips = heapq.nlargest(k, tmp)
            qids = list(map(tmp.index, qips))
    # print(qids,candidates)
endtime= datetime.datetime.now()
time = (endtime - starttime).seconds
print(time)




