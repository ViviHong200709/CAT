"""
    load raw_data user_id item_id score
    train irt and save theta ,alpha, beta
    
"""
import json
import torch
from tqdm import tqdm
import numpy as np
from CAT.distillation.model import distillModel 
from CAT.distillation.tool import get_label_and_k, split_data, transform


dataset='assistment'
cdm='irt'
stg='MFI'
with_tested_info=False
postfix = '_with_tested_info' if with_tested_info else ''
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait{postfix}.json', 'r'))
utrait = trait['user']
itrait = trait['item']
label = trait['label']
k_info = trait['k_info']
# if 'tested_info' in trait:
if with_tested_info:
    tested_info= trait['tested_info']
    user_dim=np.array(tested_info).shape[-1]+1
else:
    user_dim=1
    
train_data, test_data = split_data(utrait,label,k_info,0.8)

torch.manual_seed(0)
train_set = transform(itrait,*train_data)
test_set = transform(itrait,*test_data)
k=50
embedding_dim=15
dMFI = distillModel(k,embedding_dim,user_dim,device='cuda:0')
postfix='_s'
dMFI.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip{postfix}.pt')
# dMFI.eval(test_set,itrait)
ball_embs=[]
max_embs_len=torch.tensor(0.)
for i in tqdm(itrait.items()):
    i_emb = dMFI.model.itn(torch.tensor(i[1]).to('cuda:0'))
    ball_embs.append(i_emb.tolist())
    i_emb_len = (i_emb**2).sum()
    if i_emb_len>max_embs_len:
        max_embs_len = i_emb_len

# kd_embs=[]
# for i in tqdm(itrait.items()):
#     i_emb = dMFI.model.itn(torch.tensor(i[1]).to('cuda:0'))
#     i_emb_len = (i_emb**2).sum()
#     tmp = (max_embs_len-i_emb_len)**0.5
#     kd_embs.append(torch.cat((tmp.unsqueeze(dim=0),i_emb),0))



path_prefix = f"/data/yutingh/CAT/data/{dataset}/{stg}/"

with open(f"{path_prefix}ball_trait{postfix}.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(ball_embs, ensure_ascii=False))
    
k=10
i_label={}
for i in itrait.keys():
    i_label[int(i)]=[]
for theta,top_k in zip(utrait.values(), k_info):
    for q in top_k[:k]:
        i_label[q].append(theta)
label=[sum(i)/len(i)  if len(i)!=0 else -3 for i in i_label.values()]    
with open(f"{path_prefix}item_label.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(label, ensure_ascii=False))