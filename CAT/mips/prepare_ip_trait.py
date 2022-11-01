"""
    load raw_data user_id item_id score
    train irt and save theta ,alpha, beta
    
"""
import json
import torch
from tqdm import tqdm
import numpy as np
from CAT.distillation.MFI.model import dMFIModel 
from CAT.distillation.MFI.tool import get_label_and_k, split_data, transform

dataset='assistment'
cdm='irt'
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
label = trait['label']
k_fisher = trait['k_fisher']
train_data, test_data = split_data(utrait,label,k_fisher,0.8)

torch.manual_seed(0)
train_set = transform(itrait,*train_data)
test_set = transform(itrait,*test_data)
k=50
embedding_dim=15
dMFI = dMFIModel(k,embedding_dim,device='cuda:0')
dMFI.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_ip.pt')
# dMFI.eval(test_set,itrait)
ball_embs=[]
max_embs_len=torch.tensor(0.)
for i in tqdm(itrait.items()):
    i_emb = dMFI.model.itn(torch.tensor(i[1]).to('cuda:0'))
    ball_embs.append(i_emb.tolist())
    i_emb_len = (i_emb**2).sum()
    if i_emb_len>max_embs_len:
        max_embs_len = i_emb_len

kd_embs=[]
for i in tqdm(itrait.items()):
    i_emb = dMFI.model.itn(torch.tensor(i[1]).to('cuda:0'))
    i_emb_len = (i_emb**2).sum()
    tmp = (max_embs_len-i_emb_len)**0.5
    kd_embs.append(torch.cat((tmp.unsqueeze(dim=0),i_emb),0))

path_prefix = f"/data/yutingh/CAT/data/{dataset}/"

with open(f"{path_prefix}ball_trait.json", "w", encoding="utf-8") as f:
    # indent参数保证json数据的缩进，美观 
    # ensure_ascii=False才能输出中文，否则就是Unicode字符
    f.write(json.dumps(ball_embs, ensure_ascii=False))