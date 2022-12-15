from CAT.distillation.model import distillModel 
from CAT.distillation.tool import transform, split_data
import torch
import json
import numpy as np
from torch import Tensor

dataset='assistment'
cdm='irt'
stg='MFI'
with_tested_info=False
postfix = '_with_tested_info' if with_tested_info else ''
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait{postfix}.json', 'r'))
# stg='KLI'
# trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
label = trait['label']
k_info = trait['k_info']
if 'tested_info' in trait:
    tested_info= trait['tested_info']
    train_data, test_data = split_data(utrait,label,k_info,0.8,tested_info)
    user_dim=np.array(tested_info).shape[-1]+1
else:
    user_dim=1
    train_data, test_data = split_data(utrait,label,k_info,0.8)
    
    
torch.manual_seed(0)
train_set = transform(itrait,*train_data)
test_set = transform(itrait,*test_data)
k=50
embedding_dim=15  
epoch=25
lr=0.005 if dataset=='assistment' else 0.01
print(f'lr: {lr}')
model = distillModel(k,embedding_dim,user_dim,device='cuda:2')
model.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip{postfix}.pt')
model.train_rank(train_set,test_set,itrait,epoch=epoch,lr=lr)
# model.save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip{postfix}.pt')




