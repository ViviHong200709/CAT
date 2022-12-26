"""     
    get theta ,get a b, [theta][alpha,beta]
    compute MFI by getMFI top-k
    compute by dot production
    conpute loss
"""
from CAT.distillation.model_similarity import distillModel 
from CAT.distillation.tool import transform,split_data
import torch
import json
import numpy as np
dataset='ifytek'
cdm='irt'
stg='MFI'
with_tested_info=False
postfix = '_with_tested_info' if with_tested_info else ''
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait{postfix}.json', 'r'))
# stg='KLI'
# trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
# print(utrait,itrait)
label = trait['label']
k_info = trait['k_info']
# if 'tested_info' in trait:
if with_tested_info:
    tested_info= trait['tested_info']
    train_data, test_data = split_data(utrait,label,k_info,0.8,tested_info)
    user_dim=np.array(tested_info).shape[-1]+1
else:
    user_dim=1
    train_data, test_data = split_data(utrait,label,k_info,0.8)
    
torch.manual_seed(0)
train_set = transform(itrait,*train_data)
# for i in train_set:
#     print(i)
#     break

test_set = transform(itrait,*test_data)
k=50
embedding_dim=15  
epoch=11
lr=0.005 if dataset=='assistment' else 0.01
print(f'lr: {lr}')
model = distillModel(k,embedding_dim,user_dim,device='cuda:4')
# model.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip.pt')
model.train(train_set,test_set,itrait,epoch=epoch,lr=lr)
model.save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip_s.pt')




