"""     
    get theta ,get a b, [theta][alpha,beta]
    compute MFI by getMFI top-k
    compute by dot production
    conpute loss
"""
from CAT.distillation.model import dMFIModel 
from CAT.distillation.tool import transform,split_data
import torch
import json
import numpy as np
dataset='assistment'
cdm='irt'
stg='MFI'
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait_with_tested_info.json', 'r'))
# stg='KLI'
# trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
# print(utrait,itrait)
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
epoch=38
lr=0.005 if dataset=='assistment' else 0.01
print(f'lr: {lr}')
dMFI = dMFIModel(k,embedding_dim,user_dim,device='cuda:4')
dMFI.train(train_set,test_set,itrait,epoch=epoch,lr=lr)
dMFI.save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip_with_tested_info.pt')




