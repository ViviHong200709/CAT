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
dataset='assistment'
cdm='irt'
stg='KLI'
trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
# print(utrait,itrait)
label = trait['label']
k_info = trait['k_info']
if stg=='KLI':
    tested_info= trait['tested_info']
    train_data, test_data = split_data(utrait,label,k_info,0.8,tested_info)
else:
    train_data, test_data = split_data(utrait,label,k_info,0.8)
    
torch.manual_seed(0)
train_set = transform(itrait,*train_data)
test_set = transform(itrait,*test_data)
k=50
embedding_dim=15  
lr=0.005 if dataset=='assistment' else 0.01
print(f'lr: {lr}')
user_dim=2 if stg =='KLI'else 1
dMFI = dMFIModel(k,embedding_dim,user_dim,device='cuda:4')
dMFI.train(train_set,test_set,itrait,epoch=30,lr=lr)
dMFI.save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg}_ip.pt')




