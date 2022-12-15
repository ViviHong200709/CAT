"""
    load raw_data user_id item_id score
    train irt and save theta ,alpha, beta
    
"""
import pandas as pd 
import CAT
import json
import torch
from tqdm import tqdm
import numpy as np
from CAT.distillation.tool import get_label_and_k
import random

seed = 0
random.seed(seed)
cdm='irt'
dataset = 'assistment'
stg='MFI'
with_tested_info=False
postfix = '_with_tested_info' if with_tested_info else ''
train_triplets = pd.read_csv(
        f'/data/yutingh/CAT/data/{dataset}/train_triples.csv', encoding='utf-8').to_records(index=False)
ckpt_path = f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_with_theta.pt'
concept_map = json.load(open(f'/data/yutingh/CAT/data/{dataset}/item_topic.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
train_data = CAT.dataset.TrainDataset(train_triplets, concept_map,
                                        metadata['num_train_students'], 
                                        metadata['num_questions'], 
                                        metadata['num_concepts'])
metadata = json.load(open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
config = {
    'learning_rate': 0.2,
    'batch_size': 2048,
    'num_epochs': 1,
    'num_dim': 1,  # for IRT or MIRT
    'device': 'cuda:0',
    # for NeuralCD
    'prednet_len1': 128,
    'prednet_len2': 64,
    # 'prednet_len1': 64,
    # 'prednet_len2': 32,
}
if cdm == 'irt':
    model = CAT.model.IRTModel(**config)
elif cdm == 'ncd':
    model = CAT.model.NCDModel(**config)
model.init_model(train_data)
model.adaptest_load(ckpt_path)

user_dict={}
for user_id in tqdm(range(train_data.num_students),'gettting theta'):
    sid = torch.LongTensor([user_id]).to(config['device'])
    theta=model.get_theta(sid)
    user_dict[user_id]=np.float(theta[0])
item_dict={}
for item_id in tqdm(range(train_data.num_questions),'gettting alpha beta'):
    qid = torch.LongTensor([item_id]).to(config['device'])
    alpha=model.get_alpha(qid)
    beta=model.get_beta(qid)
    item_dict[item_id]=[np.float(alpha[0]),np.float(beta[0])]
label,k_info,tested_info = get_label_and_k(user_dict,item_dict,50,stg,model)
if with_tested_info:
    trait_dict = {
        'user':user_dict,
        'item':item_dict,
        'label':label,
        'k_info':k_info,
        'tested_info':tested_info
    }
else:
    trait_dict = {
        'user':user_dict,
        'item':item_dict,
        'label':label,
        'k_info':k_info,
    }
    
path_prefix = f"/data/yutingh/CAT/data/{dataset}/{stg}/"

with open(f"{path_prefix}trait{postfix}.json", "w", encoding="utf-8") as f:
    # indent参数保证json数据的缩进，美观 
    # ensure_ascii=False才能输出中文，否则就是Unicode字符
    f.write(json.dumps(trait_dict, ensure_ascii=False))