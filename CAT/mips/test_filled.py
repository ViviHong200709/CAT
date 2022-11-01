import sys
sys.path.append('..')
import CAT
import json
import torch
import logging
import datetime
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from CAT.distillation.MFI.model import dMFIModel 
from CAT.mips.ball_tree import BallTree,search_metric_tree

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main(dataset="junyi", cdm="irt", stg = ['Random'], test_length = 20, ctx="cuda:0", lr=0.2, num_epoch=1, efficient=False):
    setuplogger()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = {
        'learning_rate': lr,
        'batch_size': 2048,
        'num_epochs': num_epoch,
        'num_dim': 1, # for IRT or MIRT
        'device': ctx,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        # 'prednet_len1': 64,
        # 'prednet_len2': 32,
    }
    metadata = json.load(open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
    ckpt_path = f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}.pt'
    # read datasets
    # test_triplets = pd.read_csv(f'/data/yutingh/CAT/data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
    test_triplets = pd.read_csv(f'/data/yutingh/CAT/data/{dataset}/test_filled_triplets.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(open(f'/data/yutingh/CAT/data/{dataset}/item_topic.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}
    if efficient:
        ball_trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/ball_trait.json', 'r'))
        T = BallTree(dict(zip(range(len(ball_trait)),ball_trait)))
        distill_k=50
        embedding_dim=15
        dMFI = dMFIModel(distill_k,embedding_dim,device='cuda:0')
        dMFI.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_ip.pt')

    test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                            metadata['num_test_students'], 
                                            metadata['num_questions'], 
                                            metadata['num_concepts'])
    strategy_dict = {
        'Random' : CAT.strategy.RandomStrategy(),
        'MFI' : CAT.strategy.MFIStrategy(),
        'KLI' : CAT.strategy.KLIStrategy(),
        'MAAT' : CAT.strategy.MAATStrategy(),
    } 
   
    strategies = [strategy_dict[i] for i in stg]
    df = pd.DataFrame() 
    df1 = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        if cdm == 'irt':
            model = CAT.model.IRTModel(**config)
        elif cdm =='ncd':
            model = CAT.model.NCDModel(**config)
        model.init_model(test_data)
        model.adaptest_load(ckpt_path)
        test_data.reset()
        
        logging.info('-----------')
        logging.info(f'start adaptive testing with {strategy.name} strategy')

        logging.info(f'Iteration 0')
        res=[]
        starttime = datetime.datetime.now()
        for sid in tqdm(test_data.data.keys(),'testing '):
            tmp_model= deepcopy(model)
            results = tmp_model.evaluate(sid, test_data)
            tmp =[list(results.values())]
            for it in range(1, test_length + 1):
                if efficient:
                    u_emb = dMFI.model.utn(tmp_model.model.theta(torch.tensor(sid).to(ctx))).tolist()
                    candidates=dict(zip(list(range(it)),[0]*it))
                    search_metric_tree(candidates,np.array(u_emb),T)
                    untested_qids = set(candidates.keys())-set(test_data.tested[sid])
                    max_score = 0 
                    for k,v in candidates.items():
                        if k in untested_qids:
                            if v>max_score:
                                qid=k
                                max_score=v
                else:
                    qid = strategy.adaptest_select(tmp_model, sid, test_data)
                test_data.apply_selection(sid, qid)
                tmp_model.adaptest_update(sid, qid, test_data)
                results = tmp_model.evaluate(sid, test_data)
                tmp.append(list(results.values()))
            res.append(tmp)
        endtime = datetime.datetime.now()
        time =  (endtime - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2,1,0), dim=-1).tolist()
        exp_info={
            f"{stg[i]}": [f'{time}', 'acc']+res[0],
            " ": ['', 'auc']+res[1],
            # f"": ['', 'Cov']+res[2]
            }    
        exp_info = pd.DataFrame(exp_info)
        idx= ['','']
        idx.extend(range(0,test_length+1))
        exp_info.index=idx
        
        selected_num = [1,3,5,10,20]
        short_acc = [acc for i,acc in enumerate(res[0]) if i in selected_num]
        short_auc = [auc for i,auc in enumerate(res[1]) if i in selected_num]
        short_exp_info={
            f"{stg[i]}": [f'{time}', 'acc']+short_acc,
            " ": ['', 'auc']+short_auc,
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        idx= ['','']
        idx.extend(selected_num)
        short_exp_info.index=idx

        print(exp_info.transpose())
        print(short_exp_info.transpose())
        
        # df1 = df1.append(pd.DataFrame(short_exp_info))
        df1 = df1.append(short_exp_info.transpose())
        df = df.append(exp_info.transpose())
    df1.to_csv(
        f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_short_{'_'.join(stg)}.csv")
    df.to_csv(f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{'_'.join(stg)}.csv")  

def save_exp_res():
    pass

if __name__ == '__main__':
    import fire

    fire.Fire(main)
