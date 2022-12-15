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
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

# lr=0.25
# num_epoch=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main(dataset="junyi", cdm="irt", stg = ['MFI'], test_length = 20, ctx="cpu", lr=0.2, num_epoch=8):
    setuplogger()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # tensorboard
    log_dir = f"/home/yutingh/BDAA_CAT/logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}/"
    writer = SummaryWriter(log_dir)

    # choose dataset here
    # modify config here
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
    metadata = json.load(open(f'/home/yutingh/BDAA_CAT/data/{dataset}/metadata.json', 'r'))
    # modify checkpoint path here
    ckpt_path = f'/home/yutingh/BDAA_CAT/ckpt/{dataset}/{cdm}1.pt'
    # ckpt_path = f'/home/yutingh/BDAA_CAT/ckpt/{dataset}/{cdm}_{metadata["min_train_len"]}_{metadata["min_test_len"]}.pt'
    # ckpt_path = f'/home/yutingh/BDAA_CAT/ckpt/{dataset}/{cdm}_{metadata["min_train_len"]}_{metadata["max_train_len"]}_{metadata["min_test_len"]}.pt'
    # read datasets
    test_triplets = pd.read_csv(f'/home/yutingh/BDAA_CAT/data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
    support_triplets = pd.read_csv(f'/home/yutingh/BDAA_CAT/data/{dataset}/support_triples.csv', encoding='utf-8').to_records(index=False)
    query_triplets = pd.read_csv(f'/home/yutingh/BDAA_CAT/data/{dataset}/query_triples.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(open(f'/home/yutingh/BDAA_CAT/data/{dataset}/item_topic.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}

    test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                            metadata['num_test_students'], 
                                            metadata['num_questions'], 
                                            metadata['num_concepts'])
    support_data = CAT.dataset.AdapTestDataset(support_triplets, concept_map,
                                            metadata['num_test_students'], 
                                            metadata['num_questions'], 
                                            metadata['num_concepts'])
    query_data = CAT.dataset.AdapTestDataset(query_triplets, concept_map,
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
            # tmp_model=model
            results = tmp_model.evaluate(sid, test_data)
            tmp =[list(results.values())]
            for it in range(1, test_length + 1):
                # tmp_model= deepcopy(model)
                qid = strategy.adaptest_select(tmp_model, sid, test_data)
                test_data.apply_selection(sid, qid)
                # ,update_lr=lr*0.97**it
                tmp_model.adaptest_update(sid, qid, test_data)
                results = tmp_model.evaluate(sid, test_data)
                tmp.append(list(results.values()))
                # print(tmp_model.model.k_difficulty.weight)
                # print(tmp_model.model.beta.weight)
            res.append(tmp)
        endtime = datetime.datetime.now()
        time =  (endtime - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2,1,0), dim=-1).tolist()
        exp_info={
            # f"min_train_len:{metadata['min_train_len']}":[stg[i],'ACC']+res[0],
            # f"min_test_len:{metadata['min_test_len']}":['','AUC']+res[1],
            f"{stg[i]}": [f'{time}', 'ACC']+res[0],
            " ": ['', 'AUC']+res[1],
            f"": ['', 'Cov']+res[2]
            }    
        exp_info = pd.DataFrame(exp_info)
        idx= ['','']
        idx.extend(range(0,test_length+1))
        exp_info.index=idx
        selected_num = [1,3,5,10,20]
        short_exp_info={
            'acc':[acc for i,acc in enumerate(res[0]) if i in selected_num],
            'auc':[auc for i,auc in enumerate(res[1]) if i in selected_num]
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        short_exp_info.index = selected_num
        print(exp_info.transpose())
        print(short_exp_info)
        df1 = df1.append(pd.DataFrame(short_exp_info))
        # print(df1)
        df1 = df1.append(pd.Series(name=' '))

        df = df.append(exp_info.transpose())
        df =df.append(pd.Series(name = ' '))    
    df1.to_csv(
        f"/home/yutingh/BDAA_CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_short_{'_'.join(stg)}.csv")
    df.to_csv(f"/home/yutingh/BDAA_CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{'_'.join(stg)}.csv")  

if __name__ == '__main__':
    import fire

    fire.Fire(main)
