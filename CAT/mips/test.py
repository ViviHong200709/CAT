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
from CAT.distillation.model import distillModel 
from CAT.mips.ball_tree import BallTree,search_metric_tree
import heapq

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
def main(dataset="assistment", cdm="irt", stg = ['MFI'], test_length = 20, ctx="cuda:4", lr=0.2, num_epoch=1, efficient=True):
    # lr=0.05 if dataset=='assistment' else 0.2
    setuplogger()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    lr_config={
        "assistment":{
            "MFI":0.15,
            "KLI":0.15,
            "Random":0.05,
            'MAAT':0.15
        },
       "junyi":{
            "MFI":0.2,
            "KLI":0.2,
            "Random":0.2,
            'MAAT':0.15
       } 
    }
    
    metadata = json.load(open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
    ckpt_path = f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}.pt'
    # read datasets
    test_triplets = pd.read_csv(f'/data/yutingh/CAT/data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(open(f'/data/yutingh/CAT/data/{dataset}/item_topic.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}

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
        config = {
            'learning_rate': lr_config[dataset][stg[i]],
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
        if cdm == 'irt':
            model = CAT.model.IRTModel(**config)
        elif cdm =='ncd':
            model = CAT.model.NCDModel(**config)
        model.init_model(test_data)
        model.adaptest_load(ckpt_path)
        test_data.reset()
        with_tested_info=False
        postfix = '_with_tested_info' if with_tested_info else ''
        if efficient:
            trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg[i]}/trait{postfix}.json', 'r'))
            postfix=''
            ball_trait = json.load(open(f'/data/yutingh/CAT/data/{dataset}/{stg[i]}/ball_trait{postfix}.json', 'r'))
            item_label = json.load(open(f"/data/yutingh/CAT/data/{dataset}/{stg[i]}/item_label.json", 'r'))
            distill_k=50
            embedding_dim=15
            if with_tested_info:
                tested_info= trait['tested_info']
                user_dim=np.array(tested_info).shape[-1]+1
            else:
                user_dim=1
            dmodel = distillModel(distill_k,embedding_dim,user_dim,device=ctx)
            dmodel.load(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{stg[i]}_ip{postfix}.pt')
        logging.info('-----------')
        logging.info(f'start adaptive testing with {strategy.name} strategy')
        logging.info('lr: ' + str(config['learning_rate']))
        logging.info(f'Iteration 0')
        res=[]
        time=0
        # starttime = datetime.datetime.now()
        logs=[]
        for sid in tqdm(test_data.data.keys(),'testing '):
            #todo
            log=[]
            if efficient:
                # time += (datetime.datetime.now() - starttime).seconds
                qids = test_data.untested[sid]
                selected_ball_trait = {}
                for k,v in enumerate(zip(item_label,ball_trait)):
                    if k in qids:
                        selected_ball_trait[k]=v
                T = BallTree(selected_ball_trait)
                # starttime = datetime.datetime.now()
            tmp_model= deepcopy(model)
            results = tmp_model.evaluate(sid, test_data)
            results['count']=0
            tmp =[list(results.values())]
            time = datetime.timedelta(microseconds=0)
            tested_info=[]
            res_q={'qid':-1,'quantity':-1,'leaves':{}}
            # theta qids
            # candidates =set()
            candidates={}
            for it in range(1, test_length + 1):
                starttime = datetime.datetime.now()
                if efficient:
                    theta = tmp_model.model.theta(torch.tensor(sid).to(ctx))
                    if user_dim==1:
                        u_emb = dmodel.model.utn(theta).tolist()
                    else:
                        if stg[i]=='KLI':
                            u_emb = dmodel.model.utn(torch.cat((theta,torch.Tensor([it]).to(ctx)),0)).tolist()
                        elif stg[i]=='MFI':
                            if len(test_data.tested[sid])==0:
                                avg_tested_emb=np.array([0,0]).tolist()
                            else:
                                avg_tested_emb = np.array([trait['item'][str(qid)] for qid in test_data.tested[sid]]).mean(axis=0).tolist()
                            avg_tested_emb.extend([it])
                            u_emb = dmodel.model.utn(torch.cat((theta,torch.Tensor(avg_tested_emb).to(ctx)),0)).tolist()
                    tested_set = set(test_data.tested[sid])
                    untested_inl = list(set(res_q['leaves'].keys())-tested_set)
                    if len(untested_inl)!=0:
                        tmp_ip = [np.dot(np.array(u_emb),np.array(ball_trait[q])) for q in untested_inl]
                        qip = heapq.nlargest(1, tmp_ip)
                        qid = untested_inl[list(map(tmp_ip.index, qip))[0]]
                        res_q['qid']=qid
                        res_q['quantity']=qip[0]
                    else:
                        res_q={'qid':-1,'quantity':-1,'leaves':{}}
                    count  = search_metric_tree(res_q, tested_set,np.array(u_emb),T)
                    qid = res_q['qid']
                    flag = False
                    print('=================================')
                    print(theta.tolist()[0])
                    for _it, candidate in candidates.items():
                        if qid in candidate:
                            print('found in ', _it)
                            flag=True
                            break
                    if not flag:
                        print(f'save in {it}')
                        candidates[it]=list(res_q['leaves'].keys())
                    # candidates.update(set(res_q['leaves'].keys()))
                    
                    # for val in candidates.values():
                    # if qid ==233:
                    #     print(qid, count)
                    # candidates=dict(zip(list(range(metadata['num_questions'],metadata['num_questions']+it)),[0]*it))
                    # count  = search_metric_tree(candidates,np.array(u_emb),T)
                    # untested_qids = set(candidates.keys())-set(test_data.tested[sid])
                    # print(it, untested_qids)
                    # if len(untested_qids) == 1:
                    #     max_score = 0 
                    #     for k,v in candidates.items():
                    #         if k in untested_qids:
                    #             if v>max_score:
                    #                 qid=k
                    #                 max_score=v
                    # else:
                    #    qid = strategy.adaptest_select(tmp_model, sid, test_data,item_candidates=untested_qids) 
                else:
                     qid = strategy.adaptest_select(tmp_model, sid, test_data)
                log.append([float(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))),float(tmp_model.get_alpha(torch.LongTensor([qid]).to(ctx))[0]),float(tmp_model.get_beta(torch.LongTensor([qid]).to(ctx))[0])])
                # print(log)
                test_data.apply_selection(sid, qid)
                tmp_model.adaptest_update(sid, qid, test_data)
                time += (datetime.datetime.now() - starttime)
                results = tmp_model.evaluate(sid, test_data)
                del results['cov']
                results['time']=time.seconds+time.microseconds*1e-6
                results['count']=count
                tmp.append(list(results.values()))
                # print(tmp)
            res.append(tmp)
            logs.append(log)
        with open(f'/data/yutingh/CAT/data/{dataset}/{stg[i]}/log.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(logs, ensure_ascii=False))

        # time +=  (datetime.datetime.now() - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2,1,0), dim=-1).tolist()
        exp_info={
            f"{stg[i]}": ['acc']+res[0],
            " ": ['auc']+res[1],
            "": ['time']+res[2],
            "  ": ['count']+res[3]
            }    
        exp_info = pd.DataFrame(exp_info)
        idx= ['']
        idx.extend(range(0,test_length+1))
        exp_info.index=idx
        
        selected_num = [1,3,5,10,20]
        short_acc = [acc for i,acc in enumerate(res[0]) if i in selected_num]
        short_auc = [auc for i,auc in enumerate(res[1]) if i in selected_num]
        short_exp_info={
            f"{stg[i]}": ['acc']+short_acc,
            " ": ['auc']+short_auc,
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        idx= ['']
        idx.extend(selected_num)
        short_exp_info.index=idx

        print(exp_info.transpose())
        print(short_exp_info.transpose())
        
        df1 = df1.append(short_exp_info.transpose())
        df = df.append(exp_info.transpose())
    df1.to_csv(
        f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_short_{'_'.join(stg)}.csv")
    df.to_csv(f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{'_'.join(stg)}.csv")  

if __name__ == '__main__':
    import fire

    fire.Fire(main)
