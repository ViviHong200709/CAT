# from soupsieve import select
# from tensorboardX import SummaryWriter
# import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# from copy import deepcopy
import numpy as np
import datetime
import logging
import torch
import json
import CAT
# from pydoc_data.topics import topics
import sys
import os

sys.path.append('..')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def main(dataset="junyi",
         cdm="irt",
         stg=['Random'],
         test_length=20,
         ctx="cuda:0",
         lr=0.2,
         with_cognitive_structure=True,
         save=True):
    # 'Random',"MFI","KLI",
    # , 'MFI','KLI','MAAT'MFI
    setuplogger()
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    # tensorboard
    # log_dir = f"/data/yutingh/CAT/logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}/"
    # writer = SummaryWriter(log_dir)

    # choose dataset here
    # modify config here
    config = {
        'learning_rate': lr,
        'batch_size': 2048,
        'num_epochs': 1,
        'num_dim': 1,  # for IRT or MIRT
        'device': ctx,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        # 'prednet_len1': 64,
        # 'prednet_len2': 32,
    }

    cs_config = {
        # "max_depth":10,
        "decay": 0.5,
        "max_cover_rate": 0.2,
        "k": 5
    }
    metadata = json.load(
        open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
    # modify checkpoint path here
    ckpt_path = f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}.pt'
    # read datasets
    # support_triplets = pd.read_csv(
    #     f'/data/yutingh/CAT/data/{dataset}/support_triples.csv', encoding='utf-8').to_records(index=False)
    # query_triplets = pd.read_csv(
    #     f'/data/yutingh/CAT/data/{dataset}/query_triples.csv', encoding='utf-8').to_records(index=False)
    # train_triplets = pd.read_csv(
    #     f'/data/yutingh/CAT/data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
    test_triplets = pd.read_csv(
        f'/data/yutingh/CAT/data/{dataset}/test_triples.csv',
        encoding='utf-8').to_records(index=False)
    # test_triplets = pd.read_csv(
    # f'/data/yutingh/CAT/data/{dataset}/test_filled_triplets.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(
        open(f'/data/yutingh/CAT/data/{dataset}/item_topic.json', 'r'))
    concept_map = {int(k): v for k, v in concept_map.items()}

    # support_data = CAT.dataset.AdapTestDataset(support_triplets, concept_map,
    #                                            metadata['num_test_students'],
    #                                            metadata['num_questions'],
    #                                            metadata['num_concepts'])
    # query_data = CAT.dataset.AdapTestDataset(query_triplets, concept_map,
    #                                          metadata['num_test_students'],
    #                                          metadata['num_questions'],
    #                                          metadata['num_concepts'])
    # train_data = CAT.dataset.AdapTestDataset(train_triplets, concept_map,
    #                                          metadata['num_train_students'],
    #                                          metadata['num_questions'],
    #                                          metadata['num_concepts'])
    test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                            metadata['num_test_students'],
                                            metadata['num_questions'],
                                            metadata['num_concepts'])

    strategy_dict = {
        'Random': CAT.strategy.RandomStrategy(),
        'MFI': CAT.strategy.MFIStrategy(),
        'KLI': CAT.strategy.KLIStrategy(),
        'MAAT': CAT.strategy.MAATStrategy(),
    }
    if cdm == 'irt':
        model = CAT.model.IRTModel(**config)
    elif cdm == 'ncd':
        model = CAT.model.NCDModel(**config)
    model.init_model(test_data)
    model.adaptest_load(ckpt_path)
    if with_cognitive_structure:
        filter = CAT.strategy.CSStrategy()
        topic_concept = json.load(
            open(f'/data/yutingh/CAT/data/{dataset}/topic_concept.json', 'r'))
        edges = json.load(
            open(f'/data/yutingh/CAT/data/{dataset}/edges.json', 'r'))
        concept_item = json.load(
            open(f'/data/yutingh/CAT/data/{dataset}/concept_item.json', 'r'))
        item_concept = json.load(
            open(f'/data/yutingh/CAT/data/{dataset}/item_concept.json', 'r'))
        item_diff = {}
        item_disc = {}
        for item in item_concept.keys():
            qid = torch.LongTensor([int(item)]).to(config['device'])
            if cdm == 'irt':
                item_diff[item] = model.get_beta(qid)[0]
                item_disc[item] = model.get_alpha(qid)[0]
        cognitive_structure = CAT.cognitive_structure.CogntiveStructure(
            edges, topic_concept, concept_item, item_concept, item_diff,
            item_disc, **cs_config)
        item_set = set(range(metadata['num_questions']))
    strategies = [strategy_dict[i] for i in stg]
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        model.init_model(test_data)
        model.adaptest_load(ckpt_path)
        test_data.reset()

        logging.info('-----------')
        logging.info(f'start adaptive testing with {strategy.name} strategy')

        logging.info(f'Iteration 0')
        res = []
        starttime = datetime.datetime.now()
        for sid, log in tqdm(test_data.data.items(), 'testing '):
            if with_cognitive_structure:
                missing_qids = item_set.difference(set(log.keys()))
                cognitive_structure.reset(missing_qids)
                # show_fig(sid,log,item_concept,edges)
                # continue
            tmp_model = model
            if cdm == 'ncd':
                # optimizer = torch.optim.Adam(
                optimizer = torch.optim.SGD(
                    tmp_model.model.student_emb.parameters(), lr=lr)
                # scheduler = None
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=1,
                                                            gamma=0.99)
            else:
                # optimizer = None
                scheduler = None
                optimizer = torch.optim.Adam(
                    tmp_model.model.theta.parameters(), lr=lr)
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            results = tmp_model.evaluate(sid, test_data)
            tmp = [list(results.values())]
            for it in range(1, test_length + 1):
                if with_cognitive_structure:
                    theta = model.get_theta(
                        torch.LongTensor([sid]).to(config['device']))[0]
                    item_candidates = filter.adaptest_select(
                        tmp_model, sid, test_data, cognitive_structure, theta)
                    if len(item_candidates) == 0:
                        # print('last iteration:',it)
                        tmp.extend([tmp[-1]] * (test_length - it + 1))
                        break
                    qid = strategy.adaptest_select(tmp_model, sid, test_data,
                                                   item_candidates)
                    # print(concept_map[qid])
                    cognitive_structure.update(qid, test_data.data[sid][qid])
                else:
                    qid = strategy.adaptest_select(tmp_model, sid, test_data)
                # print('sid:', sid, 'it:', it)

                test_data.apply_selection(sid, qid)
                # optimizer = torch.optim.Adam(tmp_model.model.student_emb.parameters(), lr=lr)
                # optimizer=None
                tmp_model.adaptest_update(sid,
                                          qid,
                                          test_data,
                                          optimizer=optimizer,
                                          scheduler=scheduler)
                # tmp_model.adaptest_update(
                #     # sid, qid, test_data, update_lr=lr)
                #     sid, qid, test_data, update_lr=lr*0.98**it)
                results = tmp_model.evaluate(sid, test_data)
                tmp.append(list(results.values()))
            res.append(tmp)
            # long running
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2, 1, 0), dim=-1).tolist()
        prefix = "cs_" if with_cognitive_structure else ""

        exp_info = {
            # f"min_train_len:{metadata['min_train_len']}":[stg[i],'ACC']+res[0],
            # f"min_test_len:{metadata['min_test_len']}":['','AUC']+res[1],
            f"{prefix}{stg[i]}": [f'{time}', 'ACC'] + res[0],
            " ": ['', 'AUC'] + res[1],
            f"": ['', 'Cov'] + res[2]
        }

        exp_info = pd.DataFrame(exp_info)
        idx = ['', '']
        idx.extend(range(0, test_length + 1))
        exp_info.index = idx
        # print(exp_info)
        selected_num = [5, 10, 20]
        short_exp_info = {
            'acc': [acc for i, acc in enumerate(res[0]) if i in selected_num],
            'auc': [auc for i, auc in enumerate(res[1]) if i in selected_num]
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        short_exp_info.index = selected_num
        print(exp_info.transpose())
        print(short_exp_info)

        df1 = pd.concat([df1, pd.DataFrame(short_exp_info)])
        # print(df1)
        df1 = pd.concat([df1, pd.Series(name=' ', dtype=pd.StringDtype())])
        df = pd.concat([df, exp_info.transpose()])
        # df.reset_index(drop=True, inplace=True)
        # df = df.concat(pd.Series(name=' ', dtype=pd.StringDtype()))
    if save:
        df1.to_csv(
            f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}{prefix}_short_{'_'.join(stg)}.csv"
        )
        df.to_csv(
            f"/data/yutingh/CAT/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}{prefix}_{'_'.join(stg)}.csv"
        )


def show_fig(sid, log, item_concept, edges):
    concept_log = {}
    for qid, correct in log.items():
        for concept in item_concept[str(qid)]:
            if concept in concept_log:
                concept_log[concept].append(correct)
            else:
                concept_log[concept] = [correct]
    for concept, correct in concept_log.items():
        concept_log[concept] = sum(correct) / len(correct)
    # print(concept_log)
    # return
    import networkx as nx
    import graphviz
    G = nx.DiGraph()
    G.add_edges_from(edges)
    g = graphviz.Digraph('G', filename=f'sim_test_{sid}')
    g.attr(rankdir='LR')
    for node in G.nodes():
        if node not in concept_log:
            g.attr('node', style='filled', color='#7a7374')
        elif concept_log[node] < 0.5:
            g.attr('node', style='filled', color='#ec2c64')
        else:
            g.attr('node', style='filled', color='#20894d')
        g.node(str(node))
    for edge in G.edges():
        g.edge(str(edge[0]), str(edge[1]))
    try:
        g.view()
    except:
        print(f'Graph {sid} generated')
        pass


if __name__ == '__main__':
    import fire

    fire.Fire(main)
