import CAT
import sys
import json
import logging
import numpy as np
import pandas as pd

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

def run(cdm, model, dataset, *args, **kwargs):
    # read datasets
    train_triplets = pd.read_csv(f'/data/yutingh/CAT/data/{dataset}/train_triples.csv', encoding='utf-8').to_records(index=False)
    valid_triplets = pd.read_csv(f'/data/yutingh/CAT/data/{dataset}/valid_triples.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(open(f'/data/yutingh/CAT/data/{dataset}/item_topic.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}
    metadata = json.load(open(f'/data/yutingh/CAT/data/{dataset}/metadata.json', 'r'))
    train_data = CAT.dataset.TrainDataset(train_triplets, concept_map,
                                        metadata['num_train_students'], 
                                        metadata['num_questions'], 
                                        metadata['num_concepts'])
    valid_data = CAT.dataset.TrainDataset(valid_triplets, concept_map,
                    metadata['num_train_students'], 
                    metadata['num_questions'], 
                    metadata['num_concepts'])
                                        # define model here
    
    # train model
    model.init_model(train_data)
    model.train(train_data, test_data = valid_data)
    # save model
    
    # model.adaptest_save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{metadata["min_train_len"]}_{metadata["max_train_len"]}_{metadata["min_test_len"]}.pt')
    # model.adaptest_save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_{metadata["min_train_len"]}_{metadata["min_test_len"]}.pt')
    model.adaptest_save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}.pt')
    model.adaptest_save(f'/data/yutingh/CAT/ckpt/{dataset}/{cdm}_with_theta.pt',save_theta=True)

def main(dataset="assistment", cdm="irt", ctx="cuda:3", num_epochs=1, num_dim = 1, lr=0.025):
    setuplogger()
    num_epochs=15 if dataset=="assistment" else 1
    config = {
        'learning_rate': lr,
        'batch_size': 2048,
        'num_epochs': num_epochs,
        'num_dim': num_dim, # for IRT or MIRT
        'device': ctx,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        # 'prednet_len1': 128,
        # 'prednet_len2': 64,
        # 'prednet_len1': 64,
        # 'prednet_len2': 32,
    }
    if cdm == 'irt':
        model = CAT.model.IRTModel(**config)
    elif cdm == 'ncd':
        model = CAT.model.NCDModel(**config)

    run(
        cdm = cdm,
        model = model,
        dataset = dataset,
        **config
    )

if __name__ == '__main__':
    import fire

    fire.Fire(main)
