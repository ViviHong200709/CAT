from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
    
class dMFI(nn.Module):
    def __init__(self,embedding_dim):
        # self.prednet_input_len =1
        super(dMFI, self).__init__()
        self.utn = nn.Sequential(
            nn.Linear(1, 256), nn.Sigmoid(
            ),
            nn.Linear(256, 128), nn.Sigmoid(
            ), 
            nn.Linear(128, embedding_dim), nn.Sigmoid(
            ) )
        # nn.Dropout(p=0.5)
        
        self.itn = nn.Sequential(
            nn.Linear(2, 256), nn.Sigmoid(
            ),
            nn.Linear(256, 128), nn.Sigmoid(
            ), 
            nn.Linear(128, embedding_dim), nn.Sigmoid(
            ) )
    
    def forward(self,u,i):
        user =self.utn(u)
        item =self.itn(i)
        return (user * item).sum(dim=-1, keepdim=True)
        # return user*item
    
class dMFIModel(object):
    def __init__(self, k, embedding_dim,device):
        self.model = dMFI(embedding_dim)
        # 20 1 1 
        self.k = k
        self.device=device
    
    def train(self,train_data,test_data,item_pool,lr=0.01,epoch=2):
        self.model=self.model.to(self.device)
        train_data=list(train_data)
        test_data=list(test_data)
        self.items_n = len(item_pool.keys())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss = []
        for epoch_i in range(epoch):
            for data in tqdm(train_data):
                utrait,itrait,label,_=data
                itrait = itrait.squeeze()
                u_loss: torch.Tensor  = torch.tensor(0.).to(self.device)
                utrait:torch.Tensor = torch.tensor([utrait]*self.items_n).unsqueeze(-1).to(self.device)
                itrait: torch.Tensor = itrait.to(self.device)
                label: torch.Tensor = label.to(self.device)
                score = self.model(utrait,itrait).squeeze(-1)
                u_loss=((score-label)**2).sum()
                loss.append(u_loss.item())
                optimizer.zero_grad()
                u_loss.backward()
                optimizer.step()
                # print(float(np.mean(loss)))
                # self.eval(valid_data,item_pool)
            print(f'Epoch {epoch_i}:',float(np.mean(loss)))
            self.eval(test_data,item_pool)
    
    def load(self, path):
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(path), strict=False)
    
    def save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items()
                        if 'utn' in k or 'itn' in k}
        torch.save(model_dict, path)
    
    def eval(self,valid_data,item_pool):
        k_nums=[1,3,5,10,30,50]
        recall = [[]for i in k_nums]
        for data in tqdm(valid_data,'testing'):
            utrait,_,__,k_fisher=data
            k_items,k_DCG = self.getkitems(utrait,item_pool)
            # k_fisher
            k_fisher=k_fisher[0]
            for i,k in enumerate(k_nums):
                i_kitems = set(k_items[:k]).intersection(set(k_fisher[:k]))
                recall[i].append(len(i_kitems)/k)
        for i,k in enumerate(k_nums):
            print(f'recall@{k}: ',np.mean(recall[i]))
        
    # def get_k_fisher(self,theta,items):
    #     fisher_arr = []
    #     for qid,(alpha,beta) in items.items():
    #         pred = alpha * theta + beta
    #         pred = torch.sigmoid(torch.tensor(pred))
    #         q = 1 - pred
    #         fisher_info = (q*pred*(alpha ** 2)).numpy()
    #         fisher_arr.append((fisher_info,qid))
    #     fisher_arr_sorted = sorted(fisher_arr, reverse=True)
    #     return [i[1] for i in fisher_arr_sorted[:self.k]]
    
    def estimate_rank(self,score,theta,items):
        with torch.no_grad():
            self.model.eval()
            items_arr = list(range(self.items_n))
            np.random.shuffle(items_arr)
            samples = items_arr[0:self.sample_num]
            theta = torch.tensor([theta]*len(samples))
            utrait:torch.Tensor = theta.unsqueeze(-1).to(self.device)
            itrait:torch.Tensor = torch.tensor([items[str(sample)] for sample in samples]).to(device)
            scores = self.model(utrait,itrait).squeeze(-1)
            s_rank  = len([i for i in scores if i>score])
            self.model.train()
        return s_rank*(self.items_n-1)/self.sample_num+1
    
    
    def getkitems(self, utrait,item_pool):
        with torch.no_grad():
            self.model.eval()
            item_n =len(item_pool.keys())
            utrait:torch.Tensor = torch.tensor([utrait]*item_n).unsqueeze(-1).to(self.device)
            itrait:torch.Tensor = torch.tensor(list(item_pool.values())).to(self.device)
            scores = self.model(utrait,itrait).squeeze(-1)
            tmp = list(zip(scores.tolist(),item_pool.keys()))
            tmp_sorted = sorted(tmp, reverse=True)
            self.model.train()
        return [int(i[1]) for i in tmp_sorted[:self.k]],[e[0]/np.log(i+2) for i,e in enumerate(tmp_sorted[:self.k])]
        

    
    
    
    
        