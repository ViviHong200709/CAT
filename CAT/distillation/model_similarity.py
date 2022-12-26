from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
    
class distill(nn.Module):
    def __init__(self,embedding_dim,user_dim):
        # self.prednet_input_len =1
        super(distill, self).__init__()
        self.utn = nn.Sequential(
            nn.Linear(user_dim, 256), nn.Sigmoid(
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
    
class distillModel(object):
    def __init__(self, k, embedding_dim, user_dim,device):
        self.model = distill(embedding_dim,user_dim)
        # 20 1 1 
        self.k = k
        self.device=device
        self.batch_size=32
        self.warmp_up_ratio = 0.55
        self.l=torch.tensor(1.0).to(self.device)
        self.b=torch.tensor(10000.0).to(self.device)
        
    def get_distance_data(self,train_data,item_pool):
        # dissmilarity
        selected=set()
        for data in train_data:
            top_k  = data[3]
            selected.update(set(top_k))
        all_qs = set([int(i) for i in item_pool.keys()])
        unselected=all_qs-selected
        selected_itrait =  [item_pool[str(i)] for i in selected]
        unselected_itrait =  [item_pool[str(i)] for i in unselected]
        d_i=[]
        d_j=[]
        for i in selected_itrait:
            for j in unselected_itrait:
                d_i.append(i)
                d_j.append(j)
        # similarity
        s_i=[]
        s_j=[]
        for i in range(len(selected_itrait)):
            for j in range(i+1,len(selected_itrait)):
                s_i.append(selected_itrait[i])
                s_j.append(selected_itrait[j])
        
        return s_i,s_j,d_i,d_j
    
    def train(self,train_data,test_data,item_pool,lr=0.01,epoch=2):
        self.model=self.model.to(self.device)
        train_data=list(train_data)
        test_data=list(test_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        s_i,s_j,d_i,d_j = self.get_distance_data(train_data,item_pool)
        
                
        for epoch_i in range(epoch):
            # s_data=[]
            loss = []
            for data in tqdm(train_data,f'Epoch {epoch_i+1} '):
                utrait,itrait,label,_=data
                itrait = itrait.squeeze()
                # u_loss: torch.Tensor  = torch.tensor(0.).to(self.device)
                utrait:torch.Tensor = utrait.to(self.device)
                itrait: torch.Tensor = itrait.to(self.device)
                label: torch.Tensor = label.to(self.device)
                score = self.model(utrait,itrait).squeeze(-1)
                u_loss=((score-label)**2).sum()
                loss.append(u_loss.item())
                optimizer.zero_grad()
                u_loss.backward()
                optimizer.step()
                
            si:torch.Tensor = torch.tensor(s_i).to(self.device)
            sj:torch.Tensor = torch.tensor(s_j).to(self.device)
            di:torch.Tensor = torch.tensor(d_i).to(self.device)
            dj:torch.Tensor = torch.tensor(d_j).to(self.device)
            si = self.model.itn(si)
            sj = self.model.itn(sj)
            di = self.model.itn(di)
            dj = self.model.itn(dj)
            e_loss = self.b*((si-sj)**2).sum()/((di-dj)**2).sum()
            optimizer.zero_grad()
            e_loss.backward()
            optimizer.step()
            
            print('Loss: ',float(np.mean(loss)))
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
        self.model=self.model.to(self.device)
        k_nums=[1,5,10,15,20]
        recall = [[]for i in k_nums]
        for data in tqdm(valid_data,'testing'):
            utrait,_,__,k_info=data
            k_items,k_DCG = self.getkitems(utrait,item_pool)
            for i,k in enumerate(k_nums):
                i_kitems = set(k_items[:k]).intersection(set(k_info[:k]))
                recall[i].append(len(i_kitems)/k)
        for i,k in enumerate(k_nums):
            print(f'recall@{k}: ',np.mean(recall[i]))
    
    def getkitems(self, utrait,item_pool):
        with torch.no_grad():
            self.model.eval()
            utrait:torch.Tensor = utrait.to(self.device)
            itrait:torch.Tensor = torch.tensor(list(item_pool.values())).to(self.device)
            scores = self.model(utrait,itrait).squeeze(-1)
            tmp = list(zip(scores.tolist(),item_pool.keys()))
            tmp_sorted = sorted(tmp, reverse=True)
            self.model.train()
        return [int(i[1]) for i in tmp_sorted[:self.k]],[e[0]/np.log(i+2) for i,e in enumerate(tmp_sorted[:self.k])]
        

    
    
    
    
        