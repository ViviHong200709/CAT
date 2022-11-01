import numpy as np
import random

def search_metric_tree(candidates, q, T):
    min_candidates = min(candidates.values())
    if min_candidates >= np.dot(q, T.center)+T.radius*(((q**2).sum())**0.5):
        return 
    
    if T.is_leaf():
        for key,val in T.dict.items():
            if np.dot(val,q)<=min_candidates:
                continue
            for k1,v1 in candidates.items():
                if v1 == min_candidates:
                    del candidates[k1]
                    candidates[key] = np.dot(val,q)
                    min_candidates = min(candidates.values())
                    break
    else:
        I_l =np.dot(T.left.center,q)
        I_r =np.dot(T.right.center,q)
        if I_l<=I_r:
            search_metric_tree(candidates, q, T.right)
            search_metric_tree(candidates, q, T.left)
        else:
            search_metric_tree(candidates, q, T.left)
            search_metric_tree(candidates, q, T.right)
    
class BallTree(object):
    def __init__(self,item_pool,threshold=20):
        self.left=None
        self.right=None
        self.radius=None
        self.threshold=threshold
        self.make_ball(item_pool)
    
    def make_ball(self, item_pool):
        self.data = np.array(list(item_pool.values()))
        self.dict = item_pool
        if len(self.data)==0:
            return
        self.center = self.get_center()
        self.radius = self.get_radius()
        if len(self.data)<=self.threshold:
            return 
        w,b = self.make_metric_tree_split()
        items_left={}
        items_right={}
        for key,val in self.dict.items():
            if np.dot(val, w) + b <= 0:
                items_left[key]=val
            else:
                items_right[key]=val
        self.left=BallTree(items_left)
        self.right=BallTree(items_right)
    
    def is_leaf(self):
        return self.left==None   
     
    def get_center(self):
        return np.mean(self.data, axis=0)
    
    def get_l2_list(self,point):
        diff_matrix = self.data-np.expand_dims(point,0).repeat(len(self.data),axis=0)
        return (diff_matrix**2).sum(axis=1)

    def get_radius(self):
        return (self.get_l2_list(self.center).max())**0.5
        
    def make_metric_tree_split(self):   
        idx = random.randint(0,len(self.data)-1)
        x = self.data[idx] 
        A =  self.data[np.argmax(self.get_l2_list(x))]
        B =  self.data[np.argmax(self.get_l2_list(A))]
        w = B-A
        b = -1/2*((B**2).sum()-(A**2).sum())
        return (w,b)
    
    