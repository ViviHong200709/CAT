# from functools import cache
import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MFIStrategy(AbstractStrategy):
    """
    Maximum Fisher Information Strategy
    D-opt Strategy when using MIRT(num_dim != 1)
    """

    def __init__(self):
        super().__init__()
        self.I = None
        # self.cache_fisher={}
        # self.pred_all=None

    @property
    def name(self):
        return 'Maximum Fisher Information Strategy'

    def adaptest_select(self, model: AbstractModel, sid, adaptest_data: AdapTestDataset, item_candidates=None,itempool=None):
        assert hasattr(model, 'get_fisher'), \
            'the models must implement get_fisher method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        if self.I is None:
            self.I = []
            for _ in range(adaptest_data.num_students):
                self.I.append(np.zeros((model.model.num_dim, model.model.num_dim)))
            # self.I = [np.zeros((model.model.num_dim, model.model.num_dim))] * adaptest_data.num_students    
        # selection = {}
        # n = len(adaptest_data.tested[0])

        if item_candidates is None:
            untested_questions = np.array(list(adaptest_data.untested[sid]))
        else:
            available = adaptest_data.untested[sid].intersection(set(item_candidates))
            untested_questions = np.array(list(available))
        # untested_questions = item_candidates
        untested_dets = []
        untested_fisher = []
        # if sid not in self.cache_fisher:
        #     self.cache_fisher[sid]={}
        for qid in untested_questions:
            # if qid not in self.cache_fisher[sid]:
            fisher_info = model.get_fisher(sid, qid, pred_all)
            if itempool !=None:
                avg_embs = np.array(list(itempool.values())).mean(axis=0)
                p=0.001
                tested_qid = adaptest_data.tested[sid]
                # tested_qid
                if len(tested_qid)==0:
                    avg_tested_emb=np.array([0,0])
                else:
                    avg_tested_emb = np.array([itempool[str(qid)] for qid in tested_qid]).mean(axis=0)
                item_emb=itempool[str(qid)]
                diff = ((item_emb-avg_tested_emb)**2).sum()
                sim = ((item_emb-avg_embs)**2).sum()
                fisher_info = fisher_info + p*diff/sim
                # self.cache_fisher[sid][qid]=fisher_info
            # else:
                # fisher_info = self.cache_fisher[sid][qid]
            untested_fisher.append(fisher_info)
            untested_dets.append(np.linalg.det(self.I[sid] + fisher_info))
        j = np.argmax(untested_dets)
        # selection[sid] = untested_questions[j]
        # print(untested_fisher[j])
        self.I[sid] += untested_fisher[j]
        return untested_questions[j]

        # for sid in range(adaptest_data.num_students):
            # 对于某个student
            # untested_questions = np.array(list(adaptest_data.untested[sid]))
            # untested_dets = []
            # untested_fisher = []
            # for qid in untested_questions:
            #     fisher_info = model.get_fisher(sid, qid, pred_all)
            #     untested_fisher.append(fisher_info)
            #     untested_dets.append(np.linalg.det(self.I[sid] + fisher_info))
            # j = np.argmax(untested_dets)
            # selection[sid] = untested_questions[j]
            # self.I[sid] += untested_fisher[j]
        return selection
    
class DoptStrategy(MFIStrategy):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self):
        return 'D-Optimality Strategy'