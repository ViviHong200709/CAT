import numpy as np
import torch
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class CSStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Cognitive Structure Select Strategy'

    def adaptest_select(self, model: AbstractModel, sid, adaptest_data: AdapTestDataset, cognitive_structure, theta):
        # 已知 当前area
        # area中已被选择的知识点（）
        ''' 
            初始化：
            topic_arr=[]
            cognitive structure所有知识点为unselected

            if 达到该topic最大长度或者最大覆盖率
                切换topic
            if concept candidate为空
                选择度数最大的k个知识点（未被覆盖的），加入到concept candidate中
            计算在concept candidate的信息量，返回信息量最大的题目


            收到作答记录
            concept candidate=[]
            若回答正确，则前驱节点都被覆盖，将k跳后继节点（未被覆盖的）加入concept candidate.
            否则后继节点都被覆盖，将k跳前驱点（未被覆盖的）加入concept candidate.

        '''
        current_topic = cognitive_structure.current_topic
        print('cover_rate:', current_topic['cover_num']/current_topic['node_num'] )
        # if current_topic['selected_num'] >= current_topic['max_length'] or current_topic['cover_num']/current_topic['node_num'] >= cognitive_structure.max_cover_rate:
        if current_topic['selected_num'] >= cognitive_structure.max_length or current_topic['cover_num']/current_topic['node_num'] >= cognitive_structure.max_cover_rate:
            print('next topic...')
            next_topic = cognitive_structure.next_topic()
            if not next_topic:
                # print('None')
                return []
            # print('regular')
        #     current_topic = cognitive_structure.current_topic
        # if not current_topic:
        #     return None
        
        return cognitive_structure.get_item_candidate(theta)
        
