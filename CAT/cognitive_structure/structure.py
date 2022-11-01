import networkx as nx
from copy import deepcopy

from torch import Graph
import torch


class CogntiveStructure(object):
    def __init__(self, edges, topic_concept, concept_item, item_concept, item_diff, item_disc, max_length=10, max_depth=3, decay=0.9, max_cover_rate=0.45, k=5):
        self.max_depth = max_depth
        self.decay = decay
        print('decay:', decay)
        self.max_length = max_length
        self.max_cover_rate = max_cover_rate
        self.k = k

        self.topic_concept = topic_concept
        self.concept_topic = self.init_concept_topic(topic_concept)
        self.concept_item = concept_item
        self.item_concept = item_concept
        self.edges = edges
        self.item_diff = item_diff
        self.item_disc = item_disc

        self.topics = self.init_topics(self.topic_concept)
        self.init_graph(self.edges)
        self.init_max_length()

    def reset(self, items):
        # self.init_graph(self.edges)
        self.concept_candidate = []
        self.current_topic_idx = 0
        self.current_topic = self.topics[self.current_topic_idx]

        for topic in self.topics:
            G = topic['Graph']
            nodes = G.nodes()
            for node in nodes:
                attr = G.nodes[node]
                attr["covered"] = 0
                attr['missed'] = False
                attr['items'] = deepcopy(self.concept_item[str(node)])
            topic['selected_num'] = 0
            topic['cover_num'] = 0

        for item in items:
            concepts = self.item_concept[str(item)]
            for concept in concepts:
                for topic in self.topics:
                    nodes = topic['Graph'].nodes
                    if concept in nodes:
                        nodes[concept]['items'].remove(item)
                        if len(nodes[concept]['items']) == 0:
                            nodes[concept]['missed'] = True
                        break

        # self.init_concept_candidate()

    def init_max_length(self):
        total_node_num = 0
        for topic in self.topics:
            total_node_num += topic['node_num']
        for topic in self.topics:
            topic['max_length'] = int(
                topic['node_num']/total_node_num*1.1*self.max_length)

    def next_topic(self):
        if self.current_topic_idx+1 == len(self.topics):
            return False
        self.current_topic_idx += 1
        self.current_topic = self.topics[self.current_topic_idx]
        # self.init_concept_candidate()
        return True

    def init_topics(self, topic_concept):
        topic_len = [(len(val), key) for key, val in topic_concept.items()]
        c = sorted(topic_len, reverse=True)
        return [{'name': i[1], 'edges':[], 'Graph':None} for i in c]

    def init_concept_topic(self, topic_concept):
        res = {}
        for key, val in topic_concept.items():
            for i in val:
                res[i] = key
        return res

    def init_graph(self, edges):
        p_dict={}
        for e in edges:
            for topic in self.topics:
                if self.concept_topic[e[0]] == topic['name'] and self.concept_topic[e[1]] == topic['name']:
                    topic['edges'].append(e)
                    p_dict[e[0]]=e[2]['h']
                    p_dict[e[1]]=e[2]['t']
                    break
        
        for topic in self.topics:
            G = nx.DiGraph()
            G.add_edges_from(topic['edges'])
            nodes = G.nodes()
            tmp_degree = [G.degree(node) for node in nodes]
            max_degree = max(tmp_degree)
            min_degree = min(tmp_degree)
            tmp_degree = [(x-min_degree)/(max_degree-min_degree)
                          for x in tmp_degree]
            for i, node in enumerate(nodes):
                attr = G.nodes[node]
                attr['p'] = p_dict[node]
                attr["degree"] = tmp_degree[i]
                diff = [self.item_diff[str(item)]
                        for item in self.concept_item[str(node)]]
                if len(diff) != 0:
                    attr["diff"] = sum(diff)/len(diff)
                else:
                    attr["diff"] = None

                disc = [self.item_disc[str(item)]
                        for item in self.concept_item[str(node)]]
                if len(disc) != 0:
                    attr["disc"] = sum(disc)/len(disc)
                else:
                    attr["disc"] = None

            topic['Graph'] = G
            topic['node_num'] = G.number_of_nodes()

    def get_item_candidate(self, theta=None):
        # init concept candidate
        G = self.current_topic['Graph']
        # sorted by node degree
        # tmp = [(G.nodes[node]["degree"]- G.nodes[node]["covered"], node)
        tmp = [(G.nodes[node]["degree"], node)
               for node in G.nodes() if G.nodes[node]["missed"] == False and G.nodes[node]["covered"] != 1]
        tmp_sorted = sorted(tmp, reverse=True)

        # # sorted by diff-theta
        # tmp = [(abs(G.nodes[node]["diff"] + G.nodes[node]["disc"]*theta-0.5), node)
        #        for node in G.nodes() if G.nodes[node]["missed"] == False and G.nodes[node]["diff"] and G.nodes[node]["covered"] == 0]
        # tmp_sorted = sorted(tmp, reverse=False)

        self.concept_candidate = [i[1] for i in tmp_sorted[:self.k]]

        # init item candidate
        item_candidate = []
        for concept in self.concept_candidate:
            item_candidate.extend(self.concept_item[str(concept)])

        return item_candidate

    def update(self, qid, ans):
        self.current_topic['selected_num'] += 1
        concepts = self.item_concept[str(qid)]
        for concept in concepts:
            if ans ==1:
                self.cover_successors(concept, 1, ans)
            else:
                self.cover_predecessors(concept, 1, ans)
            # self.init_concept_candidate()
        # self.show_fig(qid)
        # pass

    def show_fig(self, qid):
        # junyi edges可视化
        import graphviz
        color_dict = ['#253d24', '#ff000042',
                      '#a61b29', '#57c3c2', '#fecc11', '#e8b49a']
        G = self.current_topic['Graph']
        g = graphviz.Digraph('G', filename=f'sim_test_{qid}')
        g.attr(rankdir='LR')
        for node in G.nodes():
            if G.nodes[node]['covered'] == 1:
                g.attr('node', style='filled', color=color_dict[1])
            else:
                g.attr('node', style='filled', color=color_dict[0])
            g.node(str(node))
        for edge in G.edges():
            g.edge(str(edge[0]), str(edge[1]))
        try:
            g.view()
        except:
            print(f'Graph {qid} generated')

    def cover_successors(self, concept, val, ans):
        for topic in self.topics:
            if concept not in topic['Graph'].nodes:
                continue
            G = topic['Graph']
            break
        nodes = G.nodes
        if nodes[concept]["covered"] >= val:
            return
        nodes[concept]["covered"] = val
        self.current_topic['cover_num'] += 1
        # if concept in self.concept_candidate:
        #     self.concept_candidate.remove(concept)
        # self.update_predecessors_degree(G, concept)

        successors = G.successors(concept)
        while True:
            try:
                s = next(successors)
                # if ans==1:
                #     p = G.edges[concept,s]['t11']
                # else:
                #     p = G.edges[concept,s]['t10']
                # if p!=None:
                    # self.cover_successors(s, val*p,ans)
                self.cover_successors(s, val, ans)
            except StopIteration:
                break

    def cover_predecessors(self, concept, val, ans):
        for topic in self.topics:
            if concept not in topic['Graph'].nodes:
                continue
            G = topic['Graph']
            break

        nodes = G.nodes
        if nodes[concept]["covered"] >=val:
            return
        nodes[concept]["covered"] = val
        self.current_topic['cover_num'] += 1
        
        # if concept in self.concept_candidate:
        #     self.concept_candidate.remove(concept)
        # self.update_successors_degree(G, concept)
        predecessors = G.predecessors(concept)
        while True:
            try:
                s = next(predecessors)
                # if ans==1:
                #     p = G.edges[s,concept]['h11']
                # else:
                #     p = G.edges[s,concept]['h10']
                # if p!=None:
                #     self.cover_predecessors(s, val*p, ans)
                self.cover_predecessors(s, val, ans)
            except StopIteration:
                break

    def update_successors_degree(self, G, concept):
        successors = G.successors(concept)
        while True:
            try:
                s = next(successors)
                G.nodes[s]['degree'] -= 1
            except StopIteration:
                break

    def update_predecessors_degree(self, G, concept):
        predecessors = G.predecessors(concept)
        while True:
            try:
                s = next(predecessors)
                G.nodes[s]['degree'] -= 1
            except StopIteration:
                break
