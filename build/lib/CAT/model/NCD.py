from urllib.parse import scheme_chars
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset


class NCD(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(
            self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        self.layers = nn.Sequential(
            nn.Linear(self.prednet_input_len, self.prednet_input_len//2), nn.ReLU(
            ), nn.Dropout(p=0.2))
        # self.relu=nn.ReLU6()
        self.output_layer = nn.Linear(self.prednet_input_len//2, 1)
        torch.manual_seed(0)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        # input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        # output = torch.sigmoid(self.prednet_full3(input_x))

        # stu_emb = self.student_emb(stu_id)
        # k_difficulty = self.k_difficulty(exer_id)
        # e_discrimination = self.e_discrimination(exer_id)
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        # # input_x = self.drop_1(self.prednet_full1(input_x))
        # # input_x = self.drop_2(self.prednet_full2(input_x))
        # # output = torch.sigmoid(self.prednet_full3(input_x))

        # output = torch.sigmoid(self.output_layer(self.layers(input_x)))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        # stat_emb = self.student_emb(stu_id)
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # k_difficulty = self.k_difficulty(exer_id)
        # e_discrimination = self.e_discrimination(exer_id)
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCDModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset):
        self.model = NCD(data.num_students, data.num_questions, data.num_concepts,
                         self.config['prednet_len1'], self.config['prednet_len2'])

    def train(self, train_data: TrainDataset, test_data=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = []
            log_step = 20
            for student_ids, question_ids, concepts_emb, labels in tqdm(train_loader, f'Epoch {ep}'):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                # loss += bz_loss.data.float()
                loss.append(bz_loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (ep, float(np.mean(loss))))
            if test_data is not None:
                test_loader = data.DataLoader(
                    test_data, batch_size=batch_size, shuffle=True)
                self.eval(test_loader, device)

                # if cnt % log_step == 0:
                # logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))

    def eval(self, adaptest_data: AdapTestDataset, device):
        # data = adaptest_data.data
        self.model.to(device)

        with torch.no_grad():
            self.model.eval()
            y_pred = []
            y_true = []
            y_label = []
            for student_ids, question_ids, concepts_emb, labels in tqdm(adaptest_data, "evaluating"):
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_emb = torch.Tensor(concepts_emb).to(device)
                pred: torch.Tensor = self.model(
                    student_ids, question_ids, concepts_emb).view(-1)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(labels.tolist())
                y_label.extend([0 if p < 0.5 else 1 for p in pred])
            self.model.train()

        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_label)
        auc = roc_auc_score(y_true, y_pred)
        print(classification_report(y_true, y_label, digits=4))
        print('auc:', auc)

        return {
            'acc': acc,
            'auc': auc,
        }

    def fill(self, sid, qids, adaptest_data):
        device = self.config['device']
        self.model.to(device)
        res = []

        for qid in qids:
            concepts_emb = [0.] * adaptest_data.num_concepts
            for concept in adaptest_data.concept_map[str(qid)]:
                concepts_emb[concept] = 1.0
            sid_t = torch.LongTensor([sid]).to(device)
            qid_t = torch.LongTensor([qid]).to(device)
            concepts_emb = torch.Tensor(concepts_emb).to(device)
            pred_t = self.model(sid_t, qid_t, concepts_emb).view(-1)
            if pred_t.tolist()[0] < 0.5:
                pred = 0
            else:
                pred = 1
            res.append([sid, qid, pred])
        return res

    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)

    # def _loss_function1(self, pred, real):
    #     pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
    #     output = torch.cat((pred_0, pred), 1)
    #     criteria = nn.NLLLoss()
    #     return criteria(torch.log(output), real)

    def adaptest_save(self, path):
        """
        Save the model. Do not save the parameters for students.
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items()
                      if 'student' not in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, sid, qid, adaptest_data: AdapTestDataset, update_lr=None, optimizer=None, scheduler=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)

        label = adaptest_data.data[sid][qid]
        # print('label:', label)
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in adaptest_data.concept_map[qid]:
            concepts_emb[concept] = 1.0
        sid = torch.LongTensor([sid]).to(device)
        qid = torch.LongTensor([qid]).to(device)
        label = torch.LongTensor([int(label)]).to(device)
        concepts_emb = torch.Tensor(concepts_emb).to(device)
        pred: torch.Tensor = self.model(sid, qid, concepts_emb)
        bz_loss = self._loss_function(pred, label)
        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            # print('lr:', scheduler.get_last_lr())
        self.model.apply_clipper()
        # print('concept:', concepts_emb)
        # print('difficulty:', self.model.get_exer_params(qid)[0]*concepts_emb)
        # print('disc:', self.model.get_exer_params(qid)[1]*concepts_emb)
        # print('stu_emb: ', self.model.get_knowledge_status(sid))

    def evaluate(self, sid, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        with torch.no_grad():
            self.model.eval()
            # for sid in data:
            student_ids = [sid] * len(data[sid])
            question_ids = list(data[sid].keys())
            concepts_embs = []
            for qid in question_ids:
                concepts = concept_map[qid]
                concepts_emb = [0.] * adaptest_data.num_concepts
                for concept in concepts:
                    concepts_emb[concept] = 1.0
                concepts_embs.append(concepts_emb)
            real = [data[sid][qid] for qid in question_ids]
            student_ids = torch.LongTensor(student_ids).to(device)
            question_ids = torch.LongTensor(question_ids).to(device)
            concepts_embs = torch.Tensor(concepts_embs).to(device)
            # print(student_ids,question_ids)
            output = self.model(student_ids, question_ids,
                                concepts_embs).view(-1)
            pred = output.tolist()
            # print('test:', self.model.student_emb.weight[sid].sum())
            self.model.train()

        # coverages = []
        # for sid in data:
        #     all_concepts = set()
        #     tested_concepts = set()
        #     for qid in data[sid]:
        #         all_concepts.update(set(concept_map[qid]))
        #     for qid in adaptest_data.tested[sid]:
        #         tested_concepts.update(set(concept_map[qid]))
        #     coverage = len(tested_concepts) / len(all_concepts)
        #     coverages.append(coverage)
        # cov = sum(coverages) / len(coverages)

        real = np.array(real)
        pred = np.array(pred)
        pred_label = [0 if p < 0.5 else 1 for p in pred]
        acc = accuracy_score(real, pred_label)
        auc = roc_auc_score(real, pred)
        # print('acc:', acc,'auc:', auc)
        # print('\n')

        return {
            'acc': acc,
            'auc': auc,
            'cov': 0,
        }

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids,
                                    concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
