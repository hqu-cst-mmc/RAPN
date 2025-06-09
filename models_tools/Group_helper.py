import torch
import pickle
import numpy as np
class Group(object):
    def __init__(self, args, Symmetrical=True, Max=None, Min=None):
        '''
            dataset : list of deltas (CoRe method) or list of scores (RT method)
            depth : depth of the tree
            Symmetrical: (bool) Whether the group is symmetrical about 0.
                        if symmetrical, dataset only contains th delta bigger than zero.
            Max : maximum score or delta for a certain sports.
        '''
        self.train_seg_label_path = args.train_seg_label_path
        self.train_seg_dict = self.read_pickle(self.train_seg_label_path)
        self.diff_dict = {}
        self.preprocess()
        self.dataset = []
        self.get_delta()
        self.length = len(self.dataset)
        self.np_dataset = np.array(self.dataset[int((7/8)*(self.length - 1)):])
        # self.num_leaf = 16
        self.num_leaf = 2 ** (args.RT_depth - 1)
        self.symmetrical = Symmetrical
        self.max = Max  # Max = 30
        self.min = Min  # Min = 0
        self.Group = [[] for _ in range(self.num_leaf)]
        self.build()

    def build(self):
        '''
            separate region of each leaf
        '''
        if self.symmetrical:
            # delta in dataset is the part bigger than zero.
            for i in range(self.num_leaf // 2):
                # bulid positive half first
                Region_left = self.dataset[int((i / (self.num_leaf // 2)) * (self.length - 1))]
                if Region_left == 0.0:
                    Region_left = int(Region_left)
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int(((i + 1) / (self.num_leaf // 2)) * (self.length - 1))]
                if Region_right == 0.0:
                    Region_right = int(Region_right)
                if i == self.num_leaf // 2 - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                self.Group[self.num_leaf // 2 + i] = [Region_left, Region_right]
            for i in range(self.num_leaf // 2):
                self.Group[i] = [-i for i in self.Group[self.num_leaf - 1 - i]]
            for group in self.Group:
                group.sort()
            print(1)
        else:
            for i in range(self.num_leaf):
                Region_left = self.dataset[int((i / self.num_leaf) * (self.length - 1))]
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int(((i + 1) / self.num_leaf) * (self.length - 1))]
                if i == self.num_leaf - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                self.Group[i] = [Region_left, Region_right]

    def produce_label(self, scores):
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().reshape(-1, )
        glabel = []  # glabel[]：表示在16个叶子区间中哪个区间是相对分数落入的区间，1表示落入，0表示未落入
        rlabel = []  # rlabel[]：表示在16个叶子区间中相对分数落入区间的sigma，相对分数未在区间内则是-1
        for i in range(self.num_leaf):
            # if in one leaf : left == right
            # we should treat this leaf differently
            leaf_cls = []
            laef_reg = []
            for score in scores:
                # 相对分数在右半区叶子
                if score >= 0 and (score < self.Group[i][1] and score >= self.Group[i][0]):
                    leaf_cls.append(1)
                # 相对分数在左半区叶子
                elif score < 0 and (score <= self.Group[i][1] and score > self.Group[i][0]):
                    leaf_cls.append(1)
                else:
                    leaf_cls.append(0)
                # 相对分数有落入一个区间中
                if leaf_cls[-1] == 1:
                    if self.Group[i][1] == self.Group[i][0]:
                        rposition = score - self.Group[i][0]
                    else:
                        #
                        rposition = (score - self.Group[i][0]) / (self.Group[i][1] - self.Group[i][0])
                else:
                    rposition = -1
                laef_reg.append(rposition)
            glabel.append(leaf_cls)
            rlabel.append(laef_reg)
        glabel = torch.tensor(glabel).cuda()
        rlabel = torch.tensor(rlabel).cuda()
        return glabel, rlabel

    def inference(self, probs, deltas):
        '''
            probs: bs * leaf
            delta: bs * leaf
        '''
        predictions = []
        for n in range(probs.shape[0]):
            prob = probs[n]
            delta = deltas[n]
            leaf_id = prob.argmax()
            if self.Group[leaf_id][0] == self.Group[leaf_id][1]:
                prediction = self.Group[leaf_id][0] + delta[leaf_id]
            else:
                prediction = self.Group[leaf_id][0] + (self.Group[leaf_id][1] - self.Group[leaf_id][0]) * delta[leaf_id]
            predictions.append(prediction)
        return torch.tensor(predictions).reshape(-1, 1)

    def get_Group(self):
        return self.Group

    def number_leaf(self):
        return self.num_leaf

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def preprocess(self):
        for item in list(self.train_seg_dict.keys()):
            diff = self.train_seg_dict[item]['difficulty']
            if self.diff_dict.get(diff) is None:
                self.diff_dict[diff] = []
            self.diff_dict[diff].append(item)
        return self.diff_dict

    def get_delta(self):
        # diff_dict-3.7:[0,1,2,3,264,265,267,268....]
        for diff in sorted(list(self.diff_dict.keys())):
            items = self.diff_dict[diff]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    referee_score_i = sorted(list(self.train_seg_dict[items[i]]['judge_scores']))[2:5]
                    referee_score_j = sorted(list(self.train_seg_dict[items[j]]['judge_scores']))[2:5]
                    self.dataset.append(abs(referee_score_i[0] - referee_score_j[0]))
                    self.dataset.append(abs(referee_score_i[1] - referee_score_j[1]))
                    self.dataset.append(abs(referee_score_i[2] - referee_score_j[2]))
        self.dataset = sorted(self.dataset)