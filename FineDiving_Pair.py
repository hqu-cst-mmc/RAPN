import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image

class FineDiving_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform, local_trans):
        random.seed(args.seed)
        self.subset = subset
        self.event = args.event
        self.view = args.view
        self.labelType = args.labelType
        # global transforms
        self.transforms = transform
        # local transforms
        self.local_trans = local_trans
        self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number

        # file path
        self.data_root = os.path.join(args.data_root, f'{self.event}-{self.view}')
        self.data_anno = self.read_pickle(os.path.join(args.label_path, f'{self.event}-{self.view}', 'Annotations', 'annotations_pkl', f'annotation-{self.event}.pkl'))
        self.train_anno = self.read_pickle(os.path.join(args.label_path, f'{self.event}-{self.view}', 'Annotations', 'annotations_pkl', f'annotation-{self.event}_64-4.pkl'))
        self.train_split = os.path.join(args.label_path, f'{self.event}-{self.view}', 'Annotations', 'split_pkl', f'train_split-{self.event}_64-4.pkl')
        self.test_split = os.path.join(args.label_path, f'{self.event}-{self.view}', 'Annotations', 'split_pkl', f'test_split-{self.event}(0.7-0.3).pkl')
        with open(self.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(self.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        # !!
        # self.subset = 'test'
        self.action_number_dict_test = {}
        # !!
        for item in self.train_dataset_list:
            dive_number = self.train_anno.get(item)['action_code']
            difficulty = self.train_anno.get(item)['dd']
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
            if self.difficulties_dict.get(difficulty) is None:
                self.difficulties_dict[difficulty] = []
            self.difficulties_dict[difficulty].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)['action_code']
                difficulty = self.data_anno.get(item)['dd']
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)
                if self.difficulties_dict_test.get(difficulty) is None:
                    self.difficulties_dict_test[difficulty] = []
                self.difficulties_dict_test[difficulty].append(item)
        print('Preprocessing -')
    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.train_anno[item]['action_code'] == key
            print("action_code check done!")
            for key in sorted(list(self.difficulties_dict.keys())):
                file_list = self.difficulties_dict[key]
                for item in file_list:
                    assert self.train_anno[item]['dd'] == key
            print("difficulty check done!")
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item]['action_code'] == key
            for key in sorted(list(self.difficulties_dict_test.keys())):
                file_list = self.difficulties_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item]['dd'] == key
        print('Check done!')
    def load_video(self, video_file_name, subset='train'):
        if subset == 'train':
            image_list = sorted((glob.glob(os.path.join(self.data_root, 'train', video_file_name, '*.jpg'))))
            video = [Image.open(image_path) for image_path in image_list]
            assert len(video) == 64

        elif subset == 'test':
            indices_1 = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38,
                         39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74,
                         75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95]
            image_list = sorted((glob.glob(os.path.join(self.data_root, 'test', video_file_name, '*.jpg'))))
            # 96-64
            new_image_list = []
            for i in indices_1:
                new_image_list.append(image_list[i])
            video = [Image.open(image_path) for image_path in new_image_list]
            assert len(video) == 64
        return self.transforms(video), self.local_trans(video)
        # return self.local_trans(video)


    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        # # 该训练集index 动作编号只有一个样本
        # index = 291
        # 测试集index
        # index = 17
        sample_1 = self.dataset[index]
        # 测试！！！！
        # sample_1 = ('07', 3)
        # sample_1 = ('10', 38)
        # #######################
        # 1- 选择目标视频
        # #######################
        data = {}
        data['video'], data['local_video'] = self.load_video(sample_1, self.subset)
        if self.subset == 'train':
            data['number'] = self.train_anno.get(sample_1)['action_code']
            data['final_score'] = self.train_anno.get(sample_1)['F_score']
            data['difficulty'] = self.train_anno.get(sample_1)['dd']
            data['completeness'] = round((data['final_score'] / data['difficulty']), 3)
            data['exe_scores'] = np.array(sorted(self.train_anno.get(sample_1)['E_score'])).astype(np.float32)
            data['sync_scores'] = np.array(sorted(self.train_anno.get(sample_1)['S_score'])).astype(np.float32)
            data['judge_scores'] = np.array(sorted(self.train_anno.get(sample_1)['E_score']) + sorted(self.train_anno.get(sample_1)['S_score'])).astype(np.float32)
            assert round(sum(data['judge_scores']) * 0.6, 3) == data['completeness']
        elif self.subset == 'test':
            data['number'] = self.data_anno.get(sample_1)['action_code']
            data['final_score'] = self.data_anno.get(sample_1)['F_score']
            data['difficulty'] = self.data_anno.get(sample_1)['dd']
            data['completeness'] = round((data['final_score'] / data['difficulty']), 3)
            data['exe_scores'] = np.array(sorted(self.data_anno.get(sample_1)['E_score'])).astype(np.float32)
            data['sync_scores'] = np.array(sorted(self.data_anno.get(sample_1)['S_score'])).astype(np.float32)
            data['judge_scores'] = np.array(sorted(self.data_anno.get(sample_1)['E_score']) + sorted(self.data_anno.get(sample_1)['S_score'])).astype(np.float32)
            assert round(sum(data['judge_scores']) * 0.6, 3) == data['completeness']
        # choose a exemplar
        # #######################
        # 2.1- 训练集选择对比视频
        # #######################
        if self.subset == 'train':
            # train phrase
            if self.labelType == 'DN':
                file_list = self.action_number_dict[self.train_anno[sample_1]['action_code']].copy()
            elif self.labelType == 'DD':
                file_list = self.difficulties_dict[self.data_anno[sample_1]['dd']].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'], target['local_video'] = self.load_video(sample_2, subset='train')
            target['number'] = self.train_anno.get(sample_2)['action_code']
            target['final_score'] = self.train_anno.get(sample_2)['F_score']
            target['difficulty'] = self.train_anno.get(sample_2)['dd']
            target['completeness'] = round((target['final_score'] / target['difficulty']), 3)
            target['exe_scores'] = np.array(sorted(self.train_anno.get(sample_2)['E_score'])).astype(np.float32)
            target['sync_scores'] = np.array(sorted(self.train_anno.get(sample_2)['S_score'])).astype(np.float32)
            target['judge_scores'] = np.array(sorted(self.train_anno.get(sample_2)['E_score']) + sorted(self.train_anno.get(sample_2)['S_score'])).astype(np.float32)
            assert round(sum(target['judge_scores']) * 0.6, 3) == target['completeness']

            return data, target
        # #######################
        # 2.2- 测试集选择对比视频
        # #######################
        else:
            # test phrase
            if self.action_number_choosing:
                # !!!!!!test
                # sample_1 = '24-V1-R3-4'
                # sample_1 = '07-V1-R1-4'
                # sample_1 = '10m-31-V1-R6-3'
                # !!!!!!test
                if self.labelType == 'DN':
                    train_file_list = self.action_number_dict[self.data_anno[sample_1]['action_code']].copy()
                elif self.labelType == 'DD':
                    train_file_list = self.difficulties_dict[self.data_anno[sample_1]['dd']].copy()
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
                action_len = len(choosen_sample_list)
                # 如果动作编号列表中少于选举数，则从难度系数列表中借
                if action_len < self.voter_number:
                    for _ in range(self.voter_number - action_len):
                        idx = random.randint(0, len(train_file_list) - 1)
                        choosen_sample_list.append(train_file_list[idx])
                    # borrow_list = self.difficulties_dict[self.data_anno[sample_1]['difficulty']].copy()
                    # for choosen in choosen_sample_list:
                    #     try:
                    #         if borrow_list.count(choosen) !=0 :
                    #             borrow_list.pop(borrow_list.index(choosen))
                    #     except:
                    #         print(1)
                    # random.shuffle(borrow_list)
                    # # 这里可能会有和choosen_sample_list中有重复的
                    # choosen_sample_list.extend(borrow_list[:self.voter_number - action_len])
                    # if len(choosen_sample_list) < self.voter_number:
                    #     choosen_sample_list.extend(choosen_sample_list[:self.voter_number - len(choosen_sample_list)])
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'], tmp['local_video'] = self.load_video(item, subset='train')
                tmp['number'] = self.train_anno.get(item)['action_code']
                tmp['final_score'] = self.train_anno.get(item)['F_score']
                tmp['difficulty'] = self.train_anno.get(item)['dd']
                tmp['completeness'] = round((tmp['final_score'] / tmp['difficulty']), 3)
                tmp['exe_scores'] = np.array(sorted(self.train_anno.get(item)['E_score'])).astype(np.float32)
                tmp['sync_scores'] = np.array(sorted(self.train_anno.get(item)['S_score'])).astype(np.float32)
                tmp['judge_scores'] = np.array(sorted(self.train_anno.get(item)['E_score']) + sorted(self.train_anno.get(item)['S_score'])).astype(np.float32)
                assert round(sum(tmp['judge_scores']) * 0.6, 3) == tmp['completeness']

                target_list.append(tmp)
            return data, target_list

    def __len__(self):
        return len(self.dataset)
