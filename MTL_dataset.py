import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random
import pickle
import glob
import cv2
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class MTLPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        # using Difficult Degree
        self.usingDD = args.usingDD
        # some flags
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
        self.train_seg_label_path = args.train_seg_label_path
        self.train_seg_split_path = args.seg_train_split
        # 训练运动员划分
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.train_seg_dict = self.read_pickle(self.train_seg_label_path)
        self.train_seg_split = self.read_pickle(self.train_seg_split_path)
        self.data_root = args.data_root
        self.seg_train_videos_path = args.seg_train_videos_path
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.frame_length
        # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}
        self.all_frames = args.all_frames
        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        if self.usingDD:
            self.preprocess()
            self.check()

        self.choose_list = self.split.copy()
        # 获取测试集
        if self.subset == 'test':
            self.dataset = self.split_test
        # 获取训练集
        else:
            # self.dataset = self.split
            self.dataset = self.train_seg_split
    def load_video(self, video_file_name, phase):
        if phase == 'train':
            # seg_videos
            image_list = sorted((glob.glob(
                os.path.join(self.seg_train_videos_path, str(video_file_name), '*.jpg'))))
            video = [Image.open(image_path) for image_path in image_list]
            if (len(video) < self.all_frames):
                print("Error:" + str('{:02d}_{:02d}'.format(video_file_name[0], video_file_name[1])))
        elif phase == 'test':
            image_list = sorted((glob.glob(
                os.path.join(self.data_root, str('{:02d}_{:02d}'.format(video_file_name[0], video_file_name[1])),
                             '*.jpg'))))
            frame_start_idx = 0
            video = [Image.open(image_path) for image_path in
                     image_list[frame_start_idx:frame_start_idx + self.length]]
            if (len(video) < 96):
                print("Error:" + str('{:02d}_{:02d}'.format(video_file_name[0], video_file_name[1])))
        return self.transforms(video)

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def preprocess(self):
        self.diff_dict = {}
        for item in list(self.train_seg_dict.keys()):
            # check
            completeness_1 = round((self.train_seg_dict.get(item)['final_score'] / self.train_seg_dict[item]['difficulty']),1)
            judges = self.train_seg_dict.get(item)['judge_scores']
            completeness_2 = sum(sorted(judges)[2:5])
            if completeness_1 != completeness_2:
                print("Error!!!!!!!")

            diff = self.train_seg_dict[item]['difficulty']
            if self.diff_dict.get(diff) is None:
                self.diff_dict[diff] = []
            self.diff_dict[diff].append(item)
        # if self.dive_number_choosing:
        #     # Dive Number
        #     for item in self.split:
        #         dive_number = self.label_dict.get(item)['dive_number']
        #         if self.dive_number_dict.get(dive_number) is None:
        #             self.dive_number_dict[dive_number] = []
        #         self.dive_number_dict[dive_number].append(item)
        #
        #     if self.subset == 'test':
        #         for item in self.split_test:
        #             dive_number = self.label_dict.get(item)['dive_number']
        #             if self.dive_number_dict_test.get(dive_number) is None:
        #                 self.dive_number_dict_test[dive_number] = []
        #             self.dive_number_dict_test[dive_number].append(item)
        # else:
        #     # DD
        #     for item in self.split:
        #         difficulty = self.label_dict.get(item)['difficulty']
        #         if self.difficulties_dict.get(difficulty) is None:
        #             self.difficulties_dict[difficulty] = []
        #         self.difficulties_dict[difficulty].append(item)
        #
        #     if self.subset == 'test':
        #         for item in self.split_test:
        #             difficulty = self.label_dict.get(item)['difficulty']
        #             if self.difficulties_dict_test.get(difficulty) is None:
        #                 self.difficulties_dict_test[difficulty] = []
        #             self.difficulties_dict_test[difficulty].append(item)

    def check(self):
        num = 0
        for diff in sorted(list(self.diff_dict.keys())):
            items = self.diff_dict[diff]
            for item in items:
                num += 1
                assert self.train_seg_dict[item]['difficulty'] == diff
        assert num == len(list(self.train_seg_dict))
        print('%d check done!' % (num))
        # if self.dive_number_choosing:
        #     # dive_number_dict
        #     for key in sorted(list(self.dive_number_dict.keys())):
        #         file_list = self.dive_number_dict[key]
        #         for item in file_list:
        #             assert self.label_dict[item]['dive_number'] == key
        #
        #     if self.subset == 'test':
        #         for key in sorted(list(self.dive_number_dict_test.keys())):
        #             file_list = self.dive_number_dict_test[key]
        #             for item in file_list:
        #                 assert self.label_dict[item]['dive_number'] == key
        # else:
        #     # difficulties_dict
        #     for key in sorted(list(self.difficulties_dict.keys())):
        #         file_list = self.difficulties_dict[key]
        #         for item in file_list:
        #             assert self.label_dict[item]['difficulty'] == key
        #
        #     if self.subset == 'test':
        #         for key in sorted(list(self.difficulties_dict_test.keys())):
        #             file_list = self.difficulties_dict_test[key]
        #             for item in file_list:
        #                 assert self.label_dict[item]['difficulty'] == key


    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        # 测试
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(sample_1, 'test')
            data['final_score'] = self.label_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.label_dict.get(sample_1).get('difficulty')
            data['completeness'] = (data['final_score'] / data['difficulty'])
            data['judge_scores'] = self.label_dict.get(sample_1).get('judge_scores')
            if self.usingDD:
                train_contrast_list = self.diff_dict[data['difficulty']].copy()
                random.shuffle(train_contrast_list)
                contrast_items = train_contrast_list[:self.voter_number]
            else:
                train_contrast_list = self.train_seg_split.copy()
                random.shuffle(train_contrast_list)
                contrast_items = train_contrast_list[:self.voter_number]
            exemplar_list = []
            for sample_2 in contrast_items:
                exemplar = {}
                exemplar['video'] = self.load_video(sample_2, 'train')
                exemplar['final_score'] = self.train_seg_dict.get(sample_2).get('final_score')
                exemplar['difficulty'] = self.train_seg_dict.get(sample_2).get('difficulty')
                exemplar['completeness'] = (exemplar['final_score'] / exemplar['difficulty'])
                exemplar['judge_scores'] = self.train_seg_dict.get(sample_2).get('judge_scores')
                exemplar_list.append(exemplar)
            return data, exemplar_list
        # 训练
        else:
            # train phase
            data['video'] = self.load_video(sample_1, 'train')
            data['final_score'] = self.train_seg_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.train_seg_dict.get(sample_1).get('difficulty')
            data['completeness'] = (data['final_score'] / data['difficulty'])
            data['judge_scores'] = self.train_seg_dict.get(sample_1).get('judge_scores')

            if self.usingDD:
                # 从难度系数列表中取
                contrast_list = self.diff_dict[data['difficulty']].copy()
            else :
                # 从训练集中取
                contrast_list = self.train_seg_split.copy()
            if len(contrast_list) > 4:
                abs_index = sample_1 % 4
                same_video = [(sample_1-abs_index+i) for i in range(4)]
                for item in same_video:
                    contrast_list.pop(contrast_list.index(item))
            idx = random.randint(0,len(contrast_list) - 1)
            # exemplar
            sample_2 =  contrast_list[idx]
            exemplar = {}
            exemplar['video'] = self.load_video(sample_2, 'train')
            exemplar['final_score'] = self.train_seg_dict.get(sample_2).get('final_score')
            exemplar['difficulty'] = self.train_seg_dict.get(sample_2).get('difficulty')
            exemplar['completeness'] = (exemplar['final_score'] / exemplar['difficulty'])
            exemplar['judge_scores'] = self.train_seg_dict.get(sample_2).get('judge_scores')
            return data, exemplar

    def __len__(self):
        return len(self.dataset)

