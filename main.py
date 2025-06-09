import argparse
import os
import time
import shutil

import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from dataset import TSNDataSet
from MTL_dataset import MTLPair_Dataset
from FineDiving_Pair import FineDiving_Pair_Dataset
# from models import TSN
from transforms import *
# from opts import parser
from utils import parser
from scipy import stats
from torchvideotransforms import video_transforms, volume_transforms
from utils import misc
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))
best_prec1 = 0
from models_tools import vit_encoder
from models_tools import vit_decoder
from models_tools import contrastive_fusion
from models_tools import MLP
from models_tools import MLP_timesformer
from models_tools import judge_model
from models_tools import ExeVariaNetwork
from models_tools import Group_helper
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from models_tools.pspnet import PSPNet
from timesformer.models.vit import TimeSformer
from utils.save_model import save_checkpoint, get_logger
from utils.resume_train import resume_train


def video_segment(args,video):
    # bs,1,96,3,299,299
    batch_size = video.size()[0]
    video_pack = torch.zeros(batch_size, 1, args.segments_num*args.segment_frame_num, 3, args.frame_HW,args.frame_HW).cuda()
    video = video.permute(0, 2, 1, 3, 4).unsqueeze(1)
    # 总共划分出3个阶段,3个子视频在3个阶段中分别取16帧
    # 分段采样
    if args.frame_select_strategy == "segment_sampling":
        start_idx = [[23, 39, 55], [0, 32, 64], [16, 48, 80]]
        for bs in range(0, batch_size):
            for idx in range(0, args.sub_video_num):
                video_seg = torch.cat([video[bs, :, i: i + args.segment_frame_num] for i in start_idx[idx]])
                video_pack[bs, idx] = video_seg
    # 奇偶采样
    elif args.frame_select_strategy == "odd_even_sampling":
        intersection_idx = [23, 39, 55]
        start_idx = [0 , 32, 64, 96]
        for bs in range(0, batch_size):
            if args.sub_video_num == 1:
                # 这是第三个视频取奇数帧
                # 1 :32:2->[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                # 33:64:2->[33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                # 65:96:2->[65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
                video_seg = video[bs,:,1:96:2]
                video_pack[bs, 0] = video_seg
            else:
                if args.all_frames == 64:
                    # b,2,64,3,224,224
                    video_pack = torch.zeros(batch_size, 1, args.all_frames, 3, args.frame_HW, args.frame_HW).cuda()
                    indices_1 = [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34,
                                 36, 37, 39,
                                 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72,
                                 73, 75, 76,
                                 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 95]
                    indices_2 = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33,
                                 35, 36, 38,
                                 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71,
                                 72, 74, 75,
                                 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95]
                    for bs in range(0, batch_size):
                        for idx, indice in enumerate(indices_1):
                            video_pack[bs, 0, idx] = video[bs, 0, indice]
                        # for idx_2, indice_2 in enumerate(indices_2):
                        #     video_pack[bs, 1, idx_2] = video[bs, indice_2]
                else:
                    # # 总共取3个视频
                    # # 这是第一个视频取三个阶段的交集
                    # video_seg = video[bs,:,23:23+48]
                    # video_pack[bs, 0] = video_seg
                    # # 这是第二个视频取偶数帧
                    # # 0 :32:2->[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
                    # # 32:64:2->[32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]
                    # # 64:96:2->[64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]
                    video_seg = video[bs, :,0:96:2]
                    # video_pack[bs, 1] = video_seg
                    video_pack[bs, 0] = video_seg
                    # # 这是第三个视频取奇数帧
                    # # 1 :32:2->[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                    # # 33:64:2->[33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                    # # 65:96:2->[65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
                    # video_seg = video[bs,:,1:96:2]
                    # video_pack[bs, 2] = video_seg
    return video_pack

def create_labels(judge_scores):
    judge_num = judge_scores.size(0)
    start_list = [0,6,7,7.5,8,8.5,10.5]
    score_list = []
    for i in range(0,len(start_list)-1):
        score_list.append([score for score in np.arange(start_list[i],start_list[i+1],0.5)])
    cls_labels = torch.zeros([judge_num,6],dtype=torch.int64).cuda()
    for j in range(0,judge_num):
        for i in range(0,len(score_list)):
            if judge_scores[j] in score_list[i]:
                cls_labels[j][i] = 1
                break
    return cls_labels,score_list
def create_indices(cls_prob):
    batch_size,judge_num,cls_num = cls_prob.size()
    predict_indices = torch.zeros([batch_size,judge_num],dtype=torch.int64).cuda()
    for bs in range(0,batch_size):
        for j in range(0,judge_num):
            predict_indices[bs][j] = cls_prob[bs][j].argmax(-1)
    return predict_indices

def compute_judge_score(reg,predict_indices,score_list):
    batch_size,judge_num = predict_indices.size()
    predict_judge_score = torch.zeros([batch_size,judge_num]).cuda()
    for bs in range(0,batch_size):
        for j in range(0,judge_num):
            # bs,j个裁判归属于那个类别
            index = predict_indices[bs][j]
            # 该类别的最小值
            start = score_list[index][0]
            # 该类别的最大值
            end = score_list[index][-1]
            score_range = end-start
            predict_judge_score[bs][j] = start + score_range * reg[bs][j][index]
    return predict_judge_score

def create_norm_judge_label(judge_scores,score_list):
    judge_num = judge_scores.size(0)
    norm_judge_label = torch.zeros([judge_num,6],requires_grad=True).cuda()
    for j in range(0,judge_num):
        judge_score = judge_scores[j]
        for i in range(0,len(score_list)):
            if judge_score in score_list[i]:
                if len(score_list[i]) == 1:
                    norm_judge_label[j][i] = 0
                else:
                    norm_judge_label[j][i] = (judge_score - score_list[i][0]) / (score_list[i][-1] - score_list[i][0])
            else:
                norm_judge_label[j][i] = -1
    return norm_judge_label

def select_judge(predict_scores,judge_scores):
    predict_scores = predict_scores.sort().values
    batch_size,video_num,judge_num = predict_scores.shape
    out_predict = torch.zeros(batch_size,video_num,4)
    out_judge = torch.zeros(batch_size,video_num,4)
    in_predict = torch.zeros(batch_size,video_num,3)
    in_judge = torch.zeros(batch_size,video_num,3)
    for bs in range(0,batch_size):
       for v in range(0,video_num):
            out_predict_scores = torch.cat((predict_scores[bs][v][0:2],predict_scores[bs][v][5:]))
            out_judge_scores = torch.cat((judge_scores[bs][v][0:2],judge_scores[bs][v][5:]))
            out_predict[bs][v] = out_predict_scores
            out_judge[bs][v] = out_judge_scores
            in_predict[bs][v] = predict_scores[bs][v][2:5]
            in_judge[bs][v] = judge_scores[bs][v][2:5]
    out_predict,out_judge,in_predict,in_judge

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_referee_score(group, reg, cls, scores_exemplar, batch_size, referee_num):
    predict_score = torch.zeros([batch_size, referee_num], dtype=torch.float32)
    for bs in range(batch_size):
        for j in range(referee_num):
            #               # 示例分数             # / -------------------------相对分数---------------------------- /
            judge_score = scores_exemplar[bs][j] + group[cls[bs][j]][0] + reg[bs][j] * (
                        group[cls[bs][j]][1] - group[cls[bs][j]][0])
            if judge_score < 0:
                judge_score = 0
            elif judge_score > 10:
                judge_score = 10
            predict_score[bs][j] = judge_score

    return predict_score

def main():
    global args, best_prec1

    args = parser.get_args()
    parser.setup(args)
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100

    # ------随机种子------
    random_seed = 3407
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)             # Numpy
    random.seed(random_seed)                # Python
    torch.manual_seed(random_seed)          # pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed + args.local_rank)
        torch.cuda.manual_seed_all(random_seed + args.local_rank)
        torch.backends.cudnn.deterministic = True
    # -------------------

    if args.backbone == "TSN":
        model = TSN(
                    num_class=1024, num_videos=args.sub_video_num,
                    num_segments=args.segments_num,segment_frame_num=args.segment_frame_num,
                    modality='RGB',
                    base_model='BNInception',
                    consensus_type='avg', dropout=0.8, partial_bn=not False).apply(misc.fix_bn)
        crop_size = model.crop_size
        scale_size = model.scale_size
        input_mean = model.input_mean
        input_std = model.input_std
        policies = model.get_optim_policies()
        train_augmentation = model.get_augmentation()
        normalize = GroupNormalize(input_mean, input_std)
        resize_scale = (512,320)
        resize_size = 299
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    elif args.backbone == "Resnet":
        model = PSPNet()
        resize_scale = (455,256)
        resize_size = 224
    elif args.backbone == "timesformer":
        model = TimeSformer(
            img_size=224,   # 图像尺寸
            num_classes=768,
            num_frames=48,  # 三个片段每个16帧
            attention_type=args.attention_type,
            pretrained_model=args.pretrained_model)
        resize_scale = (455, 256)
        resize_size = 224
    # Transformer Encoder
    encoder = vit_encoder.encoder_fuser(dim=768,num_heads=12,num_layers=12,
                                    segment_frame_num=args.segment_frame_num,
                                        segments_num=args.segments_num,allframes=args.all_frames,MSA_num=3)
    # Contrastive Fusion
    fuser = contrastive_fusion.decoder_fuser(dim=768,num_heads=4,num_layers=2,query_num=3)
    # EVnet
    evnet = ExeVariaNetwork.EVnet(in_channel=768)
    # Transformer Decoder
    # decoder = vit_decoder.decoder_fuser(dim=768 * 2,num_heads=12,num_layers=12,query_num=3)
    decoder = vit_decoder.decoder_fuser(dim=768 * 2,num_heads=8,num_layers=4,exe_query_num=2,sync_query_num=3)
    # Degree
    if args.benchmark == "MTL":
        out_channel = 11
    elif args.benchmark == "FineDiving":
        # out_channel = 12
        # FineSync
        out_channel = 9
    rater_exe = judge_model.Judge(in_channel=768 * 2, out_channel=out_channel)
    rater_sync = judge_model.Judge(in_channel=768 * 2 + 2, out_channel=out_channel)
    if args.backbone =="timesformer":
        mlp = MLP_timesformer.MLP_tf(in_channel=768)
    else:
        mlp = MLP.MLP(in_channel=1024)


    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model = model.cuda()
    encoder = encoder.cuda()
    fuser = fuser.cuda()
    evnet = evnet.cuda()
    decoder = decoder.cuda()
    rater_exe = rater_exe.cuda()
    rater_sync = rater_sync.cuda()
    mlp = mlp.cuda()
    # multi-Gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        encoder = nn.DataParallel(encoder)
        fuser = nn.DataParallel(fuser)
        evnet = nn.DataParallel(evnet)
        decoder = nn.DataParallel(decoder)
        rater_exe = nn.DataParallel(rater_exe)
        rater_sync = nn.DataParallel(rater_sync)
        mlp = nn.DataParallel(mlp)

    cudnn.benchmark = True

    # MTL-AQA
    # group = Group_helper.Group(args, Symmetrical=True, Max=10., Min=0)
    # MTL-AQA judge_score group
    if args.benchmark == 'MTL':
        group = [[-10, -3], [-2.5, -2], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2, 2.5], [3, 10]]
        train_loader = torch.utils.data.DataLoader(
            MTLPair_Dataset(args,transform=video_transforms.Compose([
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.Resize(resize_scale),
                    video_transforms.RandomCrop(resize_size),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
                            subset='train'),
            batch_size=args.bs_train, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker
        )
        val_loader = torch.utils.data.DataLoader(
            MTLPair_Dataset(args,transform=video_transforms.Compose([
                    video_transforms.Resize(resize_scale),
                    video_transforms.CenterCrop(resize_size),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                            subset='test'),
            batch_size=args.bs_test, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.benchmark == 'FineDiving':
        # group = [[-10, -2.6], [-2.5, -1.7], [-1.6, -1.1], [-1.0, -0.6], [-0.5, -0.5], [-0.4, 0.0],
        #          [0.0, 0.4], [0.5, 0.5], [0.6, 1.0], [1.1, 1.6], [1.7, 2.5], [2.6, 10]]
        # FineSync
        resize_scale = (256, 455)
        resize_size = (144, 144)
        group = [[-10.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0],
                 [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 10.0]]
        # sync.3m-V1
        # group_exe = [[-10.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0],
        #          [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 10.0]]
        # group_sync = [[-10.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0],
        #          [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 10.0]]
        # sync.10m-V1
        group_exe = [[-10.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0],
                 [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 10.0]]
        group_sync = [[-10.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0],
                 [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 10.0]]

        train_loader = torch.utils.data.DataLoader(
            FineDiving_Pair_Dataset(args,
                #             全局 transform
                transform=video_transforms.Compose([
                # ============1-水平随机翻转==============
                video_transforms.RandomHorizontalFlip(),
                # ==========2-resize 455x256===========
                video_transforms.Resize((256, 455)),
                # ==========3-随机 crop 224x224=========
                video_transforms.RandomCrop((224, 224)),
                # ========4-转换成tensor并归一化==========
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
                #            局部 transform
                local_trans=video_transforms.Compose([
                # ===========1-水平随机翻转==============
                video_transforms.RandomHorizontalFlip(),
                # ==========2-resize 455x256===========
                video_transforms.Resize((256, 455)),
                # ==========3-中心 crop 300x256=========
                video_transforms.CenterCrop((256, 300)),
                # ==========4-随机 crop 144x144=========
                video_transforms.RandomCrop((144, 144)),
                # =========5-转换成tensor并归一化=========
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
                            subset='train'),
            batch_size=args.bs_train, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker
        )
        val_loader = torch.utils.data.DataLoader(
            FineDiving_Pair_Dataset(args,
                #             全局 transform
                transform=video_transforms.Compose([
                # ==========1-resize 455x256===========
                video_transforms.Resize((256, 455)),
                # ==========2-中心 crop 224x224=========
                video_transforms.CenterCrop((224, 224)),
                # =========3-转换成tensor并归一化=========
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
                #             局部 transform
                local_trans=video_transforms.Compose([
                # ==========1-resize 455x256===========
                video_transforms.Resize((256, 455)),
                # ==========2-中心 crop 224x224=========
                video_transforms.CenterCrop((256, 300)),
                # ==========3-中心 crop 144x144=========
                video_transforms.CenterCrop((144, 144)),
                # =========4-转换成tensor并归一化=========
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
                            subset='test'),
            batch_size=args.bs_test, shuffle=False, num_workers=8, pin_memory=True
        )

    mse = nn.MSELoss().cuda()
    nll = nn.NLLLoss().cuda()
    bce = nn.BCELoss().cuda()
    # criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    # criterion = nn.NLLLoss().cuda()

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.00001},
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': fuser.parameters(), 'lr': 0.00001},
        {'params': evnet.parameters(), 'lr': 0.00001},
        {'params': decoder.parameters(), 'lr': 0.00001},
        {'params': rater_exe.parameters(), 'lr': 0.00001},
        {'params': rater_sync.parameters(), 'lr': 0.00001},
        {'params': mlp.parameters(), 'lr': 0.00001}], lr=args.base_lr, weight_decay=args.weight_decay)

    # 共有4个存储的地方:
    # 1-tensorboard / 2-logger / 3-预测分数 / 4-checkpoint
    # ------1-tensorboard_log------
    tensorboard_path = f'tensorboard_log/FineSync_{args.event}(0.7-0.3)_{args.suffix}'
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    # writer = SummaryWriter(log_dir=tensorboard_path)
    # ------2-Constructing Logger------
    logger_path = f'logger/FineSync_{args.event}(0.7-0.3)_{args.suffix}'
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    base_logger = get_logger(os.path.join(logger_path, f'{args.event}_{args.labelType}_{args.suffix}.log'), 'FineSync-64x4')
    #
    # best
    start_epoch = 0
    epoch_best = 0
    epoch_best_L2 = 0
    epoch_best_RL2 = 0
    epoch_best_tau = 0
    rho_best = 0
    rho_best_L2 = 0
    rho_best_RL2 = 0
    L2_min = 1000
    L2_min_rho = 0
    L2_min_RL2 = 0
    RL2_min = 1000
    RL2_min_rho = 0
    RL2_min_L2 = 0
    # kendalltau
    tau_best = 0

    # 复现！
    if args.resume:
        start_epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min = resume_train(
            model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, args)
        true_scores = []
        pred_scores = []
        test_true_scores = []
        test_pred_scores = []
        test_raw_preds = []
        test_complement = []
        diffs = []

        Acc_list_exe, Acc_list_sync = validate(val_loader, model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, mse, nll, group, test_true_scores, test_pred_scores, test_raw_preds, test_complement, diffs, optimizer, start_epoch, group_exe, group_sync)

        test_pred_scores = np.array(test_pred_scores)
        test_true_scores = np.array(test_true_scores)

        rho, p = stats.spearmanr(test_pred_scores, test_true_scores)
        tau, _ = stats.kendalltau(test_pred_scores, test_true_scores)
        L2 = np.power(test_pred_scores - test_true_scores, 2).sum() / test_true_scores.shape[0]
        RL2 = np.power((test_pred_scores - test_true_scores) / (test_true_scores.max() - test_true_scores.min()),
                       2).sum() / \
              test_true_scores.shape[0]
        print('[TEST] EPOCH: %d, correlation: %.6f, kendalltau: %.6f, RL2: %.6f, L2: %.6f' % (start_epoch - 1, rho, tau, RL2, L2))
        print('[TEST] EPOCH: %d, Exe-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (
        start_epoch - 1, Acc_list_exe[0] * 100, Acc_list_exe[1] * 100, Acc_list_exe[2] * 100))
        print('[TEST] EPOCH: %d, Syncc-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (
        start_epoch - 1, Acc_list_sync[0] * 100, Acc_list_sync[1] * 100, Acc_list_sync[2] * 100))
        exit(0)

    for epoch in range(start_epoch, args.max_epoch):
        print(f'\nStart Training - {epoch}')
        base_logger.info(f'\nStart Training - {epoch}')
        print(time.ctime(time.time()))
        base_logger.info(time.ctime(time.time()))
        # 初始化变量
        # train
        true_scores = []
        pred_scores = []
        train_raw_preds = []
        train_complement = []
        # test
        test_true_scores = []
        test_pred_scores = []
        test_raw_preds = []
        test_complement = []
        diffs = []
        # !!
        # Acc_list_exe, Acc_list_sync = validate(val_loader, model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, mse, nll, group, test_true_scores, test_pred_scores, test_raw_preds, test_complement, diffs, optimizer, start_epoch, group_exe, group_sync)

        # # ----------------train----------------
        batch_loss, batch_loss_alpha, batch_loss_beta = train(train_loader, model, encoder, fuser, evnet, decoder,
                           rater_exe, rater_sync, mlp, mse, nll, bce, group, true_scores, pred_scores,
                           train_raw_preds, train_complement, diffs, optimizer, epoch, group_exe, group_sync)
        # 评估训练集
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        train_raw_preds = np.array(train_raw_preds)
        train_complement = np.array(train_complement)
        # 评估指标
        rho, p = stats.spearmanr(pred_scores, true_scores)
        tau, _ = stats.kendalltau(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        print('[TRAIN] EPOCH: %d, correlation: %.4f, kendalltau: %.4f, batch_loss: %.4f, L2: %.4f, RL2: %.4f, loss_a: %.4f, loss_b: %.4f' % (
        epoch, rho, tau, batch_loss, L2, RL2, batch_loss_alpha, batch_loss_beta))
        base_logger.info('[TRAIN] EPOCH: %d, correlation: %.4f, kendalltau: %.4f, batch_loss: %.4f, L2: %.4f, RL2: %.4f, loss_a: %.4f, loss_b: %.4f' % (
        epoch, rho, tau, batch_loss, L2, RL2, batch_loss_alpha, batch_loss_beta))

        # # 上传到前端
        # writer.add_scalar(tag="train_Src",
        #                   scalar_value=rho,
        #                   global_step=epoch)
        # writer.add_scalar(tag="train_Loss",
        #                   scalar_value=batch_loss,
        #                   global_step=epoch)


        # ------3-Saving Checkpoint------
        save_checkpoint(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, epoch, 'last', epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args)
        # ------3-Saving Checkpoint------

        # --------------test----------------
        # if (epoch < 100 and epoch % 3 == 0) or epoch >= 100:
        # if epoch < 100:
        if (epoch >= 60) or (epoch > 0 and epoch < 60 and (epoch % 10 == 0)):
        # if False:
        # if epoch >= 100 or (epoch < 100 and (epoch % 30) == 0):
        # if epoch >= 150:

            print(time.ctime(time.time()))
            base_logger.info(time.ctime(time.time()))

            Acc_list_exe, Acc_list_sync = validate(val_loader, model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, mse, nll, group, test_true_scores, test_pred_scores, test_raw_preds, test_complement, diffs, optimizer, start_epoch, group_exe, group_sync)

            # 评估训练集
            test_pred_scores = np.array(test_pred_scores)
            test_true_scores = np.array(test_true_scores)
            test_raw_preds = np.array(test_raw_preds)
            test_complement = np.array(test_complement)

            # 是否为最佳轮
            flag = False
            rho, p = stats.spearmanr(test_pred_scores, test_true_scores)
            # kendalltau
            tau, _ = stats.kendalltau(test_pred_scores, test_true_scores)
            L2 = np.power(test_pred_scores - test_true_scores, 2).sum() / test_true_scores.shape[0]
            RL2 = np.power((test_pred_scores - test_true_scores) / (test_true_scores.max() - test_true_scores.min()),
                           2).sum() / \
                  test_true_scores.shape[0]

            # 存储true和pred
            # ------4-Saving Prediction------
            predict_path = f'predict_result/FineSync_{args.event}(0.7-0.3)_{args.suffix}/{args.event}_{args.labelType}'
            if not os.path.exists(predict_path):
                os.makedirs(predict_path)
            save_true_path = os.path.join(predict_path, 'true_Last.npy')
            save_pred_path = os.path.join(predict_path, 'pred_Last.npy')
            np.save(save_true_path, test_true_scores)
            np.save(save_pred_path, test_pred_scores)

            base_logger.info('pred-completeness  ' + str(np.round(test_raw_preds, 1).astype(str)).replace('\n', ' '))
            base_logger.info('true-completeness  ' + str(np.round(test_complement, 1).astype(str)).replace('\n', ' '))
            if L2_min > L2:
                L2_min = L2
                L2_min_rho = rho
                L2_min_RL2 = RL2
                epoch_best_L2 = epoch
            if RL2_min > RL2:
                RL2_min = RL2
                RL2_min_rho = rho
                RL2_min_L2 = L2
                epoch_best_RL2 = epoch
                # 存储true和pred
                save_true_path = os.path.join(predict_path, 'true_RL2.npy')
                save_pred_path = os.path.join(predict_path, 'pred_RL2.npy')
                np.save(save_true_path, test_true_scores)
                np.save(save_pred_path, test_pred_scores)
                # 保存模型
                save_checkpoint(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, epoch, 'best_RL2', epoch_best,
                                epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args)

            if rho > rho_best:
                flag = True
                rho_best = rho
                rho_best_L2 = L2
                rho_best_RL2 = RL2
                epoch_best = epoch
                # 存储true和pred
                save_true_path = os.path.join(predict_path, 'true_SRC.npy')
                save_pred_path = os.path.join(predict_path, 'pred_SRC.npy')
                np.save(save_true_path, test_true_scores)
                np.save(save_pred_path, test_pred_scores)
                # 保存模型
                save_checkpoint(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, epoch, 'best_SRC', epoch_best,
                                epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args)

                print('-----New best found!-----')
                base_logger.info('-----New best found!-----')

            # kendalltau
            if tau > tau_best:
                tau_best = tau
                epoch_best_tau = epoch
                # 存储true和pred
                save_true_path = os.path.join(predict_path, 'true_Tau.npy')
                save_pred_path = os.path.join(predict_path, 'pred_Tau.npy')
                np.save(save_true_path, test_true_scores)
                np.save(save_pred_path, test_pred_scores)
                # 保存模型
                save_checkpoint(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, epoch,
                                'best_Tau', epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args)

                print('-----New best found!-----')
                base_logger.info('-----New best found!-----')

            print('[TEST] EPOCH: %d, correlation: %.6f, kendalltau: %.6f, RL2: %.6f, L2: %.6f' % (epoch, rho, tau, RL2, L2))
            print('[TEST] EPOCH: %d, Exe-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (epoch, Acc_list_exe[0] * 100, Acc_list_exe[1] * 100, Acc_list_exe[2] * 100))
            print('[TEST] EPOCH: %d, Syncc-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (epoch, Acc_list_sync[0] * 100, Acc_list_sync[1] * 100, Acc_list_sync[2] * 100))

            base_logger.info('[TEST] EPOCH: %d, correlation: %.6f, kendalltau: %.6f, RL2: %.6f, L2: %.6f' % (epoch, rho, tau, RL2, L2))
            base_logger.info('[TEST] EPOCH: %d, Exe-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (epoch, Acc_list_exe[0] * 100, Acc_list_exe[1] * 100, Acc_list_exe[2] * 100))
            base_logger.info('[TEST] EPOCH: %d, Syncc-Cls k1: %.2f%%, k2: %.2f%%, k3: %.2f%%' % (epoch, Acc_list_sync[0] * 100, Acc_list_sync[1] * 100, Acc_list_sync[2] * 100))
            base_logger.info('[TEST] EPOCH: %d, best correlation: %.6f, RL2: %.6f, L2: %.6f' % (
                epoch_best, rho_best, rho_best_RL2, rho_best_L2))
            base_logger.info('[TEST] EPOCH: %d, best kendalltau: %.6f, RL2: %.6f, L2: %.6f' % (
                epoch_best_tau, tau_best, rho_best_RL2, rho_best_L2))
            base_logger.info('[TEST] EPOCH: %d, best RL2: %.6f, correlation: %.6f, L2: %.6f' % (
                epoch_best_RL2, RL2_min, RL2_min_rho, RL2_min_L2))
            base_logger.info('[TEST] EPOCH: %d, best L2: %.6f, correlation: %.6f, RL2: %.6f' % (
                epoch_best_L2, L2_min, L2_min_rho, L2_min_RL2))
            # save_checkpoint(model, encoder, fuser, decoder, rater, mlp, optimizer, 666, epoch_best, epoch_best_RL2,
            #                 epoch_best_L2, rho_best, L2_min, RL2_min, args)


            # # 上传到前端
            # writer.add_scalar(tag="test_Src",
            #                   scalar_value=rho,
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="test_MSE",
            #                   scalar_value=L2,
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="test_RL2",
            #                   scalar_value=RL2,
            #                   global_step=epoch + 1)


            # writer.add_scalar(tag="test_Loss",
            #                   scalar_value=batch_loss,
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="cls_Loss",
            #                   scalar_value=loss_cls,
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="reg_Loss",
            #                   scalar_value=loss_reg,
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="k1",
            #                   scalar_value=Acc_list[0],
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="k2",
            #                   scalar_value=Acc_list[1],
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="k3",
            #                   scalar_value=Acc_list[2],
            #                   global_step=epoch + 1)
            # writer.add_scalar(tag="mae",
            #                   scalar_value=mae,
            #                   global_step=epoch + 1)



def get_contrastive_score(judge_scores,judge_scores_exemplar):
    contrastiveScore_v1_v2 = torch.zeros([judge_scores.size(0), 5])
    contrastiveScore_v2_v1 = torch.zeros([judge_scores.size(0), 5])
    for i in range(judge_scores.size(0)):
        contrastiveScore_v1_v2[i] = judge_scores[i] - judge_scores_exemplar[i]
        contrastiveScore_v2_v1[i] = judge_scores_exemplar[i] - judge_scores[i]
    return contrastiveScore_v1_v2 , contrastiveScore_v2_v1

def get_contrastive_label(group, contrastiveScore_v1_v2, contrastiveScore_v2_v1):
    clsLabel_v1_v2 = torch.zeros(contrastiveScore_v1_v2.size(0),dtype=torch.int64).cuda()
    normLabel_v1_v2 = torch.zeros(contrastiveScore_v1_v2.size(0),dtype=torch.float64,requires_grad=True).cuda()
    clsLabel_v2_v1 = torch.zeros(contrastiveScore_v2_v1.size(0),dtype=torch.int64).cuda()
    normLabel_v2_v1 = torch.zeros(contrastiveScore_v2_v1.size(0),dtype=torch.float64,requires_grad=True).cuda()
    for score_idx in range(contrastiveScore_v1_v2.size(0)):
        for group_idx,part in enumerate(group):
            if contrastiveScore_v1_v2[score_idx] >= part[0] and contrastiveScore_v1_v2[score_idx] <= part[1]:
                clsLabel_v1_v2[score_idx] = group_idx
                if part[0] == part[1]:
                    normLabel_v1_v2[score_idx] = 0
                else:
                    normLabel_v1_v2[score_idx] = (contrastiveScore_v1_v2[score_idx] - part[0]) / (part[1] - part[0])
            if contrastiveScore_v2_v1[score_idx] >= part[0] and contrastiveScore_v2_v1[score_idx] <= part[1]:
                clsLabel_v2_v1[score_idx] = group_idx
                if part[0] == part[1]:
                    normLabel_v2_v1[score_idx] = 0
                else:
                    normLabel_v2_v1[score_idx] = (contrastiveScore_v2_v1[score_idx] - part[0]) / (part[1] - part[0])
    return clsLabel_v1_v2, clsLabel_v2_v1, normLabel_v1_v2, normLabel_v2_v1

def get_contrastive_bcelabel(group, contrastiveScore_v1_v2, contrastiveScore_v2_v1):
    clsLabel_v1_v2 = torch.zeros([contrastiveScore_v1_v2.size(0), len(group)],dtype=torch.float32).cuda()
    normLabel_v1_v2 = torch.zeros(contrastiveScore_v1_v2.size(0),dtype=torch.float64,requires_grad=True).cuda()
    clsLabel_v2_v1 = torch.zeros([contrastiveScore_v2_v1.size(0), len(group)],dtype=torch.float32).cuda()
    normLabel_v2_v1 = torch.zeros(contrastiveScore_v2_v1.size(0),dtype=torch.float64,requires_grad=True).cuda()
    for score_idx in range(contrastiveScore_v1_v2.size(0)):
        for group_idx, part in enumerate(group):
            if round(float(contrastiveScore_v1_v2[score_idx]), 1) >= part[0] and round(float(contrastiveScore_v1_v2[score_idx]), 1) <= part[1]:
                clsLabel_v1_v2[score_idx, group_idx] = 1
                if part[0] == part[1]:
                    normLabel_v1_v2[score_idx] = 0
                else:
                    normLabel_v1_v2[score_idx] = (contrastiveScore_v1_v2[score_idx] - part[0]) / (part[1] - part[0])
            if round(float(contrastiveScore_v2_v1[score_idx]), 1) >= part[0] and round(float(contrastiveScore_v2_v1[score_idx]), 1) <= part[1]:
                clsLabel_v2_v1[score_idx, group_idx] = 1
                if part[0] == part[1]:
                    normLabel_v2_v1[score_idx] = 0
                else:
                    normLabel_v2_v1[score_idx] = (contrastiveScore_v2_v1[score_idx] - part[0]) / (part[1] - part[0])

    return clsLabel_v1_v2, clsLabel_v2_v1, normLabel_v1_v2, normLabel_v2_v1

def get_pred_label(cls_prob_1, cls_prob_2):
    predLabel_v1_v2 = torch.zeros(cls_prob_1.size(0),dtype=torch.int32)
    predLabel_v2_v1 = torch.zeros(cls_prob_2.size(0),dtype=torch.int32)
    for idx in range(cls_prob_1.size(0)):
        predLabel_v1_v2[idx] = cls_prob_1[idx].argmax()
        predLabel_v2_v1[idx] = cls_prob_2[idx].argmax()
    return predLabel_v1_v2, predLabel_v2_v1

def compute_score(group, normLabel_v1_v2, predLable_v1_v2, judge_scores_exemplar, diff):
    predict_score = []
    raw_score = []
    for bs in range(normLabel_v1_v2.size(0)):
        preds = 0.0
        for judge in range(normLabel_v1_v2.size(1)):
            judge_score = judge_scores_exemplar[bs][judge] + group[predLable_v1_v2[bs][judge]][0] + normLabel_v1_v2[bs][judge] * (group[predLable_v1_v2[bs][judge]][1] - group[predLable_v1_v2[bs][judge]][0])
            if judge_score < group[0][0]:
                judge_score = 0
            elif judge_score > group[-1][1]:
                judge_score = 10
            preds += judge_score
        predict_score.append(float(preds) * 0.6 * diff[bs])
        raw_score.append(float(preds) * 0.6)
    return predict_score, raw_score

def train(train_loader, model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, mse, nll, bce, group,
          true_scores, pred_scores, train_raw_preds, train_complement, diffs, optimizer, epoch, group_exe, group_sync):

    # ViT更新参数
    model.train()
    # --------ViT冻结--------
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    # --------ViT冻结--------
    encoder.train()
    fuser.train()
    evnet.train()
    decoder.train()
    rater_exe.train()
    rater_sync.train()
    mlp.train()
    # 初始化一些变量
    num_iter = 0
    batch_loss = 0.0
    batch_loss_alpha = 0.0
    batch_loss_beta = 0.0
    end = time.time()
    for batch_idx, (data, exemplar) in enumerate(train_loader):
        start = time.time()
        loss = 0.0
        # break
        num_iter += 1
        opti_flag = False

        # 分数
        # 总分
        true_scores.extend(data['final_score'].numpy())
        # 执行分数
        true_score = data['completeness'].numpy()
        train_complement.extend(true_score)
        true_score = torch.tensor(true_score, requires_grad=True).cuda()

        # data preparing
        # video_1 is the test video ; video_2 is exemplar
        if args.benchmark == 'MTL':
            # 输入视频
            video_1 = data['video'].float().cuda()  # N, C, T, H, W
            batch_size = video_1.size(0)
            judge_scores = [sorted(items)[2:5] for items in list(data['judge_scores'].numpy())]
            judge_scores = torch.tensor(judge_scores)
            # 示例视频
            video_2 = exemplar['video'].float().cuda()
            judge_scores_exemplar = [sorted(items)[2:5] for items in list(exemplar['judge_scores'].numpy())]
            judge_scores_exemplar = torch.tensor(judge_scores_exemplar)
            if args.usingDD:
                label_1 = data['completeness'].float().reshape(-1, 1).cuda()
                label_2 = exemplar['completeness'].float().reshape(-1,1).cuda()
            else:
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                label_2 = exemplar['final_score'].float().reshape(-1,1).cuda()
            # if not args.dive_number_choosing and args.usingDD:
            # assert (data['difficulty'].float() == target['difficulty'].float()).all()
            diff = data['difficulty'].float().numpy()
            # exemplar video
            # video_2 = target['video'].float().cuda() # N, C, T, H, W
        elif args.benchmark == 'FineDiving':
            # ==================输入视频====================
            # 全局
            video_1 = data['video'].float().cuda()  # N, C, T, H, W
            # 局部
            local_1 = data['local_video'].float().cuda()   # N, C, T, H, W
            batch_size = video_1.size(0)
            # 执行分数
            exe_scores = [items for items in list(data['exe_scores'].numpy())]
            exe_scores = torch.tensor(np.stack(exe_scores, 0))
            alpha_true_1 = torch.abs(exe_scores[:, 0] - exe_scores[:, 1]).float().reshape(-1, 1).cuda()   # |E1 - E2|
            beta_true_1 = exe_scores[:, 0] + exe_scores[:, 1]   # E1 + E2
            beta_true_1 = beta_true_1.float().reshape(-1, 1).cuda()
            # 同步分数
            sync_scores = [items for items in list(data['sync_scores'].numpy())]
            sync_scores = torch.tensor(np.stack(sync_scores, 0))
            # ==================示例视频==================
            # 全局
            video_2 = exemplar['video'].float().cuda()
            # 局部
            local_2 = exemplar['local_video'].float().cuda()
            # 执行分数
            exe_scores_exemplar = [items for items in list(exemplar['exe_scores'].numpy())]
            exe_scores_exemplar = torch.tensor(np.stack(exe_scores_exemplar, 0))
            alpha_true_2 = torch.abs(exe_scores_exemplar[:, 0] - exe_scores_exemplar[:, 1]).float().reshape(-1, 1).cuda()
            beta_true_2 = exe_scores_exemplar[:, 0] + exe_scores_exemplar[:, 1]
            beta_true_2 = beta_true_2.float().reshape(-1, 1).cuda()
            # 同步分数
            sync_scores_exemplar = [items for items in list(exemplar['sync_scores'].numpy())]
            sync_scores_exemplar = torch.tensor(np.stack(sync_scores_exemplar, 0))
            if args.usingDD:
                label_1 = data['completeness'].float().reshape(-1, 1).cuda()
                label_2 = exemplar['completeness'].float().reshape(-1, 1).cuda()
            else:
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                label_2 = exemplar['final_score'].float().reshape(-1, 1).cuda()

            diff = data['difficulty'].float().numpy()
        elif args.benchmark == 'Seven':
            video_1 = data['video'].float().cuda()  # N, C, T, H, W
            label_1 = data['final_score'].float().reshape(-1, 1).cuda()
            # video_2 = target['video'].float().cuda()
            # label_2 = target['final_score'].float().reshape(-1,1).cuda()
            diff = None
        else:
            raise NotImplementedError()
        # forward
        if num_iter == args.step_per_update:
            num_iter = 0
            opti_flag = True

        # =========================   1-Model  =====================================

        # 获得3个子视频
        # [bs,1,48,3,224,224] video_1, video_2, feature_1, feature_2, encoder_res_1, encoder_res_2, combine_feature_1, combine_feature_2, combine_feature, decoder_res
        batch_size = video_1.size(0)
        # bs,1,64,3,224,224
        video_1 = video_1.permute(0, 2, 1, 3, 4).unsqueeze(1)
        video_2 = video_2.permute(0, 2, 1, 3, 4).unsqueeze(1)
        local_1 = local_1.permute(0, 2, 1, 3, 4).unsqueeze(1)
        local_2 = local_2.permute(0, 2, 1, 3, 4).unsqueeze(1)
        # # backbone
        # bs,1,64,768
        feature_1 = model(video_1)
        feature_2 = model(video_2)
        local_feature_1 = model(local_1)
        local_feature_2 = model(local_2)
        # # Encoder
        # bs,1,65,768
        encoder_res_1, encoder_local_1 = encoder(feature_1, local_feature_1)
        encoder_res_2, encoder_local_2 = encoder(feature_2, local_feature_2)
        # CFF
        # bs, 1, 768
        fuse_12 = fuser(encoder_res_1[:,:,0,:], encoder_res_2[:,:,0,:])
        fuse_21 = fuser(encoder_res_2[:,:,0,:], encoder_res_1[:,:,0,:])
        fuse_local_12 = fuser(encoder_local_1[:,:,0,:], encoder_local_2[:,:,0,:])
        fuse_local_21 = fuser(encoder_local_2[:,:,0,:], encoder_local_1[:,:,0,:])
        fuse_local_12 = fuser(q=fuse_12, v=fuse_local_12)
        fuse_local_21 = fuser(q=fuse_21, v=fuse_local_21)
        # evnet alpha = |E1 - E2| / beta = E1 + E2
        alpha_1, beta_1 = evnet(fuse_21)    # 全局-执行 / test
        alpha_2, beta_2 = evnet(fuse_12)    # 全局-执行 / exemplar
        loss_alpha = 0.0
        loss_alpha = loss_alpha + mse(alpha_1.reshape(-1, 1).float(), alpha_true_1)\
                     + mse(alpha_2.reshape(-1, 1).float(), alpha_true_2)
        loss_beta = 0.0
        loss_beta = loss_beta + mse(beta_1.reshape(-1, 1).float(), beta_true_1)\
                    + mse(beta_2.reshape(-1, 1).float(), beta_true_2)
        # relative alpha / relative beta
        alpha_12 = (alpha_1 - alpha_2) / 10.  # test - exemplar
        alpha_21 = (alpha_2 - alpha_1) / 10.  # exemplar - test
        beta_12 = (beta_1 - beta_2) / 20.  # test - exemplar
        beta_21 = (beta_2 - beta_1) / 20.  # exemplar - test
        ab_12 = torch.cat((alpha_12, beta_12),dim=-1)
        ab_12 = ab_12.expand(-1, 3, -1)
        ab_21 = torch.cat((alpha_21, beta_21),dim=-1)
        ab_21 = ab_21.expand(-1, 3, -1)
        ab_feature = torch.cat((ab_12, ab_21),dim=0)
        # bs, 1, 768 * 2
        combine_feature_1 = torch.cat((fuse_21, fuse_12),dim=-1) # v1-v2
        combine_feature_2 = torch.cat((fuse_12, fuse_21),dim=-1) # v2-v1
        combine_feature_local_1 = torch.cat((fuse_local_21, fuse_local_12),dim=-1)
        combine_feature_local_2 = torch.cat((fuse_local_12, fuse_local_21),dim=-1)
        # bs*2, 1, 768 * 2
        combine_feature = torch.cat((combine_feature_1,combine_feature_2),dim=0) # v1-v2 / v2-v1
        combine_feature_local = torch.cat((combine_feature_local_1,combine_feature_local_2),dim=0)
        # # Decoder
        decoder_res = decoder(combine_feature, 'exe_query')
        decoder_res_local = decoder(combine_feature_local, 'sync_query')
        decoder_res_local = torch.cat((decoder_res_local, ab_feature),dim=-1)
        # # rater
        reg, cls_prob, cls_prob_lf = rater_exe(decoder_res)
        reg_local, cls_prob_local, cls_prob_lf_local = rater_sync(decoder_res_local)
        # =========================   1-Model  =====================================


        # =======================  2-Results processing  ============================
        # -----
        # bs*3
        # -----1=执行分数回归值
        reg_1 = reg[0:reg.size(0) // 2].view(-1)  # v1-v2
        reg_2 = reg[reg.size(0) // 2:].view(-1)  # v2-v1
        # -----1=同步分数回归值
        reg_local_1 = reg_local[0:reg_local.size(0) // 2].view(-1)
        reg_local_2 = reg_local[reg_local.size(0) // 2:].view(-1)
        # -------------------
        # bs,3,11
        # -----2=执行分数分类结果
        cls_prob_1 = cls_prob[0:cls_prob.size(0) // 2]  # v1-v2
        cls_prob_2 = cls_prob[cls_prob.size(0) // 2:]  # v2-v1
        # -----2=同步分数分类结果
        cls_prob_local_1 = cls_prob_local[0:cls_prob_local.size(0) // 2]
        cls_prob_local_2 = cls_prob_local[cls_prob_local.size(0) // 2:]
        # -------------------

        # 1-计算两个视频的相对裁判分数
        # 相对执行分数
        contrastiveScore_v1_v2_exe = exe_scores - exe_scores_exemplar
        contrastiveScore_v2_v1_exe = exe_scores_exemplar - exe_scores
        # 相对同步分数
        contrastiveScore_v1_v2_sync = sync_scores - sync_scores_exemplar
        contrastiveScore_v2_v1_sync = sync_scores_exemplar - sync_scores

        # 制作相对执行分数分类标签，相对执行分数归一化标签                                          # exe
        clsLabel_v1_v2_exe, clsLabel_v2_v1_exe, normLabel_v1_v2_exe, normLabel_v2_v1_exe = get_contrastive_bcelabel(group_exe, contrastiveScore_v1_v2_exe.view(-1), contrastiveScore_v2_v1_exe.view(-1))
        # 制作相对同步分数分类标签，相对同步分数归一化标签                                              # sync
        clsLabel_v1_v2_sync, clsLabel_v2_v1_sync, normLabel_v1_v2_sync, normLabel_v2_v1_sync = get_contrastive_bcelabel(group_sync, contrastiveScore_v1_v2_sync.view(-1), contrastiveScore_v2_v1_sync.view(-1))

        # =======================  2-Results processing  ============================


        # =======================  3-Loss computing  =============================
        #                            计算分类损失
        # loss_cls = 0.0
        loss_cls_exe = 0.0
        loss_cls_sync = 0.0
        # 计算相对执行分数的分类损失       # exe
        loss_cls_exe = loss_cls_exe + bce(cls_prob_1.view(-1, *cls_prob_1.shape[2:]), clsLabel_v1_v2_exe)
        loss_cls_exe = loss_cls_exe + bce(cls_prob_2.view(-1, *cls_prob_2.shape[2:]), clsLabel_v2_v1_exe)
        # 计算相对同步分数的分类损失         # sync
        loss_cls_sync = loss_cls_sync + bce(cls_prob_local_1.view(-1, *cls_prob_local_1.shape[2:]), clsLabel_v1_v2_sync)
        loss_cls_sync = loss_cls_sync + bce(cls_prob_local_2.view(-1, *cls_prob_local_2.shape[2:]), clsLabel_v2_v1_sync)
        # 相对执行分数和相对同步分数的分类损失相加为总分类损失
        loss = loss + loss_cls_exe + loss_cls_sync + loss_alpha + 0.01 * loss_beta
        #
        #                             计算回归损失
        loss_reg_exe = 0.0
        loss_reg_sync = 0.0
        num_reg = 0
        # generate mask
        if args.benchmark == "FineDiving":
            # ---------执行分数
            clsLabel_v1_v2_exe_indices = clsLabel_v1_v2_exe.argmax(-1)
            clsLabel_v2_v1_exe_indices = clsLabel_v2_v1_exe.argmax(-1)
            mask_1_exe = [True if int(item.data.cpu()) in [0, 8] else False for item in clsLabel_v1_v2_exe_indices]
            mask_2_exe = [True if int(item.data.cpu()) in [0, 8] else False for item in clsLabel_v2_v1_exe_indices]
            # ---------同步分数
            clsLabel_v1_v2_sync_indices = clsLabel_v1_v2_sync.argmax(-1)
            clsLabel_v2_v1_sync_indices = clsLabel_v2_v1_sync.argmax(-1)
            mask_1_sync = [True if int(item.data.cpu()) in [0, 8] else False for item in clsLabel_v1_v2_sync_indices]
            mask_2_sync = [True if int(item.data.cpu()) in [0, 8] else False for item in clsLabel_v2_v1_sync_indices]
            # ---------
            mask_1_exe = torch.tensor(mask_1_exe).cuda()
            mask_2_exe = torch.tensor(mask_2_exe).cuda()
            mask_1_sync = torch.tensor(mask_1_sync).cuda()
            mask_2_sync = torch.tensor(mask_2_sync).cuda()
        # 执行分数
        if mask_1_exe.sum() != 0:         # exe
            loss_reg_exe = loss_reg_exe + mse(reg_1[mask_1_exe].reshape(-1, 1).float(), normLabel_v1_v2_exe[mask_1_exe].reshape(-1, 1).float())
            loss_reg_exe = loss_reg_exe + mse(reg_2[mask_2_exe].reshape(-1, 1).float(), normLabel_v2_v1_exe[mask_2_exe].reshape(-1, 1).float())
            loss = loss + loss_reg_exe
        # 同步分数
        if mask_1_sync.sum() != 0:          # sync
            loss_reg_sync = loss_reg_sync + mse(reg_local_1[mask_1_sync].reshape(-1, 1).float(), normLabel_v1_v2_sync[mask_1_sync].reshape(-1, 1).float())
            loss_reg_sync = loss_reg_sync + mse(reg_local_2[mask_2_sync].reshape(-1, 1).float(), normLabel_v2_v1_sync[mask_2_sync].reshape(-1, 1).float())
            loss = loss + loss_reg_sync
        # ----------------
        # =======================  3-Loss computing  =============================

        # =======================    4-Optimizing    ===========================

        if opti_flag:
            optimizer.zero_grad()

        loss.backward()

        if opti_flag:
            optimizer.step()

        # =======================    4-Optimizing    ===========================

        # ===================== 5-Quality Score operation ===========================

        # 1-预测的执行分数分类结果
        predLabel_v1_v2 = cls_prob_1.argmax(-1)
        # 2-预测的同步分数分类结果
        predLabel_local_v1_v2 = cls_prob_local_1.argmax(-1)

        # 3-计算执行分数
        predicts_exe = compute_referee_score(group_exe, reg_1.data.cpu().view(batch_size, 2), predLabel_v1_v2.data.cpu(), exe_scores_exemplar.numpy(), batch_size, referee_num=2)
        # 4-计算同步分数
        predicts_sync = compute_referee_score(group_sync, reg_local_1.data.cpu().view(batch_size, 3), predLabel_local_v1_v2.data.cpu(), sync_scores_exemplar.numpy(), batch_size, referee_num=3)
        # 5-计算总分
        raw = torch.cat((predicts_exe, predicts_sync), dim=-1)
        raw = torch.sum(raw, dim=-1)
        predicts = [(item * 0.6 * d) for item, d in zip(raw.numpy(), diff)]
        pred_scores.extend(predicts)
        train_raw_preds.extend([(item * 0.6) for item in raw.numpy()])

        # ===================== 5-Quality Score operation ===========================

        # -----------------
        batch_loss += loss.data.cpu().numpy()
        batch_loss_alpha += loss_alpha.data.cpu().numpy()
        batch_loss_beta += loss_beta.data.cpu().numpy()
        # ExeCls_loss = loss_cls_exe.data.cpu().numpy()
        # SyncCls_loss = loss_cls_sync.data.cpu().numpy()
        # ExeReg_loss = loss_reg_exe.data.cpu().numpy()
        # SyncReg_loss = loss_reg_sync.data.cpu().numpy()
        end = time.time()
        batch_time = end - start
        if batch_idx % args.print_freq == 0:
            print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t ExeCls_loss : %.4f \t SyncCls_loss : %.4f'
                  % (epoch, args.max_epoch, batch_idx, len(train_loader), batch_time, loss.item(), loss_cls_exe.item(), loss_cls_sync.item()))

    return batch_loss, batch_loss_alpha, batch_loss_beta


def validate(test_loader, model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, mse, nll, group,
          true_scores, pred_scores, test_raw_preds, test_complement,diffs, optimizer, epoch, group_exe, group_sync):

    # switch to train mode
    model.eval()
    encoder.eval()
    fuser.eval()
    evnet.eval()
    decoder.eval()
    rater_exe.eval()
    rater_sync.eval()
    mlp.eval()
    # 初始化一些变量
    num_iter = 0
    batch_loss = 0.0
    loss_cls = 0.0
    loss_reg = 0.0
    Acc_list_exe = [0, 0, 0]
    Acc_list_sync = [0, 0, 0]
    mae = 0.0
    cls_num_exe = 0
    cls_num_sync = 0
    reg_num = 0
    end = time.time()
    with torch.no_grad():
        for i, (data, exemplars) in enumerate(test_loader):
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            true_score = data['completeness'].numpy()
            test_complement.extend(true_score)
            true_score = torch.tensor(true_score, requires_grad=True).cuda()
            # data preparing
            # video_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'FineDiving':
                # ==================输入视频====================
                # 全局
                video_1 = data['video'].float().cuda()  # N, C, T, H, W
                # 局部
                local_1 = data['local_video'].float().cuda()
                batch_size = video_1.size(0)
                # 执行分数
                exe_scores = [items for items in list(data['exe_scores'].numpy())]
                exe_scores = torch.tensor(np.stack(exe_scores, 0))
                # 同步分数
                sync_scores = [items for items in list(data['sync_scores'].numpy())]
                sync_scores = torch.tensor(np.stack(sync_scores, 0))

                judge_scores = [items for items in list(data['judge_scores'].numpy())]
                judge_scores = torch.tensor(np.stack(judge_scores, 0))

                # ==================示例视频==================
                # 全局
                video_2_list = [exemplar['video'].float().cuda() for exemplar in exemplars]
                # 局部
                local_2_list = [exemplar['local_video'].float().cuda() for exemplar in exemplars]
                # 执行分数
                exe_scores_exemplars_list = []
                for exemplar in exemplars:
                    exe_scores_exemplar = [items for items in list(exemplar['exe_scores'].numpy())]
                    exe_scores_exemplar = torch.tensor(np.stack(exe_scores_exemplar, 0))
                    exe_scores_exemplars_list.append(exe_scores_exemplar)
                # 同步分数
                sync_scores_exemplars_list = []
                for exemplar in exemplars:
                    sync_scores_exemplar = [items for items in list(exemplar['sync_scores'].numpy())]
                    sync_scores_exemplar = torch.tensor(np.stack(sync_scores_exemplar, 0))
                    sync_scores_exemplars_list.append(sync_scores_exemplar)
                judge_scores_exemplars_list = []
                for exemplar in exemplars:
                    judge_scores_exemplar = [items for items in list(exemplar['judge_scores'].numpy())]
                    judge_scores_exemplar = torch.tensor(np.stack(judge_scores_exemplar, 0))
                    judge_scores_exemplars_list.append(judge_scores_exemplar)
                diff = data['difficulty'].float().numpy()
                diffs.extend(diff)
            else:
                raise NotImplementedError()

            # =========================   1-Model  =====================================
            # ----
            video_1 = video_1.permute(0, 2, 1, 3, 4).unsqueeze(1)
            local_1 = local_1.permute(0, 2, 1, 3, 4).unsqueeze(1)
            # # Encoder
            # encoder_res_1 = encoder(feature_1)
            # encoder_local_1 = encoder(local_feature_1)
            # ----
            batch_size = video_1.size(0)
            predicts_sum = np.zeros(batch_size)
            raw_sum = np.zeros(batch_size)
            loss = 0.0
            for video_2, local_2, exe_scores_exemplar, sync_scores_exemplar, judge_scores_exemplar in zip(video_2_list, local_2_list, exe_scores_exemplars_list, sync_scores_exemplars_list, judge_scores_exemplars_list):

                # 获得3个子视频
                # bs,1,64,3,224,224
                video_2 = video_2.permute(0, 2, 1, 3, 4).unsqueeze(1)
                local_2 = local_2.permute(0, 2, 1, 3, 4).unsqueeze(1)
                # # backbone
                # bs,3,48,768
                # # backbone
                feature_1 = model(video_1)
                local_feature_1 = model(local_1)
                feature_2 = model(video_2)
                local_feature_2 = model(local_2)
                # # Encoder
                # bs,1,65,768
                # after
                encoder_res_1, encoder_local_1 = encoder(feature_1, local_feature_1)
                encoder_res_2, encoder_local_2 = encoder(feature_2, local_feature_2)
                # CFF
                # bs,1,768 * 2
                fuse_12 = fuser(encoder_res_1[:, :, 0, :], encoder_res_2[:, :, 0, :])
                fuse_21 = fuser(encoder_res_2[:, :, 0, :], encoder_res_1[:, :, 0, :])
                fuse_local_12 = fuser(encoder_local_1[:, :, 0, :], encoder_local_2[:, :, 0, :])
                fuse_local_21 = fuser(encoder_local_2[:, :, 0, :], encoder_local_1[:, :, 0, :])
                fuse_local_12 = fuser(q=fuse_12, v=fuse_local_12)
                fuse_local_21 = fuser(q=fuse_21, v=fuse_local_21)
                # evnet alpha = |E1 - E2| / beta = E1 + E2
                alpha_1, beta_1 = evnet(fuse_21)  # 全局-执行 / test
                alpha_2, beta_2 = evnet(fuse_12)  # 全局-执行 / exemplar
                # relative alpha / relative beta
                alpha_12 = (alpha_1 - alpha_2) / 10.  # test - exemplar
                alpha_21 = (alpha_2 - alpha_1) / 10.  # exemplar - test
                beta_12 = (beta_1 - beta_2) / 20.  # test - exemplar
                beta_21 = (beta_2 - beta_1) / 20.  # exemplar - test
                ab_12 = torch.cat((alpha_12, beta_12), dim=-1)
                ab_12 = ab_12.expand(-1, 3, -1)
                ab_21 = torch.cat((alpha_21, beta_21), dim=-1)
                ab_21 = ab_21.expand(-1, 3, -1)
                ab_feature = torch.cat((ab_12, ab_21), dim=0)
                # bs*2,1,768 * 2
                combine_feature_1 = torch.cat((fuse_21, fuse_12), dim=-1)
                combine_feature_2 = torch.cat((fuse_12, fuse_21), dim=-1)
                combine_feature_local_1 = torch.cat((fuse_local_21, fuse_local_12), dim=-1)
                combine_feature_local_2 = torch.cat((fuse_local_12, fuse_local_21), dim=-1)
                combine_feature = torch.cat((combine_feature_1, combine_feature_2), dim=0)
                combine_feature_local = torch.cat((combine_feature_local_1, combine_feature_local_2), dim=0)
                # # Decoder
                # bs*2,1,3,768 * 2
                decoder_res = decoder(combine_feature, 'exe_query')
                decoder_res_local = decoder(combine_feature_local, 'sync_query')
                decoder_res_local = torch.cat((decoder_res_local, ab_feature), dim=-1)
                # # rater
                # bs*2,1,3,11
                reg, cls_prob, cls_prob_lf = rater_exe(decoder_res)
                reg_local, cls_prob_local, cls_prob_lf_local = rater_sync(decoder_res_local)
                # =========================   1-Model  =====================================

                # =======================  2-Results processing  ============================
                # -----
                # -----1=执行分数回归值
                reg_1 = reg[0:reg.size(0) // 2].view(-1)
                reg_2 = reg[reg.size(0) // 2:].view(-1)
                # -----1=同步分数回归值
                reg_local_1 = reg_local[0:reg_local.size(0) // 2].view(-1)
                reg_local_2 = reg_local[reg_local.size(0) // 2:].view(-1)
                # -------------------
                # -----2=执行分数分类结果
                cls_prob_1 = cls_prob[0:cls_prob.size(0) // 2]
                cls_prob_2 = cls_prob[cls_prob.size(0) // 2:]
                cls_prob_lf_1 = cls_prob_lf[0:cls_prob_lf.size(0) // 2].view(-1, *cls_prob_1.shape[2:])
                cls_prob_lf_2 = cls_prob_lf[cls_prob_lf.size(0) // 2:].view(-1, *cls_prob_1.shape[2:])
                # -----2=同步分数分类结果
                cls_prob_local_1 = cls_prob_local[0:cls_prob_local.size(0) // 2]
                cls_prob_local_2 = cls_prob_local[cls_prob_local.size(0) // 2:]
                # -----

                # -----
                # 1-计算预测的相对裁判分数标签
                # 相对执行分数
                contrastiveScore_v1_v2_exe = exe_scores - exe_scores_exemplar
                contrastiveScore_v2_v1_exe = exe_scores_exemplar - exe_scores
                # 相对同步分数
                contrastiveScore_v1_v2_sync = sync_scores - sync_scores_exemplar
                contrastiveScore_v2_v1_sync = sync_scores_exemplar - sync_scores

                # 制作相对执行分数分类标签，相对执行分数归一化标签                                          # exe
                clsLabel_v1_v2_exe, clsLabel_v2_v1_exe, normLabel_v1_v2_exe, normLabel_v2_v1_exe = get_contrastive_bcelabel(
                    group_exe, contrastiveScore_v1_v2_exe.view(-1), contrastiveScore_v2_v1_exe.view(-1))
                # 制作相对同步分数分类标签，相对同步分数归一化标签                                              # sync
                clsLabel_v1_v2_sync, clsLabel_v2_v1_sync, normLabel_v1_v2_sync, normLabel_v2_v1_sync = get_contrastive_bcelabel(
                    group_sync, contrastiveScore_v1_v2_sync.view(-1), contrastiveScore_v2_v1_sync.view(-1))
                # -----
                # =======================  2-Results processing  ============================

                # ===================== 3-Quality Score operation ===========================
                # 1-预测的执行分数分类结果
                predLabel_v1_v2 = cls_prob_1.argmax(-1)
                # 2-预测的同步分数分类结果
                predLabel_local_v1_v2 = cls_prob_local_1.argmax(-1)
                # 3-计算执行分数
                predicts_exe = compute_referee_score(group_exe, reg_1.data.cpu().view(batch_size, 2),
                                                     predLabel_v1_v2.data.cpu(), exe_scores_exemplar.numpy(),
                                                     batch_size, referee_num=2)
                # 4-计算同步分数
                predicts_sync = compute_referee_score(group_sync, reg_local_1.data.cpu().view(batch_size, 3),
                                                      predLabel_local_v1_v2.data.cpu(), sync_scores_exemplar.numpy(),
                                                      batch_size, referee_num=3)
                # 5-计算总分
                raw = torch.cat((predicts_exe, predicts_sync), dim=-1)
                raw = torch.sum(raw, dim=-1)
                # 预测总分
                predicts = [(item * 0.6 * d) for item, d in zip(raw.numpy(), diff)]
                predicts = np.array(predicts)
                # 预测执行分数
                pred_raw = [(item * 0.6) for item in raw.numpy()]
                pred_raw = np.array(pred_raw)
                # 总分累加
                predicts_sum = predicts_sum + predicts
                # 执行分数累加
                raw_sum = raw_sum + pred_raw

                # 6-计算分类准确率Acc
                dist_exe = torch.abs(predLabel_v1_v2.view(-1).cpu() - clsLabel_v1_v2_exe.argmax(-1).cpu())
                dist_sync = torch.abs(predLabel_local_v1_v2.view(-1).cpu() - clsLabel_v1_v2_sync.argmax(-1).cpu())
                cls_num_exe += dist_exe.size(0)
                cls_num_sync += dist_sync.size(0)
                # 执行分数分类准确率
                for item in dist_exe:
                    if int(item) == 0:
                        Acc_list_exe[0] += 1
                    if int(item) <= 1:
                        Acc_list_exe[1] += 1
                    if int(item) <= 2:
                        Acc_list_exe[2] += 1
                # 同步分数分类准确率
                for item in dist_sync:
                    if int(item) == 0:
                        Acc_list_sync[0] += 1
                    if int(item) <= 1:
                        Acc_list_sync[1] += 1
                    if int(item) <= 2:
                        Acc_list_sync[2] += 1
                # ===================== 3-Quality Score operation ===========================

            # -------
            end = time.time()
            batch_time = end - start
            if i % args.print_freq == 0:
                print('[Testing][%d/%d][%d/%d] \t Batch_time %.2f\t'% (epoch, args.max_epoch, i, len(test_loader), batch_time))
            voter_num = len(video_2_list)
            pred_scores.extend([(pred / voter_num) for pred in predicts_sum])
            test_raw_preds.extend([(raw / voter_num) for raw in raw_sum])
        # 执行分数分类准确率
        for idx in range(len(Acc_list_exe)):
            Acc_list_exe[idx] = Acc_list_exe[idx] / cls_num_exe
        # 同步分数分类准确率
        for idx in range(len(Acc_list_sync)):
            Acc_list_sync[idx] = Acc_list_sync[idx] / cls_num_sync
    return Acc_list_exe, Acc_list_sync


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if __name__ == '__main__':
    main()
