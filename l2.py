import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from dataset import TSNDataSet
from MTL_dataset import MTLPair_Dataset
from models import TSN
from transforms import *
# from opts import parser
from utils import parser
from scipy import stats
from torchvideotransforms import video_transforms, volume_transforms
from utils import misc
from utils import save_model
from utils import resume_train
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))
best_prec1 = 0
from models_tools import vit_encoder
from models_tools import MLP
from models_tools import MLP_timesformer
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models_tools.pspnet import PSPNet
from timesformer.models.vit import TimeSformer

def video_segment(args,video):
    # bs,3,96,224,224
    batch_size = video.size()[0]
    video_pack = torch.zeros(batch_size, args.sub_video_num, args.segments_num, args.segment_frame_num, 3, args.frame_HW,args.frame_HW).cuda()
    # bs,1,96,3,224,224
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
                # 这是第二个视频取偶数帧
                # 0 :32:2->[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
                # 32:64:2->[32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]
                # 64:96:2->[64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]
                video_seg = torch.cat(
                    [video[bs, :, start_idx[i]:start_idx[i + 1]:2] for i in range(0, len(start_idx) - 1)])
                video_pack[bs, 0] = video_seg
            else:
                # 总共取3个视频
                # 这是第一个视频取三个阶段的交集
                video_seg = torch.cat([video[bs, :, i: i + args.segment_frame_num] for i in intersection_idx])
                video_pack[bs, 0] = video_seg
                # 这是第二个视频取偶数帧
                # 0 :32:2->[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
                # 32:64:2->[32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]
                # 64:96:2->[64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]
                video_seg = torch.cat([video[bs, :, start_idx[i]:start_idx[i + 1]:2] for i in range(0, len(start_idx) - 1)])
                video_pack[bs, 1] = video_seg
                # 这是第三个视频取奇数帧
                # 1 :32:2->[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                # 33:64:2->[33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                # 65:96:2->[65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
                video_seg = torch.cat([video[bs, :, start_idx[i] + 1:start_idx[i + 1]:2] for i in range(0, len(start_idx) - 1)])
                video_pack[bs, 2] = video_seg
    return video_pack

def train_video_segment(args,video):
    # 训练集只划分出一个视频
    # 以16帧采样为例子
    # 训练集由一个96帧视频分为1个16帧的视频
    # bs,3,96,224,224
    batch_size = video.size()[0]
    # bs,96,3,224,224
    video = video.permute(0, 2, 1, 3, 4)
    # 步长
    stride = 96//args.all_frames
    video_num = stride
    # bs,1,16,3,224,224
    video_pack = torch.zeros(batch_size, 1, args.all_frames, 3, args.frame_HW, args.frame_HW).cuda()
    if args.all_frames == 64:
        pass
    else:
        start_idx = random.randint(0,stride-1)
        for bs in range(0,batch_size):
            video_seg = video[bs,start_idx:96:stride]
            video_pack[bs,0]=video_seg

    return video_pack
def test_video_segment(args,video):
    # 以16帧采样为例
    # 为了进行补偿
    # 测试集由一个96帧视频分为6个16帧的视频
    # bs,3,96,224,224
    batch_size = video.size()[0]
    # bs,96,3,224,224
    video = video.permute(0, 2, 1, 3, 4)
    # 步长
    stride = 96//args.all_frames
    video_num = stride
    # bs,6,16,3,224,224
    # 64帧特殊处理
    if args.all_frames == 64:
        video_pack = torch.zeros(batch_size, 2, args.all_frames, 3, args.frame_HW, args.frame_HW).cuda()
        for bs in range(0,batch_size):
            for i in range(0,stride):
                video_seg = video[bs,i:96:stride]
                video_pack[bs,i]=video_seg
    else:
        video_pack = torch.zeros(batch_size, video_num, args.all_frames, 3, args.frame_HW, args.frame_HW).cuda()
        for bs in range(0,batch_size):
            for i in range(0,stride):
                video_seg = video[bs,i:96:stride]
                video_pack[bs,i]=video_seg
    return video_pack
def main():
    global args, best_prec1
    args = parser.get_args()
    parser.setup(args)
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100

    # args = parser.parse_args()
    # if args.dataset == 'ucf101':
    #     num_class = 101
    # elif args.dataset == 'hmdb51':
    #     num_class = 51
    # elif args.dataset == 'kinetics':
    #     num_class = 400
    # else:
    #     raise ValueError('Unknown dataset '+args.dataset)

    # model = TSN(101, args.num_segments, args.modality,
    #             base_model=args.arch,
    #             consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn).apply(
    #     misc.fix_bn)
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
    encoder = vit_encoder.encoder_fuser(dim=768,num_heads=12,num_layers=12,
                                    segment_frame_num=args.segment_frame_num,
                                        segments_num=args.segments_num,allframes=args.all_frames)
    if args.backbone =="timesformer":
        mlp = MLP_timesformer.MLP_tf(in_channel=768)
    else:
        mlp = MLP.MLP(in_channel=1024)


    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model = model.cuda()
    model = nn.DataParallel(model)
    encoder = encoder.cuda()
    encoder = nn.DataParallel(encoder)
    mlp = mlp.cuda()
    mlp = nn.DataParallel(mlp)

    cudnn.benchmark = True

    # MTL-AQA
    train_loader = torch.utils.data.DataLoader(
        MTLPair_Dataset(args,transform=video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                video_transforms.Resize(resize_scale),
                video_transforms.RandomCrop(resize_size),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
                        subset='train'),
        batch_size=args.bs_train, shuffle=False,num_workers=8, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        MTLPair_Dataset(args,transform=video_transforms.Compose([
                video_transforms.Resize(resize_scale),
                video_transforms.CenterCrop(resize_size),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                        subset='test'),
        batch_size=args.bs_test, shuffle=False,num_workers=8, pin_memory=True
    )

    criterion = nn.MSELoss(reduction='mean').cuda()


    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.base_lr * args.lr_factor},
        {'params': encoder.parameters()},
        {'params': mlp.parameters()}], lr=args.base_lr, weight_decay=args.weight_decay)

    # 一些参数
    # best
    start_epoch = 0
    epoch_best = 0
    epoch_best_L2 = 0
    epoch_best_RL2 = 0
    rho_best = 0
    rho_best_L2 = 0
    rho_best_RL2 = 0
    L2_min = 1000
    L2_min_rho = 0
    L2_min_RL2 = 0
    RL2_min = 1000
    RL2_min_rho = 0
    RL2_min_L2 = 0

    # 复现！
    if args.resume:
        start_epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min = resume_train.resume_train(model, encoder, mlp, optimizer, args)

    # tensorboard
    tensorboard_original_path = '/media/lmz/disk/q/backbone_encoder_decoder/tsn(VIT)-encoder/tsn-pytorch/tensorboard_log/vit_encoder_mlp_frames'
    log_dir_path = os.path.join(tensorboard_original_path,str(args.all_frames))
    writer = SummaryWriter(log_dir=log_dir_path+'/2')

    for epoch in range(start_epoch, args.max_epoch):
        # adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # 初始化变量
        true_scores = []
        pred_scores = []
        test_true_scores = []
        test_pred_scores = []
        raw_preds = []
        diffs = []

        # !!
        # batch_loss = validate(val_loader, model, encoder, mlp, criterion, test_true_scores, test_pred_scores, raw_preds,diffs,optimizer, epoch)

        # # ----------------train----------------
        batch_loss = train(train_loader, model, encoder, mlp, criterion, true_scores, pred_scores, raw_preds, diffs, optimizer, epoch)
        # 评估训练集
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)


        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        print('[TRAIN] EPOCH: %d, correlation: %.4f, batch_loss: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f' % (
        epoch, rho, batch_loss, L2, RL2, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

        # 保存迭代数据
        metrics_root_path = '/media/lmz/disk/q/backbone_encoder_decoder/tsn(VIT)-encoder/tsn-pytorch/Metrics_iterator_res/'
        metrics_path = os.path.join(metrics_root_path, 'frames_' + str(args.all_frames))
        train_path = metrics_path + '/train.txt'
        with open(train_path, mode="a+", encoding="utf-8") as f:
            f.write(time.ctime(time.time()) + "\n")
            f.write("***train_第%d轮***\n" % (epoch + 1))
            f.write('[TRAIN] EPOCH: %d, correlation: %.4f, batch_loss: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f' % (
                        epoch, rho, batch_loss, L2, RL2, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
            f.write('\n')
        f.close()
        # 上传到前端
        writer.add_scalar(tag="train_Src",
                          scalar_value=rho,
                          global_step=epoch + 1)
        writer.add_scalar(tag="train_Loss",
                          scalar_value=batch_loss,
                          global_step=epoch + 1)

        # --------------test----------------
        batch_loss = validate(val_loader, model, encoder, mlp, criterion, test_true_scores, test_pred_scores, raw_preds,
                              diffs,
                              optimizer, epoch)
        # 评估训练集
        test_pred_scores = np.array(test_pred_scores)
        test_true_scores = np.array(test_true_scores)
        # 是否为最佳轮
        flag = False
        rho, p = stats.spearmanr(test_pred_scores, test_true_scores)
        L2 = np.power(test_pred_scores - test_true_scores, 2).sum() / test_true_scores.shape[0]
        RL2 = np.power((test_pred_scores - test_true_scores) / (test_true_scores.max() - test_true_scores.min()),
                       2).sum() / \
              test_true_scores.shape[0]
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
        if rho > rho_best:
            flag = True
            rho_best = rho
            rho_best_L2 = L2
            rho_best_RL2 = RL2
            epoch_best = epoch
            print('-----New best found!-----')
        print('[TEST] EPOCH: %d, correlation: %.6f, RL2: %.6f, L2: %.6f' % (epoch, rho, RL2, L2))
        print('[TEST] EPOCH: %d, best correlation: %.6f, RL2: %.6f, L2: %.6f' % (epoch_best, rho_best, rho_best_RL2, rho_best_L2))
        print('[TEST] EPOCH: %d, best RL2: %.6f, correlation: %.6f, L2: %.6f' % (epoch_best_RL2, RL2_min, RL2_min_rho, RL2_min_L2))
        print('[TEST] EPOCH: %d, best L2: %.6f, correlation: %.6f, RL2: %.6f' % (epoch_best_L2, L2_min, L2_min_rho, L2_min_RL2))

        # 保存模型
        save_model.save_checkpoint(model, encoder, mlp, optimizer, epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args)

        # 保存迭代结果
        # 测试路径
        metrics_root_path = '/media/lmz/disk/q/backbone_encoder_decoder/tsn(VIT)-encoder/tsn-pytorch/Metrics_iterator_res/'
        metrics_path = os.path.join(metrics_root_path, 'frames_' + str(args.all_frames))
        test_path = metrics_path + '/test.txt'
        with open(test_path, mode="a+", encoding="utf-8") as f:
            f.write(time.ctime(time.time()) + "\n")
            f.write("***test_第%d轮***\n" % (epoch + 1))
            if flag:
                f.write('-----New best found!-----\n')
            f.write('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f\n' % (epoch, rho, RL2, L2))
            f.write('[TEST] EPOCH: %d, best correlation: %.6f, L2: %.6f, RL2: %.6f\n' % (
            epoch_best, rho_best, rho_best_RL2, rho_best_L2))
            f.write('[TEST] EPOCH: %d, best RL2: %.6f, correlation: %.6f, L2: %.6f\n' % (
            epoch_best_RL2, RL2_min, RL2_min_rho, RL2_min_L2))
            f.write('[TEST] EPOCH: %d, best L2: %.6f, correlation: %.6f, RL2: %.6f\n' % (
            epoch_best_L2, L2_min, L2_min_rho, L2_min_RL2))
        f.close()
        # 上传到前端
        writer.add_scalar(tag="test_Src",
                          scalar_value=rho,
                          global_step=epoch + 1)
        writer.add_scalar(tag="test_MSE",
                          scalar_value=L2,
                          global_step=epoch + 1)
        writer.add_scalar(tag="test_RL2",
                          scalar_value=RL2,
                          global_step=epoch + 1)


def train(train_loader, model, encoder, mlp, criterion,
          true_scores, pred_scores, raw_preds, diffs, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # if args.no_partialbn:
    #     model.module.partialBN(False)
    # else:
    #     model.module.partialBN(True)
    # model.module.partialBN(True)

    # switch to train mode
    model.train()
    encoder.train()
    mlp.train()
    # 初始化一些变量
    num_iter = 0
    batch_loss = 0.0
    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        start = time.time()
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0.0
        # break
        num_iter += 1
        opti_flag = False

        true_scores.extend(data['final_score'].numpy())
        true_score = data['completeness'].numpy()
        true_score = torch.tensor(true_score, requires_grad=True).cuda()
        # data preparing
        # video_1 is the test video ; video_2 is exemplar
        if args.benchmark == 'MTL':
            video_1 = data['video'].float().cuda()  # N, C, T, H, W
            if args.usingDD:
                label_1 = data['completeness'].float().reshape(-1, 1).cuda()
                # label_2 = target['completeness'].float().reshape(-1,1).cuda()
            else:
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                # label_2 = target['final_score'].float().reshape(-1,1).cuda()
            # if not args.dive_number_choosing and args.usingDD:
            # assert (data['difficulty'].float() == target['difficulty'].float()).all()
            diff = data['difficulty'].float().numpy()
            # exemplar video
            # video_2 = target['video'].float().cuda() # N, C, T, H, W

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

        # bs,3,96,224,224
        # bs,1,16,3,224,224
        video_pack=train_video_segment(args,video_1)
        # compute output
        # bs,1,16,768 提取特征
        output = model(video_pack)
        # bs,1,17,768 时序编码
        encoder_res = encoder(output)
        # cls_tokens
        # bs,1,768
        video_level_feature = encoder_res[:,:,0,:]
        # Mlp
        # bs,1
        predicts = mlp(video_level_feature).squeeze(-1)
        # loss
        # for b in range(0,predicts.size()[0]):
        #     for v in range(0,predicts.size()[1]):
        #         loss += criterion(predicts[b][v].float(),true_score[b].float())
        predicts = predicts.mean(1)
        loss +=criterion(predicts.float(),true_score.float())
        batch_loss += loss.data.cpu().numpy()
        if opti_flag:
            optimizer.zero_grad()
        loss.backward()
        if opti_flag:
            optimizer.step()
        end = time.time()
        batch_time = end - start
        if batch_idx % args.print_freq == 0:
            print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
                  % (epoch, args.max_epoch, batch_idx, len(train_loader),
                     batch_time, loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        # predicts = torch.round(predicts,decimals=3).data.cpu().numpy()
        # predicts = predicts * 1000
        # predicts = torch.round(predicts)
        # predicts = predicts / 1000
        predicts = predicts.data.cpu().numpy()
        raw_preds.extend(pred_scores)
        diffs.extend(diff)
        # pred_scores.extend([round(i[0].item() * i[1].item(),3) for i in zip(predicts, diff)])
        pred_scores.extend([i[0].item() * i[1].item() for i in zip(predicts, diff)])
    return batch_loss
        # loss.backward()
        #
        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        #
        # optimizer.step()
        #
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        #
        # if i % args.print_freq == 0:
        #     print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #            epoch, i, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(test_loader, model, encoder, mlp, criterion,
          true_scores, pred_scores, raw_preds, diffs, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # if args.no_partialbn:
    #     model.module.partialBN(False)
    # else:
    #     model.module.partialBN(True)
    # model.module.partialBN(True)

    # switch to train mode
    model.eval()
    encoder.eval()
    mlp.eval()
    # 初始化一些变量
    num_iter = 0
    batch_loss = 0.0
    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            start = time.time()
            # measure data loading time
            data_time.update(time.time() - end)

            true_scores.extend(data['final_score'].numpy())
            true_score = data['completeness'].numpy()
            true_score = torch.tensor(true_score, requires_grad=True).cuda()
            # data preparing
            # video_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'MTL':
                video_1 = data['video'].float().cuda()  # N, C, T, H, W
                if args.usingDD:
                    label_1 = data['completeness'].float().reshape(-1, 1).cuda()
                    # label_2 = target['completeness'].float().reshape(-1,1).cuda()
                else:
                    label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                    # label_2 = target['final_score'].float().reshape(-1,1).cuda()
                # if not args.dive_number_choosing and args.usingDD:
                # assert (data['difficulty'].float() == target['difficulty'].float()).all()
                diff = data['difficulty'].float().numpy()
                # exemplar video
                # video_2 = target['video'].float().cuda() # N, C, T, H, W

            elif args.benchmark == 'Seven':
                video_1 = data['video'].float().cuda()  # N, C, T, H, W
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                # video_2 = target['video'].float().cuda()
                # label_2 = target['final_score'].float().reshape(-1,1).cuda()
                diff = None
            else:
                raise NotImplementedError()

            # 获得3个子视频
            # [bs,6,16,3,224,224]
            video_pack=test_video_segment(args,video_1)
            # compute output
            # bs,6,16,768 提取特征
            output = model(video_pack)
            # bs,6,17,768 时序编码
            encoder_res = encoder(output)
            # cls_tokens
            # bs,6,768
            video_level_feature = encoder_res[:,:,0,:]
            # Mlp
            # bs,6
            predicts = mlp(video_level_feature).squeeze(-1)
            predicts = predicts.mean(1)

            end = time.time()
            batch_time = end - start
            if i % args.print_freq == 0:
                print('[Testing][%d/%d][%d/%d] \t Batch_time %.2f \t'
                      % (epoch, args.max_epoch, i, len(test_loader),
                         batch_time))
            # predicts = torch.round(predicts,decimals=3).data.cpu().numpy()
            # predicts = predicts * 1000
            # predicts = torch.round(predicts)
            # predicts = predicts / 1000
            predicts = predicts.data.cpu().numpy()
            raw_preds.extend(pred_scores)
            diffs.extend(diff)
            # pred_scores.extend([round(i[0].item() * i[1].item(),3) for i in zip(predicts, diff)])
            pred_scores.extend([i[0].item() * i[1].item() for i in zip(predicts, diff)])
    return batch_loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
