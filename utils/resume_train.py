import torch
import os
def resume_train(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, args):
    # 复现轮次
    ckp_type = 'best_Tau'
    # 权重文件地址
    root_path = f'./models_checkpoint/FineSync_{args.event}(0.7-0.3)_{args.suffix}'
    ckpt_path = os.path.join(root_path, ckp_type + '.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        exit(0)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    # 读取权重文件字典
    state_dict = torch.load(ckpt_path,map_location='cpu')

    # 读取model,encoder,mlp三个模型的权重
    # 1-parameter resume of base model
    # model_ckpt = {k.replace("module.model.", ""): v for k, v in state_dict['model'].items()}
    model_ckpt = {k:v for k, v in state_dict['model'].items()}
    model.load_state_dict(model_ckpt)
    # 2-parameter resume of encoder
    encoder_ckpt = {k:v for k, v in state_dict['encoder'].items()}
    encoder.load_state_dict(encoder_ckpt)
    # 3-parameter resume of decoder
    fuser_ckpt = {k: v for k, v in state_dict['fuser'].items()}
    fuser.load_state_dict(fuser_ckpt)
    # 3-parameter resume of decoder
    evnet_ckpt = {k: v for k, v in state_dict['evnet'].items()}
    evnet.load_state_dict(evnet_ckpt)
    # 3-parameter resume of decoder
    decoder_ckpt = {k: v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)
    # 4-parameter resume of mlp
    mlp_ckpt = {k:v for k, v in state_dict['mlp'].items()}
    mlp.load_state_dict(mlp_ckpt)
    # 5-parameter resume of rater_exe
    rater_exe_ckpt = {k:v for k, v in state_dict['rater_exe'].items()}
    rater_exe.load_state_dict(rater_exe_ckpt)
    # 6-parameter resume of rater_sync
    rater_sync_ckpt = {k: v for k, v in state_dict['rater_sync'].items()}
    rater_sync.load_state_dict(rater_sync_ckpt)

    #5-optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best = state_dict['epoch_best']
    epoch_best_RL2 = state_dict['epoch_best_RL2']
    epoch_best_L2 = state_dict['epoch_best_L2']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    # start_epoch = 450

    return start_epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min
