import torch
import os
def resume_train(model, encoder, fuser, decoder, rater, mlp, optimizer, args):
    # 复现轮次
    resume_epoch = 888
    # 权重文件地址
    root_path = './models_checkpoint/fd_seg64_4'
    ckpt_path = os.path.join(root_path,str(resume_epoch) + '.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
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
    decoder_ckpt = {k: v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)
    # 4-parameter resume of mlp
    mlp_ckpt = {k:v for k, v in state_dict['mlp'].items()}
    mlp.load_state_dict(mlp_ckpt)
    # 5-parameter resume of rater
    rater_ckpt = {k:v for k, v in state_dict['rater'].items()}
    rater.load_state_dict(rater_ckpt)

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

    start_epoch = 450

    return start_epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min
