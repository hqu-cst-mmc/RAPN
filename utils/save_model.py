import torch
import os
import logging

def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger

def save_checkpoint(model, encoder, fuser, evnet, decoder, rater_exe, rater_sync, mlp, optimizer, epoch, ckp_type, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args):
    root_path = './models_checkpoint'
    save_path = os.path.join(root_path, f'FineSync_{args.event}(0.7-0.3)_{args.suffix}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
                'model' : model.state_dict(),
                'encoder' : encoder.state_dict(),
                'fuser': fuser.state_dict(),
                'evnet': evnet.state_dict(),
                'decoder': decoder.state_dict(),
                'rater_exe': rater_exe.state_dict(),
                'rater_sync': rater_sync.state_dict(),
                'mlp' : mlp.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'epoch_best': epoch_best,
                'epoch_best_RL2': epoch_best_RL2,
                'epoch_best_L2': epoch_best_L2,
                'rho_best' : rho_best,
                'L2_min' : L2_min,
                'RL2_min' : RL2_min,
                }, os.path.join(save_path, ckp_type + '.pth'))