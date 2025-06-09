import torch
import os
def save_checkpoint(model, encoder, fuser, decoder, rater, mlp, optimizer, epoch, epoch_best, epoch_best_RL2, epoch_best_L2, rho_best, L2_min, RL2_min, args):
    root_path = './models_checkpoint'
    save_path = os.path.join(root_path,'fd_seg64_4')
    torch.save({
                'model' : model.state_dict(),
                'encoder' : encoder.state_dict(),
                'fuser': fuser.state_dict(),
                'decoder': decoder.state_dict(),
                'rater': rater.state_dict(),
                'mlp' : mlp.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'epoch_best': epoch_best,
                'epoch_best_RL2': epoch_best_RL2,
                'epoch_best_L2': epoch_best_L2,
                'rho_best' : rho_best,
                'L2_min' : L2_min,
                'RL2_min' : RL2_min,
                }, os.path.join( save_path, str(epoch) + '.pth'))