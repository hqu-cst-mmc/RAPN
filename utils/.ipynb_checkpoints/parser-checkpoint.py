import os
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--benchmark', type = str, choices=['MTL', 'Seven'], help = 'dataset')
    # parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--benchmark', type = str, default='FineDiving', help = 'dataset MTL|FineDiving')
    parser.add_argument('--exp_name', type = str, default='try', help = 'experiment name')
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--resume', action='store_true', default=True ,help = 'autoresume training from exp dir(interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--Seven_cls', type = int, default=1, choices=[1,2,3,4,5,6], help = 'class idx in Seven')
    parser.add_argument('--lr_steps', default=[30, 60], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--segment_frame_num', type = int, default=16, help='segment avg. frame number')
    parser.add_argument('--sub_video_num', type = int, default=3, help='subvideo number')
    parser.add_argument('--segments_num', type = int, default=3, help='segment number')
    parser.add_argument('--frame_select_strategy', type = str, default="odd_even_sampling", help = 'frame select strategy')
    parser.add_argument('--backbone', type = str, default="TSN", help = 'backbone')
    parser.add_argument('--frame_HW', type = int, default=224, help='frame`s size,H and W')
    parser.add_argument('--attention_type', type=str, default="space_only", help='backbone')
    parser.add_argument('--pretrained_model', type=str,
                        default="./weight/TimeSformer_divST_96x4_224_K600.pyth",
                        help='weight')
    parser.add_argument('--all_frames', type = int, default=48, help='视频帧采样策略，96帧视频采样n帧(n可取16,32,48,64,96)')

    # 多卡
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    if args.benchmark == 'Seven':
        print(f'Using CLASS idx {args.Seven_cls}')
        args.class_idx = args.Seven_cls
    return args

def setup(args):
    args.config = 'configs/{}_CoRe.yaml'.format(args.benchmark)
    args.experiment_path = os.path.join('./experiments', 'CoRe_RT', args.benchmark, args.exp_name)
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print('Resume yaml from %s' % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)

def get_config(args):
    print('Load config yaml from %s' % args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    
def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path,'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)