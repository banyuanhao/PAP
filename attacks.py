import argparse
from utils.uap import uap
from utils.fff import fff
from utils.std import std
from utils.L4A_fuse import L4A_fuse
from utils.L4A_base import L4A_base
from utils.L4A_ugs import L4A_ugs
from utils.uapepgd import uapepgd
from utils.ssp import ssp
from tools import get_pretrained_imagenet_simclr,get_pretrained_imagenet_vit

parser = argparse.ArgumentParser(description='PAP training')

# dataset
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--data_path', type=str, default='/data/yangdingcheng/ILSVRC2012/train')
parser.add_argument('--workers', type=int, default=8)

# model
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_arch', type=str,choices=['mae','simclr'])
parser.add_argument('--model_name', type=str, default=['r50_1x_sk1','r101_1x_sk1','vit_base_patch16'])

# basic config
parser.add_argument('--mode', type=str,choices=['fff','std','uap','uapepgd','ssp','L4A_base','L4A_fuse','L4A_ugs'])
parser.add_argument('--bound', type= float,default=0.05)
parser.add_argument('--disable_tqdm', action='store_true')                                                      
parser.add_argument('--save_every_iter',type=int,default=1000)
parser.add_argument('--save_path',type=str)
parser.add_argument('--alpha',type=float,default=0.0002)

#fff
parser.add_argument('--prior_type', choices=['no_data', 'mean_std', 'one_sample'],
                    help='Which kind of prior to use')

#L4A_base
parser.add_argument('--target_layer',type=int)


parser.add_argument('--lamuda', type=float)

#L4A_fuse
parser.add_argument('--target_layer1', type=int)
parser.add_argument('--target_layer2', type=int)

#L4A_ugs
parser.add_argument('--mean_std',type=str,choices=['uniform', 'imagenet'])
parser.add_argument('--std_lo', type=float, default=0.05)
parser.add_argument('--std_hi', type=float, default=0.1)
parser.add_argument('--mean_lo', type=float, default=0.40)
parser.add_argument('--mean_hi', type=float, default=0.60)



def main():
    args = parser.parse_args()
    mode = args.mode

    if args.model_arch == 'simclr':
        model = get_pretrained_imagenet_simclr(args.model_name)
        model.to(args.device).eval()
    elif args.model_arch == 'mae':
        model = get_pretrained_imagenet_vit(args.model_name)
        model.to(args.device).eval()
    else:
        print('not supported')
        exit(1)

    print(args)
        
    if mode == 'fff':
        fff(model=model,args=args)
    elif mode =='uap':
        uap(model=model,args=args)
    elif mode == 'std':
        std(model=model,args=args)
    elif mode == 'L4A_base':
        L4A_base(model=model,args=args)
    elif mode == 'L4A_fuse':
        L4A_fuse(model=model,args=args)
    elif mode == 'uap':
        uap(model=model,args=args)
    elif mode == 'L4A_ugs':
        L4A_ugs(model=model, args = args)
    elif mode == 'uapepgd':
        uapepgd(model=model,args=args)
    elif mode == 'ssp':
        ssp(model=model,args=args)
    else:
        print('not complete')
        exit(1)

if __name__ == "__main__":
    main()
