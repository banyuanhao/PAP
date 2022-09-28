from resnet import name_to_params, get_resnet
import torch
import timm
import models_vit
assert timm.__version__ == "0.3.2" # version check
import torch
from util.pos_embed import interpolate_pos_embed
from pathlib import Path
import os
data_dict = {'cars':196,'pets':37,'food':101,'DTD':47,'cifar10':10,'cifar100':100,'fgvc':100,'cub':200,'svhn':10,'stl10':10,}

model_path = {
    'r50_1x_sk1':'finetuning/pretrained/r50_1x_sk1.pth',
    'r101_1x_sk1':'finetuning/pretrained/r101_1x_sk1.pth',
    'vit_base_patch16':'finetuning/pretrained/vit_base_patch16.pth'
}

finetuned_models_RES101={
    'cifar100':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/cifar100.pth',
    'cifar10':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/cifar10.pth',
    'pets':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/pets.pth',
    'svhn':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/svhn.pth',
    'fgvc':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/fgvc.pth',
    'food':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/food.pth',
    'DTD':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/DTD.pth',
    'cub':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/cub.pth',
    'stl10':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/stl10.pth',
    'cars':'/data/yuanhao.ban/finetunedmodels/SimCLRv2101/normalize/cars.pth',
}

finetuned_models_RES50 = {
    'cifar100':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/cifar100.pth',
    'cifar10':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/cifar10.pth',
    'pets':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/pets.pth',
    'svhn':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/svhn.pth',
    'fgvc':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/fgvc.pth',
    'food':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/food.pth',
    'DTD':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/DTD.pth',
    'cub':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/cub.pth',
    'stl10':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/stl10.pth',
    'cars':'/data/yuanhao.ban/finetunedmodels/SimCLRv250/normalize/cars.pth',
}


def get_pretrained_imagenet_simclr(model_name):
    pre_model,_ = get_resnet(*name_to_params(model_name))
    pre_model.load_state_dict(torch.load(model_path[model_name])['resnet'])
    return pre_model

def get_finetuned_models_simclr(model_name,dataset_name):
    pre_model,_ = get_resnet(*name_to_params(model_name),data_dict[dataset_name])

    if  model_name =='r50_1x_sk1':
        a = torch.load(finetuned_models_RES50[dataset_name])['resnet']
        b = torch.load(finetuned_models_RES50[dataset_name])['linearprob']
        a['fc.weight'] =b['linear.weight']
        a['fc.bias'] =b['linear.bias']
        msg = pre_model.load_state_dict(a)
        print(msg)
        
    elif model_name =='r101_1x_sk1':
        a = torch.load(finetuned_models_RES101[dataset_name])['resnet']
    else:
        print('not complete')
        exit(1)

    msg = pre_model.load_state_dict(a)
    return pre_model


finetuned_models_MAE_B16={
    'cifar100':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/cifar100.pth',
    'cifar10':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/cifar10.pth',
    'pets':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/pets.pth',
    'svhn':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/svhn',
    'fgvc':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/fgvc',
    'food':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/food.pth',
    'DTD':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/DTD.pth',
    'cub':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/cub.pth',
    'stl10':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/stl10.pth',
    'cars':'/data/yuanhao.ban/finetunedmodels/MAE/normalize/cars.pth',
}

def get_pretrained_imagenet_vit(model_name):

    model = models_vit.__dict__[model_name](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=False,
    )

    checkpoint = torch.load(model_path[model_name], map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % model_path[model_name])
    checkpoint_model = checkpoint['model']

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model.to('cuda')

def get_finetuned_models_mae(model_name,dataset_name):

    model = models_vit.__dict__[model_name](
    num_classes=data_dict[dataset_name],
    drop_path_rate=0.1,
    global_pool=0.25,
    )

    if  model_name == 'vit_base_patch16':
        checkpoint = torch.load(finetuned_models_MAE_B16[dataset_name])
        print("Load pre-trained checkpoint from: %s" % finetuned_models_MAE_B16[dataset_name])
    else:
        print('not complete')
        exit(1)

    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to('cuda').eval()
    return model

def get_base_path(args):
    base = Path('perturbations',args.model_arch,args.model_name,args.mode)

    if args.mode =='fff':
        base = base/args.prior_type/args.save_path
    else:
        base = base/args.save_path

    if not os.path.isdir(base):
        os.makedirs(base)

    return base

