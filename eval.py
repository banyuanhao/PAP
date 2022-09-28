from datasetplus.dataset import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from tools import get_finetuned_models_mae, get_finetuned_models_simclr, get_pretrained_imagenet_simclr
import torch

data_dict = {'cars':196,'pets':37,'food':101,'DTD':47,'cifar10':10,'cifar100':100,'fgvc':100,'cub':200,'svhn':10,'stl10':10,}
parser = argparse.ArgumentParser(description='Evalation')

parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--uap_path',type=str,default=None)
parser.add_argument('--model_name',type=str)
parser.add_argument('--model_arch',type=str)
parser.add_argument('--data_path',type=str,default='/data/yuanhao.ban/PAP')


def get_fooling_rate(delta, dataset_name, args):

    device = args.device


    if args.model_arch == 'simclr':
        model = get_finetuned_models_simclr(args.model_name,dataset_name)
    elif args.model_arch == 'mae':
        model = get_finetuned_models_mae(args.model_name,dataset_name)
    else:
        print('not supported')
        exit(1)

    model.to(device).eval()

    _dataset = Dataset('/data/yuanhao.ban/PAP')
    dataset = _dataset.get_test_dataset(dataset_name)
    dataloader = DataLoader(dataset, num_workers=4, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for (image, label) in tqdm(dataloader,disable=args.disable_tqdm):
            #print(image)
            image = image.to(device) +delta
            label = label.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            #print(pred,label)
            correct += torch.sum(pred==label)
            total += label.shape[0]

    print(f'dataset : {dataset_name}')
    print(f'attack sucess rate {1 - float(correct)/float(total)}')

def get_fooling_rate_imagenet(model_name, delta, args):
    device = args.device
    model = get_pretrained_imagenet_simclr(model_name)

    dataset = Dataset('/data/yuanhao.ban/val')
    testing_dataset = dataset.get_test_dataset('imagenet')
    test_loader = DataLoader(
        testing_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True)
    total = 0
    acc = 0
    model.to(device).eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            labels = labels.to(device)
            images = images.to(device)
            adv_images = torch.add(delta, images).clamp(0, 1)
            adv_outputs = model(adv_images)
            _, adv_predicted = torch.max(adv_outputs, 1)
            acc += (adv_predicted == labels).sum().item()
            total += images.size(0)

    print(f'modelname : {model_name} ')
    print(f'dataset : imagenet ')
    print(f'attack sucess rate {1 - float(acc)/float(total)}')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.uap_path:
        print(f'valuing uap in {args.uap_path}\n')
        delta = torch.load(args.uap_path,map_location=args.device)
    else:
        print('clean samples\n')
        delta = torch.zeros(1,3,224,224,device=args.device)

    for keys in data_dict.keys():
        get_fooling_rate(delta, keys, args)   