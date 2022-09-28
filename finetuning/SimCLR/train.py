import torch
import sys
from datasetplus.dataset import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import argparse
from resnet import get_resnet,name_to_params
from pathlib import Path
import os

model_path = {
    'r50_1x_sk1':'/home/yuanhao.ban/contrastive/SimCLRv2/finetunedmodels/PAP/finetuning/pretrained/r50_1x_sk1.pth',
    'r101_1x_sk1':'',
}

data_dict = {'cifar100':100,'cifar10':10,'imagenet':1000,'flowers':102,'cars':196,'food':101,'DTD':47,'pets':37,'stl10':10,'svhn':10,'cub':200,'fgvc':100}

parser = argparse.ArgumentParser(description='PyTorch standard fintuning')

parser.add_argument('--dataset', type=str,
                    choices=['stl10', 'cifar10', 'imagenet','cifar100','food','cars','DTD','pets','svhn','fgvc','cub'])

parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr',type=float,default=5e-4)
parser.add_argument('--weight_decay',type=float,default=3e-5)
parser.add_argument('--epochs',type=int,default=300)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--momentum', type=str, default=0.9)
parser.add_argument('--save_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--data_path', type=str, default='/data/yuanhao.ban/PAP')

def run(args):

    print(args)
    path = Path(sys.path[0],'models',args.model_name,args.dataset, args.save_path)
    if not os.path.isdir(path):
        os.makedirs(path)

    device = args.device

    best_acc = 0.5

    data = Dataset(args.data_path)
    dataset_train = data.get_train_dataset(args.dataset,date_aug = True)
    dataset_test = data.get_test_dataset(args.dataset)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)

    print("=> creating model '{}'".format(args.model_name))
    model, _ = get_resnet(*name_to_params(args.model_name),data_dict[args.dataset])


    a = torch.load(model_path[args.model_name])['resnet']
    del a['fc.bias']
    del a['fc.weight']
    msg = model.load_state_dict(a,strict= False)
    print(msg)

    model = model.to(device).eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay,nesterov=True,momentum=args.momentum)

#training
    for i in range(args.epochs):
        model.train()
        pos = 0
        total = 0
        loss_sum = 0
        for images, labels in tqdm(data_loader_train,disable=args.disable_tqdm):
            images = images.to(device)
            output = model(images.to(device))
            _, pred = torch.max(output,dim=1)
            loss = criterion(output,labels.to(device))
            loss_sum = loss_sum*0.5 + loss*0.5
            pos += torch.sum((pred==labels.to(device)))
            total = total + len(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {i}, loss : {float(loss_sum)} , acc : {float(pos)/float(total)}")
        sys.stdout.flush()

        #testing
        pos = 0
        total = 0
        model.eval()
        for images, labels in tqdm(data_loader_test, disable=args.disable_tqdm):
            with torch.no_grad():
                output = model(images.to(device))
                _, pred = torch.max(output,dim=1)
                pos += torch.sum((pred==labels.to(device)))
                total = total + len(labels)

        acc = float(pos)/float(total)

        if acc > best_acc:
            best_acc = acc
            torch.save({'resnet': model.state_dict()}, path/f"i_{i}.pth")

        print(f"testing :, acc : {acc}")
        print(f"best_acc : {best_acc}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
