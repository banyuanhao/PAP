import torch
from datasetplus.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from tools import get_base_path

def uapepgd(model, args =None):

    path = get_base_path(args)
    device = args.device
    model.to(device).eval()

    bound = args.bound

    dataset = Dataset(args.data_path)
    training_dataset = dataset.get_train_dataset('imagenet', date_aug=False)
    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    delta = bound * torch.rand((1, 3, 224, 224), device=device, requires_grad=False) * 2 - bound
    k = 0

    for (image,_) in tqdm(train_loader, disable=args.disable_tqdm):
        k = k + 1
        
        image = image.to(device).reshape(1,3,224,224)
        
        with torch.no_grad():
            clean_pre = model(image)
            clean_pre = F.normalize(clean_pre, dim = 1)
            _, clean_label = torch.max(clean_pre, 1)    
            
            adv_pre = model(image+ delta)
            adv_pre = F.normalize(adv_pre, dim = 1)
            _, adv_label = torch.max(adv_pre, 1)
            
        if adv_label == clean_label:
            image_copy = image.clone().detach() + delta
            delta0 = get_delta(model, image_copy, clean_label, device, bound, alpha = 0.002 if k < 10000 else 0.0002) 
            delta = torch.clip(delta + delta0, -bound, bound).detach()
            

        if k % args.save_every_iter == 0:
            filename = path/f'k={k}.pt'
            torch.save(delta, filename)


def get_delta(model, image_cpoy, clean_label, device, bound, alpha):
    model.zero_grad()
    t = 0
    v = torch.zeros(1,3,224,224, device=device, requires_grad=False)
    beta = 0.5
    gamma = 1e-5
    delta0 = (bound * torch.rand((1, 3, 224, 224), device=device, requires_grad=True) * 2 - bound)
    delta0.retain_grad()
    cri = torch.nn.CrossEntropyLoss()
    while(t < 10):
        t = t + 1
        adv_pre = model(image_cpoy + delta0)
        adv_pre = F.normalize(adv_pre, dim = 1)
        _, adv_label = torch.max(adv_pre, 1)
        loss = cri(adv_pre, clean_label)
        loss.backward()
        with torch.no_grad():
            v = beta * v + gamma * (image_cpoy + delta0) + delta0.grad
            delta0 = delta0 + v.sign() * alpha
        delta0 = delta0.detach().requires_grad_(True)
        if clean_label != adv_label:
            break
        
    return delta0