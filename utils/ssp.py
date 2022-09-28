import torch
from datasetplus.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools import get_base_path

def ssp(model, args =None):

    target_layer = 2 if args.model_arch == 'simclr' else 5
    device = args.device

    path = get_base_path(args)
    model.to(device).eval()

    bound = args.bound
    alpha = args.alpha

    dataset = Dataset(args.data_path)
    training_dataset = dataset.get_train_dataset('imagenet',date_aug=False)
    train_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)

    delta = bound * torch.rand((1,3, 224, 224), device=device, requires_grad=True) * 2 - bound
    k = 0
    cri = torch.nn.MSELoss()

    for (image,_) in tqdm(train_loader, disable=args.disable_tqdm):
        delta = delta.detach()
        delta.requires_grad = True
        k = k + 1

        image = image.to(device)
        image_cpoy = image.clone().detach().to(device)
        
        activations = []
        remove_handles = []
        
        def activation_recorder_hook(self, input, output):
            activations.append(torch.square(output))
            return None
        
        if args.model_arch == 'simclr':
            handle = model.net[target_layer].register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)
        else:
            handle = model.blocks[target_layer].register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)
        
        with torch.no_grad():
            model(image)
        clean_layer = activations[0].clone().detach().to(device).requires_grad_(True)
        
        for handle in remove_handles:
            handle.remove()
            
        activations = []
        remove_handles = []
        handle = model.net[target_layer].register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)
        
        model(image_cpoy + delta)
        
        for handle in remove_handles:
            handle.remove()

        loss = cri(clean_layer,activations[0])
        loss.backward()
        
        delta = delta + alpha * delta.grad.sign()
        delta.clamp_(-bound, bound)

        if k % args.save_every_iter == 0:
            print(f'iters : {k} \n')
            filename = path/f'k={k}.pt'
            torch.save(delta, filename)

