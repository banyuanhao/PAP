from datasetplus.dataset import Dataset
from tqdm import tqdm
from tools import get_base_path
from torch.utils.data import DataLoader
import torch
mean_std = {'imagenet':[(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)]}

def L4A_ugs(model, args):

    path = get_base_path(args)

    device = args.device
    bound = args.bound
    step_size = args.alpha

    model.to(device).eval()


    _dataset = Dataset(args.data_path)
    dataset = _dataset.get_train_dataset('imagenet', date_aug=False)
    dataloader = DataLoader(dataset, num_workers=args.workers, shuffle=True, batch_size=args.batch_size,
                            pin_memory=True)


    delta = torch.rand(1, 3, 224, 224, device=args.device) * 2 * bound - bound

    delta.requires_grad = True

    k = 0


    for (image, _) in tqdm(dataloader, disable=args.disable_tqdm):

        if k % 20 == 0 and args.mean_std == 'uniform':
            std = torch.rand(3)* (args.std_hi- args.std_lo) + args.std_lo
            mean = torch.rand(3)* (args.mean_hi- args.mean_lo) + args.mean_lo
        model.zero_grad()

        activations = []
        remove_handles = []
        
        def activation_recorder_hook(self, input, output):
            activations.append(torch.square(output))
            return None
        
        if args.model_arch == 'simclr':
            handle = model.net[args.target_layer].register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)
        else:
            handle = model.blocks[args.target_layer].register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)

        handle = model.net[args.target_layer].register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

        delta = delta.detach().requires_grad_()
        dataplus = torch.randn(8,3,224,224,device=device)

        if args.mean_std == 'imagenet':
            dataplus[:,0,:,:] = dataplus[:,0,:,:] * mean_std['imagenet'][1][0] + mean_std['imagenet'][0][0]
            dataplus[:,1,:,:] = dataplus[:,1,:,:] * mean_std['imagenet'][1][1] + mean_std['imagenet'][0][1]
            dataplus[:,2,:,:] = dataplus[:,2,:,:] * mean_std['imagenet'][1][2] + mean_std['imagenet'][0][2]
        elif args.mean_std == 'uniform':
            dataplus[:,0,:,:] = dataplus[:,0,:,:] * std[0] + mean[0]
            dataplus[:,1,:,:] = dataplus[:,1,:,:] * std[1] + mean[1]
            dataplus[:,2,:,:] = dataplus[:,2,:,:] * std[2] + mean[2]
        else :
            print('not complete')
            exit(1)


        dataplus.requires_grad = False
        image = image.to(device)
        
        data = torch.cat((dataplus,image),dim=0)
        
        model(data + delta)

        loss = torch.tensor(0., device=device)

        loss+= (torch.mean(activations[0][0:8])+\
            torch.mean(activations[0][8:-1])*args.lamuda)

        loss.backward()
        delta = delta + step_size * delta.grad.sign()
        delta.clamp_(-bound, bound)
        k = k + 1

        for handle in remove_handles:
            handle.remove()

        if k % args.save_every_iter == 0:
            print(f'iters : {k} \n')
            filename = path/f'k={k}.pt'
            torch.save(delta, filename)

