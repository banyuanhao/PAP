from datasetplus.dataset import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from tools import get_base_path

def L4A_base(model,args):

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

        delta = delta.detach().requires_grad_()
        image = image.to(device)
        model(image.to(device) + delta)

        loss = torch.tensor(0., device=device)

        loss+= torch.mean(activations[0])
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



