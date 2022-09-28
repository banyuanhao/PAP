import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from tools import get_base_path
mean_std = {'imagenet':[(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)]}

def get_conv_layers(model):
    return [module for module in model.modules() if type(module) == torch.nn.Conv2d]

def get_rate_of_saturation(delta, xi):
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)


def l2_layer_loss(model, delta, device, prior_type, model_arch):
    loss = torch.tensor(0.,device=device)
    activations = []
    remove_handles = []

    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None


    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)


    model.eval()
    model.zero_grad()
    if prior_type == 'mean_std':
        dataplus = torch.randn(3,224,224,device=device)
        dataplus[0] = dataplus[0]*mean_std['imagenet'][1][0] + mean_std['imagenet'][0][0]
        dataplus[1] = dataplus[1]*mean_std['imagenet'][1][1] + mean_std['imagenet'][0][1]
        dataplus[2] = dataplus[2]*mean_std['imagenet'][1][2] + mean_std['imagenet'][0][2]
        dataplus.requires_grad = False
        model(torch.clip(delta+ dataplus,0,1))
    elif prior_type == 'one_sample':
        import PIL
        import torchvision.transforms as transforms
        image = PIL.Image.open("path_of_one_image.png")
        transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
        image = transform(image).to(device)
        model(torch.clip(delta+image,0,1))
    elif prior_type == 'no_data':
        model(delta)
    else:
        pass

    for handle in remove_handles:
        handle.remove()

    loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2), activations)))
    return loss


def fff(model,args = None):
    """
    Returns a universal adversarial perturbation tensor
    """
    path = get_base_path(args)
    device = args.device
    model.to(device).eval()

    bound = args.bound

    max_iter = 10000

    sat_threshold = 0.00001
    sat = 0
    sat_min = 0.5
    sat_should_rescale = False

    k = 0

    delta = 2 * bound *torch.rand((1, 3, 224, 224), device=device) - bound
    delta.requires_grad = True
    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=0.1)


    for i in tqdm(range(max_iter), disable=args.disable_tqdm):
        k += 1
        optimizer.zero_grad()
        loss = l2_layer_loss(model, delta, device, args.prior_type, args.model_arch)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i}, Loss: {loss.item()}")

        # clip delta after each step
        with torch.no_grad():
            delta.clamp_(-bound, bound)

        # compute rate of saturation on a clamped delta
        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), bound)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            sat_should_rescale = True


        if k % args.save_every_iter == 0:
            print(f'iters : {k} \n')
            filename = path/f'k={k}.pt'
            torch.save(delta, filename)

        if sat_should_rescale:
            with torch.no_grad():
                delta.data = delta.data / 2
            sat_should_rescale = False
        
    return delta