import torch
from datasetplus.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients
from tools import get_base_path


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = net(Variable(image[ :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[ :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)
            #x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image

def uap(model, args =None):

    path = get_base_path(args)
    device = args.device
    model.to(device).eval()
    bound = args.bound

    dataset = Dataset(args.data_path)
    training_dataset = dataset.get_train_dataset('imagenet',date_aug=True)
    train_loader = DataLoader(training_dataset,batch_size=1,shuffle=True,num_workers=args.workers)

    delta = bound *torch.rand((3, 224, 224), device=device)*2 - bound
    k = 0


    for (image,_) in tqdm(train_loader,disable=args.disable_tqdm):
        image = image.to(device)
        clean_pre = model(image).argmax()
        adv_pre = model(image+delta).argmax()
        k = k + 1

        if clean_pre == adv_pre:
            pertub,_,_,_,_ = deepfool(image+delta,model)
            pertub = torch.from_numpy(pertub).to(device)
            delta = delta + pertub
            delta.clamp_(-bound, bound)

        if k % args.save_every_iter == 0:
            filename = path/f'k={k}.pt'
            torch.save(delta, filename)
