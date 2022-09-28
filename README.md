## Pre-trained Adversarial Perturbations
This is a repo for Pre-trained Adversarial Perturbations.

### Enviroment setup
```
Pytorch 1.8.1
Torchvision
tqdm
timm 0.3.2
```

### Preparation
The repo need the pretrained models and the finetuned ones.\
To get SimCLRv2 pretrained models, please use this [repo](https://github.com/Separius/SimCLRv2-Pytorch) to convert the tensorflow models provided [here](https://github.com/google-research/simclr) into Pytorch ones. Please download MAE pretrained models from [here](https://github.com/facebookresearch/mae).\
To get finetuned models and test the adversarial samples, please follow this [repo](finetuning/SimCLR/README.md)
Please configure the paths of pre-trained and finetuned models in [tools.py](tools.py).



### Attacking
We provide the testing code of sereval baselines [STD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Enhancing_Cross-Task_Black-Box_Transferability_of_Adversarial_Examples_With_Dispersion_Reduction_CVPR_2020_paper.pdf), [SSP](https://arxiv.org/abs/2006.04924), [FFF](https://arxiv.org/abs/1707.05572), [UAP](https://arxiv.org/abs/1610.08401), [UAPEPGD](https://ieeexplore.ieee.org/abstract/document/9191288/).
We omit [ASV](https://arxiv.org/abs/1709.03582) bacause it can hardly be integrated into the framework. Anyone interested in it may download the official code and have a try. 

Performing L4A<sub>base</sub> on Resnet50 pretrained by SimCLRv2. 
```
python attacks.py --mode l4a_base --model_name r50_1x_sk1 --model_arch simclr --data_path your_data_folder --target_layer 0 --save_path your_save_path
```

Performing L4A<sub>fuse</sub> on Resnet101 pretrained by SimCLRv2. 
```
python attacks.py --mode l4a_fuse --model_name r101_1x_sk1 --model_arch simclr --data_path your_data_folder --target_layer1 0 --target_layer2 1 --lamuda 1 --save_path your_save_path
```

Performing L4A<sub>ugs</sub> on ViT-B pretrained by MAE.
```
python attacks.py --mode l4a_ugs --model_name vit_base_patch16 --data_path your_data --mean_std uniform --mean_hi 0.6 --mean_lo 0.4 --std_hi 0.10 --std_lo 0.05 --lamuda 0.01 --save_path your_save_path
```

Performing SSP on Resnet50 pretrained by SimCLRv2. 
```
python attacks.py --mode ssp --model_name r50_1x_sk1 --model_arch simclr --data_path your_data_folder --save_path your_save_path
```

Note: if you want to perform UAP or UAPEPGD on MAE models, you have to obtain models that linearprobes on the Imagenet.
Please refer to the [MAE repo](https://github.com/facebookresearch/mae).

### Evaluating
Testing PAPs on Resnet101
```
python eval.py --model_name r50_1x_sk1 --model_arch simclr --uap_path your_pap_path
```

### models
We provide several finetuned models on the shelf. Please check the following table.

|  SimCLRv2   | SimCLRv2  | MAE |
|  ----  | ----  | ----  |
| r50_1x_sk1  | r101_1x_sk1 | vit_base_patch16 |
| [models]()  | [models]() | [models]() |

## Acknowledgements
* This repo is based on the [SimCLRv2 repo](https://github.com/google-research/simclr), [SimCLRv2-Pytorch repo](https://github.com/Separius/SimCLRv2-Pytorch) and [MAE repo](https://github.com/facebookresearch/mae).

* We use many parts of [UAP repo](https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch), [FFF repo](https://github.com/val-iisc/fast-feature-fool). Thanks a lot.