## Pre-trained Adversarial Perturbations
This is a repo for finetuning SimCLR models.

### Enviroment setup
```
Pytorch 1.8.1
Torchvision
tqdm
```

### Preparation
#### Pretrained models
The repo need the pretrained models and the finetuned ones. 
To generated the adversarial samples, please use this [repo](https://github.com/Separius/SimCLRv2-Pytorch) to convert the tensorflow models provided [here](https://github.com/google-research/simclr) into Pytorch ones and configure the model_path in [tools.py](tools.py).
To test the adversarial samples, please refer to the [repo](finetuning/SimCLR/README.md) 
#### Datasets
Please download the [Imagenet](https://image-net.org/index.php), [CARS](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [PETS](https://www.robots.ox.ac.uk/~vgg/data/pets/), [FOOD](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [FGVC](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SVHN](http://ufldl.stanford.edu/housenumbers/), [STL10](https://cs.stanford.edu/~acoates/stl10/) datasets and put them in a folder. 

### About AME
We adopt the official code to finetune the MAE ViTs. Anyone needing finetuned MAE models can refer to the [repo](https://github.com/facebookresearch/mae).

### Acknowledgements
* This repo is a based on the [SimCLRv2 repo](https://github.com/google-research/simclr).

* Thanks to [SimCLRv2-Pytorch repo](https://github.com/Separius/SimCLRv2-Pytorch), which provides a way to convert the pre-trained SimCLRv2 Tensorflow checkpoints into Pytorch ones.