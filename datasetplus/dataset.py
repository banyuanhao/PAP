from torchvision.transforms import transforms
from torchvision import transforms, datasets
from datasetplus.food import Food101
from datasetplus.cars import StanfordCars
from datasetplus.DTD import DTD
from datasetplus.pets import OxfordIIITPet
from datasetplus.FGVC import FGVCAircraft
from datasetplus.cub2011 import Cub2011
import os


class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_train_dataset(self, name,date_aug):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'cifar100': lambda: datasets.CIFAR100(self.root_folder,train=True,
                                                                transform=self.Transform_without_aug() if date_aug else self.Transform_without_aug()),
                          'stl10': lambda: datasets.STL10(self.root_folder,split='train',
                                                          transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'caltech101': lambda : datasets.Caltech101(self.root_folder,
                                                          transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'imagenet':lambda: datasets.Imagenet(root=self.root_folder,split='train', transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'food': lambda : Food101(self.root_folder,split='train',
                                                          transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'cars': lambda: StanfordCars(self.root_folder, split='train',
                                                  transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'DTD': lambda: DTD(self.root_folder, split='train',
                                                       transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'pets': lambda: OxfordIIITPet(self.root_folder, split='trainval',
                                             transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'svhn':lambda : datasets.SVHN(root=self.root_folder,split='train',
                                                    transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'fgvc':lambda :FGVCAircraft(root=self.root_folder,split='train',download=True,
                                                    transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          'cub':lambda :Cub2011(root=self.root_folder,train=True,
                                                    transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug()),
                          }
        dataset_fn = valid_datasets[name]
        return dataset_fn()

    def get_val_dataset(self, name):
        valid_datasets = {'food': lambda: Food101(self.root_folder, split='val',
                                                        transform=self.Transform_without_aug()),
                          'stl10': lambda: datasets.STL10(self.root_folder,split='val',
                                                          transform= self.Transform_without_aug() ),
                          'cars': lambda: StanfordCars(self.root_folder, split='val',
                                                       transform=self.Transform_without_aug()),
                          'imagenet':lambda: datasets.ImageFolder(self.root_folder,),
                          'DTD': lambda: DTD(self.root_folder, split='val',
                                                      transform=self.Transform_without_aug()),
                          'svhn': lambda: datasets.SVHN(root=self.root_folder, split='val',
                                                        transform=self.Transform_without_aug()),
                          'fgvc': lambda: FGVCAircraft(root=self.root_folder, split='val',
                                                        transform=self.Transform_without_aug()),
                          }
        dataset_fn = valid_datasets[name]
        return dataset_fn()

    def get_test_dataset(self, name,date_aug = False):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=self.Transform_without_aug()),
                          'stl10': lambda: datasets.STL10(self.root_folder,split='test',
                                                          transform= self.Transform_without_aug()),
                          'cifar100': lambda: datasets.CIFAR100(self.root_folder,  train=False,
                                                                transform=self.Transform_without_aug()),
                          'imagenet':lambda: datasets.ImageNet(self.root_folder,split='test',
                                                               transform=self.Transform_with_aug() if date_aug else self.Transform_without_aug(),),
                          'food': lambda: Food101(self.root_folder, split='test',
                                                  transform=self.Transform_without_aug()),
                          'cars': lambda :StanfordCars(self.root_folder, split='test',download = True,
                                                  transform= self.Transform_without_aug()),
                          'DTD': lambda: DTD(self.root_folder, split='test',
                                                      transform=self.Transform_without_aug()),
                          'pets': lambda: OxfordIIITPet(self.root_folder, split='test',
                                                     transform=self.Transform_without_aug()),
                          'svhn': lambda: datasets.SVHN(root=self.root_folder, split='test',
                                                        transform=self.Transform_without_aug()
                                                        ),
                          'fgvc': lambda: FGVCAircraft(root=self.root_folder, split='test',
                                                       transform=self.Transform_without_aug()),
                          'cub': lambda: Cub2011(root=self.root_folder, train=False,
                                                 transform= self.Transform_without_aug()),

                          }
        dataset_fn = valid_datasets[name]
        return dataset_fn()


    @staticmethod
    def Transform_with_aug():
        return transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])

    @staticmethod
    def Transform_without_aug():
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    @staticmethod
    def Transform_without_aug_imagenet():
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    @staticmethod
    def Transform_with_aug_imagenet():
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
