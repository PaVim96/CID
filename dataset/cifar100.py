from __future__ import print_function

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import dataset.spurious_feature as sf
from copy import deepcopy



def get_data_folder():
    
    data_folder = './data'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class BinaryCIFAR(datasets.CIFAR10):
    #CIFAR10 consists of: 
    #[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    #=> transportation means
    #[1, 1, 0, 0, 0, 0, 0, 1, 1, 1] => 50% of images per class
    def __init__(self, **kwargs):
        currenct_kwargs = deepcopy(kwargs)
        self.is_instance = currenct_kwargs.pop("is_instance")
        spurious_args = currenct_kwargs.pop("spurious_args")
        super().__init__(**currenct_kwargs)
        self.tenway_to_binary_classes()
        spurious_args["is_train"] = self.train
        self.data, self.sampler_cls = sf.apply_spurious(self.data, spurious_args)

    def tenway_to_binary_classes(self): 
        self.classes = ["transportation", "no-transportation"]
        self.class_to_idx = {
            "transportation": [0,1,7,8,9], 
            "no-transportation": [2,3,4,5, 6]
        }
        for i in range(len(self.targets)): 
            if self.targets[i] in [0,1,7,8,9]: 
                self.targets[i] = 0 
            else: 
                self.targets[i] = 1


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        """ else:
            img, target = self.test_data[index], self.test_labels[index] """

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_instance:
            return img, target, index
        else: 
            return img, target

            

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __init__(self, **kwargs): 
        self.is_instance = kwargs.pop("is_instance")
        super().__init__(**kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        """ else:
            img, target = self.test_data[index], self.test_labels[index] """

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_instance:
            return img, target, index
        else: 
            return img, target


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __init__(self, **kwargs): 
        self.is_instance = kwargs.pop("is_instance")
        super().__init__(**kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        """ else:
            img, target = self.test_data[index], self.test_labels[index] """

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_instance:
            return img, target, index
        else: 
            return img, target


def get_cifar_dataloaders(num_classes, batch_size=128, num_workers=8, is_instance=False, is_binary = False, spurious_args = None):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if num_classes == 100: 
        if not is_binary:
            cls_inst = CIFAR100Instance
        else: 
            raise ValueError("CIFAR100 not supported yet")
    elif num_classes == 10: 
        if not is_binary: 
            cls_inst = CIFAR10Instance
        else: 
            cls_inst = BinaryCIFAR


    train_set = cls_inst(
        root = data_folder, 
        download = True, 
        train = True, 
        transform = train_transform, 
        is_instance = is_instance
    )

    test_set = cls_inst(
        root=data_folder,
        train=False, 
        download=True, 
        transform = test_transform, 
        is_instance = is_instance
    )

    n_data = len(train_set)
        

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)


    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader