from typing import Any
import torch 
import torchvision.transforms as tf
import random
from enum import Enum
import numpy as np
from copy import deepcopy

class SamplingTypes(Enum):
    RANDOM = 0 
    INVERTED = 1


def batch_to_grayscale(imgs : np.ndarray, idxs: list):
    if idxs is None: 
        return imgs
    #https://stackoverflow.com/questions/63668883/converting-multiple-numpy-images-to-gray-scale
    rgb_vals = [0.2989, 0.5870, 0.1140]
    copy_imgs = deepcopy(imgs)
    print("-----------")
    print("SHAPES:")
    print(imgs.shape)
    print(copy_imgs.shape)
    assert len(copy_imgs.shape) == 4, "wrong shape"
    to_change = copy_imgs[idxs]
    print(to_change.shape)
    to_change = np.dot(to_change[..., :3], rgb_vals)
    to_change = to_change[:, :, :, np.newaxis]
    to_change = np.repeat(to_change, 3, axis = 3)
    copy_imgs[idxs] = to_change
    return copy_imgs

class SpuriousSampling(): 
    def __init__(self, *args):
        self.has_applied = False
        self.num_to_inject = 0 
        self.idxs_to_inject = None

    def apply(self):
        raise NotImplementedError("Not implemented in abstract class")
    
    def number_of_injected(self): 
        return self.num_to_inject
    
    def get_injected_ids(self): 
        return self.idxs_to_inject

class InvertedSampling(SpuriousSampling): 
    def __init__(self, *args):
        super().__init__()


    def apply(self):
        pass
    
    def number_of_injected(self):
        return self.num_to_injecet
    
    def get_injected_ids(self):
        return self._idxs_to_inject


    
class IdentitySampling(SpuriousSampling): 
    def __init__(self, *args):
        super().__init__()


    def apply(self):
        pass
    
    def number_of_injected(self):
        return self.num_to_injecet
    
    def get_injected_ids(self):
        return self._idxs_to_inject
    

class RandomSampling(SpuriousSampling): 
    def __init__(self, skew_ratio: float, dataset: torch.Tensor, is_train : bool): 
        super().__init__()
        self.is_train = is_train
        self.dataset = dataset
        assert isinstance(self.dataset, np.ndarray) and len(self.dataset.shape) == 4
        self.dataset = dataset
        if self.is_train:
            self.num_to_inject = int(len(dataset) * skew_ratio)
            self.idxs_to_inject = random.sample(range(len(self.dataset)), self.num_to_inject)

        print(f"Spurious for: {self.is_train}")
        if self.idxs_to_inject is None: 
            print("No idxs to inject")
        print(self.num_to_inject)
        if self.num_to_inject == 0: 
            assert self.idxs_to_inject is None, "cannot be 0 injected and existing idxs"

    def apply(self):
        if not self.is_train or self.has_applied: 
            return self.dataset
        else: 
            return batch_to_grayscale(self.dataset, self.get_injected_ids())




def apply_spurious(dataset: torch.Tensor, spurious_args: dict):
    if spurious_args is None: 
        return IdentitySampling(dataset)
    else: 
        assert "type" in spurious_args, "need to specify type of sampling"
        type = spurious_args.pop("type")
        spurious_args["dataset"] = dataset
        if type == SamplingTypes.RANDOM:
            sampler_cls = RandomSampling(**spurious_args)
        elif type == SamplingTypes.INVERTED: 
            sampler_cls = InvertedSampling(**spurious_args)
        else: 
            raise NotImplementedError("not implemented yet")
        
        data = sampler_cls.apply()
        return data, sampler_cls
        
    


