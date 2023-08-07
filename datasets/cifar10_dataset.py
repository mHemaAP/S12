import numpy as np
import torch
import torchvision
import cv2
import albumentations as A

from .abstract_dataset import dataSet

torch.manual_seed(11)

#### Class to transform CIFAR10 Data Set with Albumentations
class albumentationTransforms(torchvision.datasets.CIFAR10):
    def __init__(self, root, data_alb_transform=None, **kwargs):
        super(albumentationTransforms, self).__init__(root, **kwargs)
        self.data_alb_transform = data_alb_transform

    def __getitem__(self, index):
        trans_image, trans_label = super(albumentationTransforms, self).__getitem__(index)
        if self.data_alb_transform is not None:
            trans_image = self.data_alb_transform(image=np.array(trans_image))['image']
        return trans_image, trans_label
    

#### CIFAR10 Data Set Class
class cifar10Set(dataSet):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    dataSet_transformation = albumentationTransforms

    train_idata_transforms_default = [

            A.ColorJitter(brightness=0, contrast=0.1, 
                            saturation=0.2, hue=0.1, p=0.5),
            A.ToGray(p=0.2),
            A.PadIfNeeded(40, 40, p=1),
            A.RandomCrop(32, 32, p=1),

            # IAAFliplr is deprecated, hence using Horizontal Flip instead                                
            
            A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
            #                    rotate_limit=15),
            # A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            # Since normalisation was the first step, mean is already 0, 
            # so cutout fill = 0
            A.CoarseDropout(max_holes=1, 
                            max_height=8, 
                            max_width=8, p=1, 
                            fill_value=0),
            #A.CenterCrop(32, 32, p=1)
        ]