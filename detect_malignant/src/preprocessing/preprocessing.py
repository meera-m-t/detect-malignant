import cv2
import numpy as np
import torchvision.transforms as T
from albumentations import (Compose, Flip, GaussNoise,  RandomRotate90, ShiftScaleRotate,
                            Transpose)
from albumentations.augmentations.transforms import ColorJitter
from PIL import Image



class CustomCombinedTransform:
    def __init__(
        self,
        config,
        imsize,     
        mode,
        p,
        setting,
    
    ):
        self.config = config
        self.imsize = imsize
        self.mode = mode
        self.p = p
        self.setting = setting
      
        self.alb_transform_train = self.alb_transform_train(self.p, self.setting)        
        self.tv_transform = self.tv_transform(self.imsize)


    @staticmethod
    def alb_transform_train(p=1, setting=0):
        if setting is None:
            setting = 0

        if setting == 0:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.01),
                    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                ],
                p=1,
            )
        if setting == 1:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                ],
                p=1,
            )
        elif setting == 2:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                ],
                p=1,
            )
        elif setting == 3:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                ],
                p=1,
            )
        elif setting == 4:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    ColorJitter(brightness=0.15, contrast=0, saturation=0.1, hue=0.025, p=0.7),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                ],
                p=1,
            )
        elif setting == 5:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    Flip(),
                    Transpose(),
                    ColorJitter(brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                ],
                p=1,
            )
        elif setting == 30:
            albumentations_transform = Compose(
                [
                    ColorJitter(brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7),
                ],
                p=1,
            )

        # Apply the transformation on the image
        def apply_transform(image):
            transformed = albumentations_transform(image=image)
            return transformed["image"]

        return apply_transform




    def tv_transform(self, imsize=224):
        tv_transform = T.Compose(
            [
                T.ToPILImage(),  
                T.Resize((imsize, imsize)),           
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        return tv_transform

