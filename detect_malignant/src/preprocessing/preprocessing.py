import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from albumentations import (Compose, Flip, GaussNoise,  RandomRotate90, ShiftScaleRotate,
                            Transpose)
from albumentations.augmentations.transforms import ColorJitter


class RemoveHair:
    def __call__(self, image):
        # Convert PIL Image to NumPy array
        image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define the kernel for the blackhat operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Apply blackhat operation
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold the blackhat image
        _, mask1 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out small contours
        for contour in contours:
            if cv2.contourArea(contour) < 70:  
                cv2.drawContours(mask1, [contour], -1, 0, -1)
        
        # Create the inverse mask
        inverse_mask1 = cv2.bitwise_not(mask1)
        
        # Apply the mask to the original image
        result_image = cv2.bitwise_and(image, image, mask=inverse_mask1)
        
        # Inpaint the masked areas
        inpainted_image = cv2.inpaint(result_image, mask1, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        
        # Convert back to PIL Image
        inpainted_image = Image.fromarray(inpainted_image)
        
        return inpainted_image

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
        self.tv_transform = self.tv_transform()
        self.tv_remove_hair = self.tv_remove_hair(self.imsize)


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
                    ColorJitter(brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7),
                ],
                p=1,
            )

        # Apply the transformation on the image
        def apply_transform(image):
            transformed = albumentations_transform(image=image)
            return transformed["image"]

        return apply_transform

    def tv_remove_hair(self, imsize=224):
        tv_transform = T.Compose([
                RemoveHair(),                 
                T.Resize((imsize, imsize)),            
            ]
        )
        return tv_transform
    
    def tv_transform(self):
        tv_transform = T.Compose(
            [
                T.ToPILImage(),          
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        return tv_transform

