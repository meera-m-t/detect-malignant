import os
import cv2
import numpy as np
import pandas as pd
import torch
import time

from PIL import Image
from torch.utils.data import Dataset
from fastai.data.load import DataLoader as FastDataLoader
from torch.utils.data import  DistributedSampler
from torchsampler import ImbalancedDatasetSampler

from detect_malignant.src.preprocessing.preprocessing import CustomCombinedTransform


class MalignantDataset(Dataset):
    def __init__(
        self, config, data_df, num_classes, mode
    ):
        """
        Params:
            config: ExperimentationConfig object
            data_df: data DataFrame of image name and labels   
            mode: train or test/validation            
        """
        super().__init__()
        self.config = config
        self.exp_config = config.expconfig
        self.mode = mode
        self.num_classes = num_classes 
        self.data_df = data_df  
        self.imsize = self.exp_config.imsize  
        self.images_df = pd.read_csv(self.data_df, index_col=False)
        self.prep_data_sheet()     
        self.kwargs_augmentation = self.exp_config.kwargs_augmentation

        self.transform = CustomCombinedTransform(
            self.config,
            imsize=self.imsize,           
            mode=self.mode,
            **dict(self.kwargs_augmentation),
        )

    def __len__(self) -> int:
        return len(self.images_df)

    def __getitem__(self, idx: int) -> tuple:
        # Load image and label from DataFrame
        try:
            current_sample = self.images_df.iloc[idx]
            img_path = current_sample["Id"]
            label = current_sample["y"]    
            img = Image.open(img_path).convert("RGB")  # Load image
                  
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        except Exception as e:
            print(f"Error applying transformation on image with index {idx}: {e}")
            print(f"Image path: {img_path}")
            return self.__getitem__((idx + 1) % self.__len__())

        if img is None:
            return self.__getitem__((idx + 1) % self.__len__())
        
      
      
        if self.mode == "Train":            
            img = self.transform.alb_transform_train(np.array(img))

        # cv2.imwrite(f"img_{time.time()}.jpg", img)     
        img = self.transform.tv_transform(np.array(img))     
        return img, label

    def prep_data_sheet(self):
        self.images_df = self.images_df[(self.images_df["y"] != -1) & (self.images_df["Split"] == self.mode)].copy()
        self.images_df = self.images_df[self.images_df['Id'].apply(lambda x: os.path.exists(x))]        
        self.images_df = self.images_df[self.images_df["y"] != self.num_classes]     
        self.images_df = self.images_df[self.images_df["Split"] == self.mode]
        if not os.path.exists("tmp"):
            os.makedirs("tmp", exist_ok=True)
        self.images_df.to_csv("tmp/current_run_{}_dataset.csv".format(self.mode), index=False)

    def get_labels(self):
        return self.images_df["y"]




def get_fastai_dataloaders(config, train_Dataset, valid_Dataset) -> list:
    world_size = torch.cuda.device_count()
    # rank = int(os.getenv('LOCAL_RANK', '0'))  
    # sampler = DistributedSampler(train_Dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dl = FastDataLoader(
        train_Dataset,
        batch_sampler=ImbalancedDatasetSampler(train_Dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # change here
        # sampler = sampler
    )
    # sampler = DistributedSampler(valid_Dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_dl = FastDataLoader(
        valid_Dataset,
        batch_size=config.batch_size,      
        shuffle=False,
        num_workers=config.num_workers
    )

    return [train_dl, valid_dl]