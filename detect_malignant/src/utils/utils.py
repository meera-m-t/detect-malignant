import argparse
import glob
import cv2
import random
import torch
import torchvision.transforms as T



from PIL import Image
from torchvision.utils import save_image
from os import  makedirs, path
from typing import List, Optional


class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, data):
        print(f"{self.name} - {data}")


def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_rgb_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    save_name: str,
    predicted_masks: Optional[List[torch.Tensor]] = None,
    padding=10,
):
    if predicted_masks is not None:
        predicted_masks = [torch.unsqueeze(predicted_mask * 255, dim=0) for predicted_mask in predicted_masks]

    mask = torch.unsqueeze(mask * 255, dim=0)
    save_image(
        [image.float(), torch.cat([mask, mask, mask], dim=0)]
        if predicted_masks is None
        else [
            image.float(),
            torch.cat([mask] * 3, dim=0),
            *[torch.cat([predicted_mask] * 3, dim=0) for predicted_mask in predicted_masks],
        ],
        save_name,
        padding=padding,
        normalize=False,
        pad_value=4,
    )


def str2bool(v):
    if v.lower() in ["true", 1]:
        return True
    elif v.lower() in ["false", 0]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_as_tensor(image_path: str, size_img: int):
    transform = T.Compose([T.ToPILImage(), T.Resize((size_img, size_img)), T.ToTensor()])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image)


def get_image_tensors(image_directory: str, img_size: int, limit=100):
    all_images = glob.glob(image_directory + "/*.png") + glob.glob(image_directory + "/*.jpg")
    if len(all_images) > limit:
        all_images = random.sample(all_images, limit)
    tensors = []
    print(all_images, img_size)
    for image in all_images:
        image_tensor = read_as_tensor(image, img_size)
        tensors.append(image_tensor)

    return torch.stack(tensors)




def clean_df(df, mc=False):
    if mc:
        return df
    else:
        df = df.dropna(subset=["y"])
        df = df[df.y != -1]
        return df


def trim_ghost_imgs(df):
    for i in df.index:
        img_fn = df.loc[i, "Id"]
        if not path.exists(img_fn):
            df = df.drop(i)
    return df


