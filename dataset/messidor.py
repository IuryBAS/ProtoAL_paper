"""
Load datasets
"""
import PIL.Image as io
import torch
import torch.utils.data as TD
from torchvision.transforms import (Compose,
                                    ToTensor,
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip)

from utils import std_mean


class Messidor(TD.Dataset):
    '''
    Load Messidor Dataset, applying given transforms. If none transformation is inputed,
    apply the default tranforms
    
    '''
    def __init__(self, dataframe, train_transform=None, label_transform=None):
        self.image = dataframe['Image name']
        self.label = dataframe['Retinopathy grade']
        self.transform, self.label_transform = self.default_transforms()
        if train_transform:
            self.transform = train_transform
        if label_transform:
            self.label_transform = label_transform
    
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_pil = io.open(self.image[index])
        image = self.transform(image_pil)
        label = self.label[index]
        if self.label_transform:
            label = self.label_transform(label)
        return index, image, label

    def default_transforms(self):
        std, mean = std_mean['messidor']['std'], std_mean['messidor']['mean']
        img_transform=Compose([
                RandomRotation(degrees=15),
                RandomResizedCrop(
                    512, scale=(0.9, 1.0), ratio=(1, 1), antialias=True),
                RandomHorizontalFlip(),
                #  tvt.RandomVerticalFlip(),
                
                ToTensor(),
                Normalize(std=std, mean=mean),
            ])
        getitem_transform=lambda x: (
            torch.tensor(int(x != 0)))
        return img_transform, getitem_transform
