"""
Data loading
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.data_utils import SegmentationMapToMultiClass, SegmentationMapAndMultiClass

def load_data(mode, dataset_root, img_resolution, batch_size, download=False):
    """ Load transforms and images into a DataLoader. """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif mode == 'val':
        if batch_size == 1: # Keep original image dimensions
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.Resize(int(img_resolution * 1.05)),
                transforms.CenterCrop(img_resolution),
                transforms.ToTensor(),
                normalize
            ])
    else:
        raise Exception(f'Invalid dataset mode specified: {mode}')

    num_classes = 20
    if mode == 'val' and batch_size == 1:
        target_transform = SegmentationMapAndMultiClass(num_classes)
    else:
        target_transform = SegmentationMapToMultiClass(num_classes)
    imagefolder = datasets.VOCSegmentation(dataset_root, '2012', mode, download=download, transform=data_transforms,
                                            target_transform=target_transform)
    shuffle = mode == 'train'
    dataloader = DataLoader(imagefolder, batch_size=batch_size, shuffle=shuffle)
    return dataloader
