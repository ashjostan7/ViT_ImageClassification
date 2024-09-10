'''
Dataloader function for image classification
'''

import os 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_OF_WORKERS = os.cpu_count()

def create_dataloaders(
                        train_dir : str,
                        test_dir : str,
                        transform : transforms.Compose,
                        batch_size : int,
                        num_workers : int = NUM_OF_WORKERS):
    
    """
    Create train and test dataloaders
    """

    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform= transform)

    #class names:
    class_names = train_data.classes

    #train dataloader:
    train_dataloader = DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    ) 
    # test dataloader:
    test_dataloader = DataLoader(
        test_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    ) 
    
    return train_dataloader, test_dataloader, class_names
