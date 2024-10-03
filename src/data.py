from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

def get_dataloaders(img_size, batch_size):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (img_size, img_size), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )

    full_trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

   
    val_size = len(full_trainset) // 5  # 20% 
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader