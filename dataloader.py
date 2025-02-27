import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.image_names = [name for name in self.image_names if not os.path.isdir(os.path.join(image_dir, name))]
        self.mask_names = [name for name in self.mask_names if not os.path.isdir(os.path.join(mask_dir, name))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.where(mask > 0, 1, 0).float()
        return image, mask

def create_dataloaders():
    train_image_dir = "data/train/image"
    train_mask_dir = "data/train/mask"
    test_image_dir = "data/test/MDvsFA/image"
    test_mask_dir = "data/test/MDvsFA/mask"

    train_dataset = SegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
        mask_transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    )
    test_dataset = SegmentationDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
        mask_transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    return train_loader, test_loader