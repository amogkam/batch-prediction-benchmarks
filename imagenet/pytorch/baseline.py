# PyTorch only baseline.
import click
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

import torch
from torch.utils.data import DataLoader

from theoretical import BATCH_SIZE

def main():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
    ])

    class CachedImageFolder(ImageFolder):
        """An ImageFolder dataset that caches the preprocessed images."""
        cache = {}

        def __getitem__(self, index):
            if index in self.cache:
                return self.cache[index]
            else:
                obj = super().__getitem__(index)
                image = obj[0] # Drop the label.
                self.cache[index] = obj
                return obj

    dataset = CachedImageFolder(root="/home/ray/batch-prediction-benchmark/imagenet/data", transforms=preprocess)

    # Iterate through the dataset once to apply and cache all the transformations.
    # This mimics the same behavior as Ray Datasets where we preprocess first and then do prediction.
    for _ in dataset:
        pass

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=..., 
        pin_memory=..., 
        prefetch_factor=..., 
        persistent_workers=False, # Does not matter for prediction as we only iterate through the dataset once. 
        pin_memory_device=...)

    model = resnet50(weights=ResNet50_Weights)
    model.eval()

    total_images = 0
    for batch in dataloader:
        tensor = torch.Tensor(batch, device="cuda")
        with torch.inference_mode():
            model(tensor)
        total_images += len(batch)

if __name__ == "__main__":
    main()

