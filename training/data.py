import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import random
from functools import partial
from config import NUM_WORKERS

class ImageCSVDataset(Dataset):
    """
    Expects a CSV with two columns:
        filename  - path to the image file (absolute, or relative to `img_dir`)
        label     - integer class index
    """
 
    def __init__(self, dataframe, img_dir="", transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_name"]
        if img_name[:3] == 'bad':
            img_path = os.path.join(self.img_dir, "BadSeed", self.df.loc[idx, "image_name"])
        else:
            img_path = os.path.join(self.img_dir, "GoodSeed", self.df.loc[idx, "image_name"])
        image    = Image.open(img_path).convert("RGB")
        label    = int(self.df.loc[idx, "label"])
    
        if self.transform:
            image = self.transform(image)
 
        return image, label
    
aug_t = transforms.Compose([ # augmentations, e.g. flip, rotate, jitter
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                        std =[0.229, 0.224, 0.225]),
])

train_t = transforms.Compose([ # Resizing, Tensor, Normalise. Basic transforms.
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                        std =[0.229, 0.224, 0.225]),
])

val_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
])


def get_dataloaders(csv_path, aug_transform, train_transform, val_transform, img_dir="", batch_size=32,
                    val_split=0.2, num_workers=2, concat_data=True, seed=42):
    
    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=seed
    )

    aug_train_data = ImageCSVDataset(train_df, img_dir, transform=aug_transform)
    og_train_data  = ImageCSVDataset(train_df, img_dir, transform=train_transform)
    val_dataset    = ImageCSVDataset(val_df,   img_dir, transform=val_transform)

    # ── reproducibility ──────────────────────────────
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    # ─────────────────────────────────────────────────────────

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)

    if concat_data:
        combined_train_dataset = torch.utils.data.ConcatDataset([og_train_data, aug_train_data])
        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    worker_init_fn=seed_worker, generator=g, drop_last=True)

        return combined_train_dataset, train_loader, val_dataset, val_loader
    else:
        train_loader = DataLoader(og_train_data, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    worker_init_fn=seed_worker, generator=g, drop_last=True)
        aug_loader   = DataLoader(aug_train_data, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    worker_init_fn=seed_worker, generator=g, drop_last=True)

        return og_train_data, train_loader, aug_train_data, aug_loader, val_dataset, val_loader
    
get_baseline_dataloaders = partial(
    get_dataloaders,
    aug_transform=aug_t,
    train_transform=train_t,
    val_transform=val_t
)

class SeedCropDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # You NEED a way to map row → image file
        # assuming filename can be reconstructed
        img_name = f"{row['seed_name']} (s{row['set_no']}).JPG"
        img_path = f"{self.img_dir}/Set{row['set_no']}/{img_name}"

        image = Image.open(img_path).convert("RGB")

        # crop: (left, upper, right, lower)
        crop = image.crop((
            row["xmin"],
            row["ymin"],
            row["xmax"],
            row["ymax"]
        ))

        if self.transform:
            crop = self.transform(crop)
        label = row["label"]
        return crop, img_path, label
    
def make_batch(df, batch_no, img_dir, transform, batch_size=32, shuffle=False, num_workers=NUM_WORKERS):
    batch_df = df.loc[df["batch_no"] == batch_no].reset_index(drop=True)
    batch_dataset = SeedCropDataset(batch_df, img_dir=img_dir, transform=transform)
    batch_loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return batch_df, batch_dataset, batch_loader
