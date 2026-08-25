import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.nn import functional as F
from torchvision.models import resnet18, resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import (
         accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        ConfusionMatrixDisplay
    )
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from functools import partial
import time
import copy
import datetime
import torch
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math
from itertools import product
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from itertools import product
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torchinfo import summary

if __name__ == '__main__':
    
    
    

    
    
    
    
    seed = 42
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    ##!pip install grad-cam
    
    
    #---
    
    NUM_WORKERS = 4
    
    #---
    
    # %%
    """
    # Baseline model 
    """
    
    #---
    
    # %%
    """
    ## Helper Functions
    """
    
    #---
    
    def write_im_path(good_seeds_path, bad_seeds_path, csv_rel_path="csv_folder/train.csv"):
        csv_path = os.path.join(os.getcwd(), csv_rel_path)
    
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "label"])
    
            # Write good seeds
            for filename in good_seeds_path:
                writer.writerow([filename, 1])
    
            # Write bad seeds (FIXED: not nested)
            for filename in bad_seeds_path:
                writer.writerow([filename, 0])
    
        print(f"{csv_path} written.")
    
        return pd.read_csv(csv_path)
    
    def denormalize(tensor):
        """Reverse ImageNet normalization for display."""
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_dn  = tensor.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        img_dn  = std * img_dn + mean                 # undo normalize
        img_dn  = np.clip(img_dn, 0, 1)
        return img_dn
    
    
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = denormalize(inp)
        plt.imshow(inp)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    #---
    
    goodseed_train = sorted(os.listdir("dataset/batch1/train/GoodSeed"))
    badseed_train = sorted(os.listdir("dataset/batch1/train/BadSeed"))
    print(f"{len(goodseed_train)} good seeds")
    print(f"{len(badseed_train)} bad seeds")
    print(f"Training set: {len(goodseed_train) + len(badseed_train)} seeds.")
    
    #---
    
    batch1_train_df = write_im_path(
        goodseed_train,
        badseed_train,
        csv_rel_path = "csv_folder/train.csv"
    )
    
    #---
    
    # ─────────────────────────────────────────────
    # 1.  Dataset
    # ─────────────────────────────────────────────
    class ImageCSVDataset(Dataset):
        "\n" +\
    "    Expects a CSV with two columns:\n" +\
    "        filename  - path to the image file (absolute, or relative to `img_dir`)\n" +\
    "        label     - integer class index\n" +\
    "    "
     
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
    
    #---
    
    # %%
    """
    ## Defining augmentations
    """
    
    #---
    
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
    
    #---
    
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
    
    #---
    
    get_baseline_dataloaders = partial(
        get_dataloaders,
        aug_transform=aug_t,
        train_transform=train_t,
        val_transform=val_t
    )
    
    #---
    
    # %%
    """
    ## Training the baseline model
    """
    
    #---
    
    train_dataset, train_loader, val_dataset, val_loader = get_baseline_dataloaders(
        csv_path = "csv_folder/train.csv",
        img_dir = "dataset/batch1/train",
        batch_size=32,
        val_split=0.2,
        num_workers=NUM_WORKERS
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    all_dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    dataset_sizes = {'train' : len(train_dataset),
                     'val' : len(val_dataset)
                     }
    
    class_names = ['bad', 'good']
    
    print(dataset_sizes)
    print(class_names)
    
    
    # Grab one batch from each loader, take the first image
    train_images, train_labels = next(iter(train_loader))
    val_images,   val_labels   = next(iter(val_loader))
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 16))
    
    out = torchvision.utils.make_grid(
        train_images[:8]
    )
    
    imshow(out, title = [class_names[x] for x in train_labels[:8]])
    
    
    
    #---
    
    def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, val_acc, path):
        checkpoint = {
            "epoch"      : epoch,
            "model_state_dict"      : model.state_dict(),
            "optimizer_state_dict"  : optimizer.state_dict(),
            "scheduler_state_dict"  : scheduler.state_dict(),
            "val_loss"   : val_loss,
            "val_acc"    : val_acc,
        }
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved → {path}")
    
    #---
    
    
    def train_model(model, crtrn, optimizer=None, scheduler=None, num_epochs=25, ckpt_save_interval = 20, checkpoint_dir = "ckpts", history=None):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        os.makedirs(checkpoint_dir, exist_ok = True)
        checkpoint_dir_dt = os.path.join(checkpoint_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(checkpoint_dir_dt, exist_ok = False)
    
        if history is None:
            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for ins, labs in all_dataloaders[phase]:
                    ins = ins.to(device)
                    labs = labs.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outs = model(ins)
                        _, model_pred = torch.max(outs, 1)
                        loss_fn = crtrn(outs, labs)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_fn.backward()
                            optimizer.step()
                # END BATCH
    
                    # statistics
                    running_loss += loss_fn.item() * ins.size(0)
                    running_corrects += torch.sum(model_pred == labs.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if phase == 'train':
                    scheduler.step()
        
    
                else:
    
                    if (epoch + 1) % ckpt_save_interval == 0 and epoch + 1 < num_epochs:
                        save_checkpoint(
                            epoch, model, optimizer, scheduler, epoch_loss, epoch_acc,
                            path=os.path.join(checkpoint_dir_dt, f"epoch_{epoch:03d}.pth")
                        )        
                        torch.save(
                            history,
                            os.path.join(checkpoint_dir_dt, f"history_epoch_{epoch:03d}.pth")
                        )
                    elif epoch + 1 == num_epochs:
                        save_checkpoint(
                            epoch, model, optimizer, scheduler, epoch_loss, epoch_acc,
                            path=os.path.join(checkpoint_dir_dt, f"last.pth")
                        )  
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(float(epoch_acc))
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            print()
             
        torch.save(best_model_wts, os.path.join(checkpoint_dir_dt, "best.pth"))
    
        # ── save history alongside checkpoints ───────────────────────
        torch.save(history, os.path.join(checkpoint_dir_dt, "history.pth"))
        # ─────────────────────────────────────────────────────────────
           
    
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, history
    
    #---
    
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
    
        with torch.no_grad():
            for i, (inputs, labs) in enumerate(all_dataloaders['val']):
                inputs = inputs.to(device)
                labs = labs.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}\n actual: {class_names[labs[j]]}')
                    imshow(inputs.cpu().data[j])
    
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
    
    #---
    
    def get_baseline_models(model_name = "resnet18", none_weights = False):
        if model_name == "resnet50_ffe":
            model_ft = resnet50(weights=None if none_weights else ResNet50_Weights.IMAGENET1K_V1)
            for param in model_ft.parameters():
                param.requires_grad = False
                
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 2)
            
            
                
        elif model_name == "resnet50_mlp_ffe":
            model_ft = resnet50(weights=None if none_weights else ResNet50_Weights.IMAGENET1K_V1)
            for param in model_ft.parameters():
                param.requires_grad = False
                
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
                
        elif model_name == "resnet50":
            model_ft = resnet50(weights=None if none_weights else ResNet50_Weights.IMAGENET1K_V1)
    
            for param in model_ft.parameters():
                param.requires_grad = False
                
            for param in model_ft.layer4.parameters():
                param.requires_grad = True
            for param in model_ft.layer3.parameters():
                param.requires_grad = True
    
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 2)
    
        elif model_name == "resnet50_mlp":
            model_ft = resnet50(weights=None if none_weights else ResNet50_Weights.IMAGENET1K_V1)
    
            for param in model_ft.parameters():
                param.requires_grad = False
                
            for param in model_ft.layer4.parameters():
                param.requires_grad = True
            for param in model_ft.layer3.parameters():
                param.requires_grad = True
    
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Sequential(
                nn.Linear(num_ftrs, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
    
        else:
            model_ft = None
    
        return model_ft
    
    #---
    
    # resnet50_ffe
    criterion = nn.CrossEntropyLoss()
    model_name = "resnet50_ffe"
    if not os.path.exists(f"models/{model_name}.pt"):
        model_rn50_ffe = get_baseline_models(
            model_name = model_name
        )
    
        model_rn50_ffe = model_rn50_ffe.to(device)
    
        # checkpoint = torch.load("path/to/checkpoint.pth", map_location=device)
        optimizer_ft = optim.SGD(model_rn50_ffe.parameters(), lr=0.001, momentum=0.9)
        
        # optimizer_ft.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
        # exp_lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        model_rn50_ffe, history = train_model(model_rn50_ffe, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=150)
    
        visualize_model(model_rn50_ffe)
        torch.save(model_rn50_ffe.state_dict(), f"models/{model_name}.pt")
    
    else:
        model_rn50_ffe = get_baseline_models(
            model_name=model_name,
            none_weights=True
        )
        model_rn50_ffe.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
        model_rn50_ffe = model_rn50_ffe.to(device)
        model_rn50_ffe.eval()
        for param in model_rn50_ffe.parameters():
            param.requires_grad = True
    
    
    #---
    
    # resnet50_mlp_ffe
    criterion = nn.CrossEntropyLoss()
    model_name = "resnet50_mlp_ffe"
    if not os.path.exists(f"models/{model_name}.pt"):
        model_rn50_mlp_ffe = get_baseline_models(
            model_name = model_name
        )
    
        model_rn50_mlp_ffe = model_rn50_mlp_ffe.to(device)
    
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_rn50_mlp_ffe.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
        model_rn50_mlp_ffe, history = train_model(model_rn50_mlp_ffe, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=150)
    
        visualize_model(model_rn50_mlp_ffe)
        torch.save(model_rn50_mlp_ffe.state_dict(), f"models/{model_name}.pt")
    
    else:
        model_rn50_mlp_ffe = get_baseline_models(
            model_name=model_name,
            none_weights=True
        )
        model_rn50_mlp_ffe.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
        model_rn50_mlp_ffe = model_rn50_mlp_ffe.to(device)
        model_rn50_mlp_ffe.eval()
        for param in model_rn50_mlp_ffe.parameters():
            param.requires_grad = True
    
    #---
    
    # resnet50_mlp
    criterion = nn.CrossEntropyLoss()
    model_name = "resnet50_mlp"
    if not os.path.exists(f"models/{model_name}.pt"):
        model_rn50_mlp = get_baseline_models(
            model_name = model_name
        )
    
        model_rn50_mlp = model_rn50_mlp.to(device)
    
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_rn50_mlp.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
        model_rn50_mlp, history = train_model(model_rn50_mlp, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=150)
    
        visualize_model(model_rn50_mlp)
        torch.save(model_rn50_mlp.state_dict(), f"models/{model_name}.pt")
    
    else:
        model_rn50_mlp = get_baseline_models(
            model_name=model_name,
            none_weights=True
        )
        model_rn50_mlp.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
        model_rn50_mlp = model_rn50_mlp.to(device)
        model_rn50_mlp.eval()
        for param in model_rn50_mlp.parameters():
            param.requires_grad = True
    
    #---
    
    # resnet50
    criterion = nn.CrossEntropyLoss()
    model_name = "resnet50"
    if not os.path.exists(f"models/{model_name}.pt"):
        model_rn50 = get_baseline_models(
            model_name = model_name
        )
    
        model_rn50 = model_rn50.to(device)
    
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_rn50.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
        model_rn50, history = train_model(model_rn50, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=150)
    
        visualize_model(model_rn50)
        torch.save(model_rn50.state_dict(), f"models/{model_name}.pt")
    
    else:
        model_rn50 = get_baseline_models(
            model_name=model_name,
            none_weights=True
        )
        model_rn50.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
        model_rn50 = model_rn50.to(device)
        model_rn50.eval()
        for param in model_rn50.parameters():
            param.requires_grad = True
    
    #---
    
    directories = [
        # "ckpts/ResNet50_ffe",
        # "ckpts/ResNet50_mlp_ffe",
        "ckpts/ResNet50_mlp",
        "ckpts/ResNet50"
    ]
    
    
    def plot_histories(directories, fig_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        labels = [os.path.basename(d) for d in directories]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        for i, (d, label) in enumerate(zip(directories, labels)):
            history = torch.load(os.path.join(d, "history.pth"), weights_only=True)
            color = colors[i % len(colors)]
    
            axes[0].plot(history["train_acc"], label=f"{label} (train)", color=color, linestyle="--")
            axes[0].plot(history["val_acc"],   label=f"{label} (val)",   color=color, linestyle="-")
    
            axes[1].plot(history["train_loss"], label=f"{label} (train)", color=color, linestyle="--")
            axes[1].plot(history["val_loss"],   label=f"{label} (val)",   color=color, linestyle="-")
    
        for ax, title, ylabel in zip(
            axes,
            ["Accuracy over Epochs", "Loss over Epochs"],
            ["Accuracy", "Loss"]
        ):
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=12)
            ax.grid(True)
    
        plt.tight_layout()
        plt.savefig(fig_name, dpi=150)
        plt.show()
    
    plot_histories(directories, fig_name = "baseline_training_histories.png")
    
    #---
    
    # %%
    """
    ## Evaluating baseline model on batch 1
    """
    
    #---
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    
    #---
    
    goodseed_test = sorted(os.listdir("dataset/batch1/test/GoodSeed"))
    badseed_test = sorted(os.listdir("dataset/batch1/test/BadSeed"))
    print(f"{len(goodseed_test)} good seeds")
    print(f"{len(badseed_test)} bad seeds")
    print(f"Testing set: {len(goodseed_test) + len(badseed_test)} seeds.")
    
    #---
    
    batch1_test_df = write_im_path(
        goodseed_test,
        badseed_test,
        csv_rel_path = "csv_folder/test.csv"
    )
    
    #---
    
    img_dir = "dataset/batch1/test"
    
    batch_size = 32
    num_workers = 4
    
    test_dataset = ImageCSVDataset(batch1_test_df, img_dir, transform=test_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    
    
    
    #---
    
    
    
    # pip install grad-cam
    
    #---
    
    def plot_cases_grad_eigen_cam(images, preds, labels, probs, model, target_layer, title, 
                                   class_names=None, max_images=32, ncols=4, eigen=False, minimal=False):
        "\n" +\
    "    target_layer: the conv layer to hook into, e.g. model_ft.layer4[-1] for ResNet\n" +\
    "    ncols is smaller here since each cell is wider (image + cam side by side)\n" +\
    "    minimal: if True, show only the CAM overlay grid (no original images)\n" +\
    "    "
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
        n = min(len(images), max_images)
        ncols = min(ncols, n)
        nrows = math.ceil(n / ncols)
        if not eigen:
            cam = GradCAM(model=model, target_layers=[target_layer])
        else:
            cam = EigenCAM(model=model, target_layers=[target_layer])
    
        # In minimal mode: 1 column per image; otherwise: 2 columns per image (original + overlay)
        fig_ncols = ncols if minimal else ncols * 2
        fig, axes = plt.subplots(nrows, fig_ncols, figsize=(ncols * (3 if minimal else 5), nrows * 3))
        axes = np.array(axes).reshape(nrows, fig_ncols)
    
        model.eval()
    
        for i in range(n):
            img_tensor = images[i]
            img_vis    = (img_tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
    
            input_tensor = img_tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(preds[i].item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
            overlay = show_cam_on_image(img_vis.astype(np.float32), grayscale_cam, use_rgb=True)
    
            pred_name  = class_names[preds[i]]  if class_names else f"cls {preds[i].item()}"
            label_name = class_names[labels[i]] if class_names else f"cls {labels[i].item()}"
            conf = probs[i][preds[i].item()].item() * 100
    
            if minimal:
                row, col = divmod(i, ncols)
                axes[row, col].imshow(overlay)
                axes[row, col].set_title(f"({i+1}) GT: {label_name}\nPred: {pred_name} ({conf:.2f}%)", 
                                         fontsize=9, color="red")
                axes[row, col].axis("off")
            else:
                row, col_base = divmod(i, ncols)
                col_base *= 2
                axes[row, col_base].imshow(img_vis)
                axes[row, col_base].set_title(f"({i + 1}) GT: {label_name}", fontsize=12)
                axes[row, col_base].axis("off")
                axes[row, col_base + 1].imshow(overlay)
                axes[row, col_base + 1].set_title(f"Pred: {pred_name} ({conf:.2f}%)", fontsize=12, color="red")
                axes[row, col_base + 1].axis("off")
    
        # Hide leftover cells
        for i in range(n, nrows * ncols):
            row, col_base = divmod(i, ncols)
            if minimal:
                axes[row, col_base].axis("off")
            else:
                col_base *= 2
                axes[row, col_base].axis("off")
                axes[row, col_base + 1].axis("off")
    
        fig.suptitle(f"{'EigenCAM' if eigen else 'GradCAM'} {title} ({len(images)} total, showing {n})", fontsize=13)
        plt.tight_layout()
        plt.show()
        return fig
    
    #---
    
    
    
    # model = model_rn50_ffe
    # model = model_rn50_mlp_ffe
    # model = model_rn50_mlp
    model      = model_rn50 # Change this
    model.eval()
    model_name = [k for k, v in globals().items() if v is model][0]
    
    all_preds = []
    all_labels = []
    all_images = []  # store images too
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:   # <-- USE TEST LOADER
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
    
            # classification
            preds = torch.argmax(outputs, dim=1)
    
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_images.append(images.cpu())  # collect images
            all_outputs.append(outputs.cpu())
    
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
        
    
    
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_images = torch.cat(all_images)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    all_probs = torch.softmax(all_outputs, dim=1)
    
    # --- Failure cases ---
    fp_mask = (all_preds == 1) & (all_labels == 0)
    fn_mask = (all_preds == 0) & (all_labels == 1)
    tp_mask = (all_preds == 1) & (all_labels == 1)
    tn_mask = (all_preds == 0) & (all_labels == 0)
    
    
    # High heat regions imply that the feature map activations in that region had large positive gradients w.r.t to logits
    
    
    
    
    layer_index = -1
    
    all_cases = {
        "False Positive Cases": fp_mask,
        "False Negative Cases": fn_mask,
        "True Positive Cases":  tp_mask,
        "True Negative Cases":  tn_mask,
    }
    
    for case, layer_name, use_eigen, minimal_plots in product(
        ["False Positive Cases", "False Negative Cases"],
        ["layer4"],
        [False],
        [True],
    ):
        mask = all_cases[case]
        target_layer = getattr(model, layer_name)[layer_index]
        layer_label  = f"{layer_name}[{layer_index}]"
        cam_type     = "EigenCAM" if use_eigen    else "GradCAM"
        plot_style   = "Minimal"  if minimal_plots else "Full"
        num_images_plot = 4 if minimal_plots else 32
    
    
        fig = plot_cases_grad_eigen_cam(
            all_images[mask], all_preds[mask], all_labels[mask], all_probs[mask],
            model=model,
            title="",
            target_layer=target_layer,
            class_names=None,
            max_images=num_images_plot,
            ncols=4,
            eigen=use_eigen,
            minimal=minimal_plots,
        )
    
    
    #---
    
    # =========================================================
    # 1) Put your models here
    # =========================================================
    models_dict = {
        "RN50": model_rn50,
        "RN50_MLP": model_rn50_mlp,
        "RN50_FFE": model_rn50_ffe,
        "RN50_MLP_FFE": model_rn50_mlp_ffe,
    }
    model_order = list(models_dict.keys())
    # =========================================================
    # 2) Helper: evaluate one model
    # =========================================================
    def evaluate_model_batch1(model, loader, criterion, device):
        model.eval()
    
        all_preds = []
        all_labels = []
        all_outputs = []
        all_images = []
    
        total_loss = 0.0
    
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)   # shape: [B, num_classes]
                loss = criterion(outputs, labels)
    
                preds = torch.argmax(outputs, dim=1)
    
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
                all_images.append(images.cpu())
    
                total_loss += loss.item() * images.size(0)
    
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_images = torch.cat(all_images, dim=0)
    
        # Probabilities for positive class (binary classification)
        all_probs = torch.softmax(all_outputs, dim=1)[:, 1]
    
        avg_loss = total_loss / len(loader.dataset)
    
        # Metrics
        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        prec = precision_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        rec = recall_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
    
        # AUC only valid if both classes exist in y_true
        try:
            auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        except ValueError:
            auc = np.nan
    
        cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    
        # Masks / case subsets if you want them later
        fp_mask = (all_preds == 1) & (all_labels == 0)
        fn_mask = (all_preds == 0) & (all_labels == 1)
        tp_mask = (all_preds == 1) & (all_labels == 1)
        tn_mask = (all_preds == 0) & (all_labels == 0)
    
        return {
            "loss": avg_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "cm": cm,
            "preds": all_preds,
            "labels": all_labels,
            "probs": all_probs,
            "outputs": all_outputs,
            "images": all_images,
            "fp_mask": fp_mask,
            "fn_mask": fn_mask,
            "tp_mask": tp_mask,
            "tn_mask": tn_mask,
        }
    
    # =========================================================
    # 3) Helper: plot confusion matrix
    # =========================================================
    def plot_confusion_matrix(cm, ax, class_names=("Negative", "Positive"), title="Confusion Matrix"):
        cm = np.array(cm, dtype=np.int64)
    
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    
        ax.set_title(title)
    
    # =========================================================
    # 4) Evaluate all models
    # =========================================================
    results_batch1 = {}
    rows_batch1 = []
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        out = evaluate_model_batch1(model, test_loader, criterion, device)
        results_batch1[model_name] = out
    
        rows_batch1.append({
            "Model": model_name,
            "Loss": out["loss"],
            "Accuracy": out["accuracy"],
            "Precision": out["precision"],
            "Recall": out["recall"],
            "F1": out["f1"],
            "AUC": out["auc"],
        })
    
    # =========================================================
    # 5) Metrics table
    # =========================================================
    metrics_df = pd.DataFrame(rows_batch1)
    metrics_df["Model"] = pd.Categorical(
        metrics_df["Model"],
        categories=model_order,
        ordered=True
    )
    metrics_df = metrics_df.sort_values("Model").reset_index(drop=True)
    
    display(metrics_df)
    
    # Optional nicer formatting
    display(metrics_df.style.format({
        "Loss": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "AUC": "{:.4f}",
    }))
    
    # =========================================================
    # 6) Plot conservative confusion matrices
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    model_order = list(models_dict.keys())
    
    for ax, model_name in zip(axes, model_order):
        out = results_batch1[model_name]
    
        plot_confusion_matrix(
            out["cm"],
            ax=ax,
            class_names=("Bad", "Good"),
            title=model_name
        )
    
    plt.tight_layout()
    plt.show()
    
    #---
    
    # %%
    """
    ## Evaluating baseline on Batch-2 and Batch-3
    """
    
    #---
    
    # %%
    """
    ### Helper functions to parse Batch-2 and Batch-3 images
    """
    
    #---
    
    def crop_and_save(
        jpg_path,
        bboxes,
        batch_no,
        set_no,
        seed_layout="SpaceOutRandom",
        seed_class="Mix"
    ):
        out_dir = Path(f"cropped/batch{batch_no}")
        out_dir.mkdir(parents=True, exist_ok=True)
    
        image = Image.open(jpg_path).convert("RGB")
        draw  = ImageDraw.Draw(image)
    
        for idx, bb in enumerate(bboxes):
            draw.rectangle(
                [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]],
                outline="red",
                width=3,
            )
            draw.text(
                (bb["xmin"], bb["ymin"] - 12),
                bb["name"],
                fill="red",
            )
    
        safe_layout = seed_layout.replace(" ", "")
        out_filename = f"{safe_layout}_{seed_class}_Set{set_no}_annotated.png"
        out_path     = out_dir / out_filename
    
        image.save(out_path)
        print(f"Saved: {out_path}  ({image.width}×{image.height} px)")
    
        return out_path
    
    #---
    
    def get_bboxes(root):
        bboxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
    
            bndbox = obj.find('bndbox')
    
    
            if bndbox is None:
                continue
            bboxes.append({
                "name":  name,
                "xmin":  int(bndbox.findtext("xmin")),
                "ymin":  int(bndbox.findtext("ymin")),
                "xmax":  int(bndbox.findtext("xmax")),
                "ymax":  int(bndbox.findtext("ymax")),
            })
            
        return bboxes
    
    
    def seed_xml_parser(
        batch_no, 
        seed_layout, 
        seed_class, 
        set_no,
        verbose = False,
    ):
        if batch_no == 2:
            base_dir = Path(f"dataset/batch2/NormalRoomLighting/Set{set_no}")
            jpg_path = base_dir / f"{seed_layout}_{seed_class} (s{set_no}).JPG"
            seed_path = base_dir / f"{seed_layout}_{seed_class} (s{set_no}).xml"
        else:
            base_dir = Path(f"dataset/batch3/LightBox/Set{set_no}")
            jpg_path = base_dir / f"{seed_layout}_{seed_class.replace("_", "")} (s{set_no}).JPG"
            seed_path = base_dir / f"{seed_layout}_{seed_class.replace("_", "")} (s{set_no}).xml"
    
    
        try:
            tree = ET.parse(str(seed_path))
            root = tree.getroot()
            if verbose:
                print(f"===== Parsed \'{seed_path.stem}.xml\' =====")
            return root, base_dir, jpg_path, seed_path
        except FileNotFoundError:
            print(f"The file '{seed_path}' was not found.")
        except ET.ParseError:
            print("Error parsing the XML file.")
            
    
            
    def crop_and_save_to_df(
        batch_no,
        set_no,
        seed_layout = "SpaceOutRandom",
        seed_class = "Mix",
        df = None,
    ):
        seed_data = []
    
        root, _, _, _ = seed_xml_parser(batch_no, seed_layout, seed_class, set_no)
        bboxes = get_bboxes(root)
        for idx, bb in enumerate(bboxes):
            
            # PIL crop box is (left, upper, right, lower)
            # crop = image.crop((bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))
    
            
            seed_name = f"{seed_layout}_{seed_class if batch_no == 2 else seed_class.replace("_", "")}"
            class_name = 0 if bb["name"] == "Bad Seed" else 1
            seed_data.append([seed_name, idx, batch_no, set_no, bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"], class_name])
            
        if df is None: 
            df = pd.DataFrame(seed_data, columns = ["seed_name", "seed_no", "batch_no", "set_no", "xmin", "ymin", "xmax", "ymax", "label"])
        
        else:
            df = pd.concat((df, pd.DataFrame(seed_data, columns = df.columns)), axis = 0)
        
        
        return df, bboxes
    
    def show_image_with_bb(seed_path):
        try:
            tree = ET.parse(str(seed_path + ".xml"))
            root = tree.getroot()
    
        except FileNotFoundError:
            print(f"The file '{str(seed_path + '.xml')}' was not found.")
            return
        except ET.ParseError:
            print("Error parsing the XML file.")
            return
        
        bboxes = get_bboxes(root)
        image = Image.open(f"{seed_path}.JPG").convert("RGB")
        draw  = ImageDraw.Draw(image)
    
        for idx, bb in enumerate(bboxes):
            draw.rectangle(
                [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]],
                outline="red",
                width=3,
            )
            draw.text(
                (bb["xmin"], bb["ymin"] - 12),
                bb["name"],
                fill="red",
            )
        plt.figure(figsize=(4, 4))  # smaller figure
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    
    #---
    
    tree = None # initialise to None
    df = None
    erroneous_annotations = [
        rf"dataset/batch3/LightBox/Set19/SpaceOutRandom_GoodSeeds (s19)",
        rf"dataset/batch3/LightBox/Set14/SpaceOutRandom_GoodSeeds (s14)",
        rf"dataset/batch3/LightBox/Set12/SpaceOutRandom_GoodSeeds (s12)",
        rf"dataset/batch3/LightBox/Set19/Line_Mix (s19)",
        rf"dataset/batch3/LightBox/Set12/Line_Mix (s12)",
        rf"dataset/batch3/LightBox/Set4/Line_Mix (s4)",
    ]
    jpg_path = ""
    for batch_no in [2, 3]:
        if batch_no == 2:
            max_set = 15
        else:
            max_set = 20
        for set_no in range(1, max_set + 1):
            for seed_layout in ["Line", "SpaceOutRandom"]:
                
                for seed_class in ["Bad_Seeds", "Good_Seeds", "Mix"]:
                    if batch_no == 2:
                        jpg_path = rf"dataset/batch{batch_no}/NormalRoomLighting/Set{set_no}/{seed_layout}_{seed_class} (s{set_no}).JPG"
                    else:
                        jpg_path = rf"dataset/batch{batch_no}/LightBox/Set{set_no}/{seed_layout}_{seed_class.replace("_", "")} (s{set_no}).JPG"
                    if jpg_path[:-4] in erroneous_annotations:
                        print(f"Skipped erroneous annotations. {jpg_path}")
                        continue
                    df, bboxes = crop_and_save_to_df(batch_no, set_no, seed_layout, seed_class, df)
    
                        
                    out_path = crop_and_save(
                        jpg_path,
                        bboxes = bboxes,
                        batch_no = batch_no,
                        set_no = set_no,
                        seed_layout = seed_layout,
                        seed_class = seed_class
                    )
    
    
    
    
    #---
    
    for seed_path in erroneous_annotations:
        show_image_with_bb(seed_path)
    
    #---
    
    
    
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
    
    
    batch2_df, batch2_dataset, batch2_loader = make_batch(
        df, batch_no=2, img_dir="dataset/batch2/NormalRoomLighting", transform=test_transform
    )
    
    batch3_df, batch3_dataset, batch3_loader = make_batch(
        df, batch_no=3, img_dir="dataset/batch3/LightBox",          transform=test_transform
    )
    
    
    #---
    
    def show_batch_grid(loader, class_names, n=8, figsize=(16, 4)):
        images, paths, labels = next(iter(loader))
        grid = torchvision.utils.make_grid(images[:n])
        print(paths[:n])
        imshow(grid, title=[class_names[x] for x in labels[:n]])
    
    
    show_batch_grid(batch2_loader, class_names)
    show_batch_grid(batch3_loader, class_names, n = 10)
    
    #---
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    
    #---
    
    # =========================================================
    # 2) Evaluation function
    # =========================================================
    def evaluate_model_b2b3(model, loader, criterion, device):
        model.eval()
    
        all_preds = []
        all_labels = []
        all_outputs = []
        all_images = []
    
        total_loss = 0.0
    
        with torch.no_grad():
            for images, _, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)
                loss = criterion(outputs, labels)
    
                preds = torch.argmax(outputs, dim=1)
    
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
                all_images.append(images.cpu())
    
                total_loss += loss.item() * images.size(0)
    
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_images = torch.cat(all_images, dim=0)
    
        all_probs = torch.softmax(all_outputs, dim=1)[:, 1]
        avg_loss = total_loss / len(loader.dataset)
    
        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        prec = precision_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        rec = recall_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
    
        try:
            auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        except ValueError:
            auc = np.nan
    
        cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
        
        
        fp_mask = (all_preds == 1) & (all_labels == 0)
        fn_mask = (all_preds == 0) & (all_labels == 1)
        tp_mask = (all_preds == 1) & (all_labels == 1)
        tn_mask = (all_preds == 0) & (all_labels == 0)
    
        return {
            "loss": avg_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "cm": cm,
            "preds": all_preds,
            "labels": all_labels,
            "probs": all_probs,
            "outputs": all_outputs,
            "images": all_images,
            "fp_mask": fp_mask,
            "fn_mask": fn_mask,
            "tp_mask": tp_mask,
            "tn_mask": tn_mask,
        }
    
    #---
    
    # =========================================================
    # 1) Models + fixed order
    # =========================================================
    models_dict = {
        "RN50": model_rn50,
        "RN50_MLP": model_rn50_mlp,
        "RN50_FFE": model_rn50_ffe,
        "RN50_MLP_FFE": model_rn50_mlp_ffe,
    }
    
    model_order = list(models_dict.keys()) 
    
    
    
    
    
    # =========================================================
    # 3) Confusion matrix plot helper
    # =========================================================
    def plot_confusion_matrix(cm, ax, class_names=("Bad", "Good"), title=""):
        cm = np.array(cm, dtype=np.int64)
    
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    
        for text in disp.text_.ravel():
            text.set_fontsize(16)
    
        ax.set_title(title)
    
    
    # =========================================================
    # 4) Evaluate models
    # =========================================================
    results_batch2 = {}
    rows_batch2 = []
    
    for model_name in model_order:
        model = models_dict[model_name]
    
        print(f"Evaluating {model_name}...")
        out = evaluate_model_b2b3(model, batch2_loader, criterion, device)
    
        results_batch2[model_name] = out
    
        rows_batch2.append({
            "Model": model_name,
            "Loss": out["loss"],
            "Accuracy": out["accuracy"],
            "Precision": out["precision"],
            "Recall": out["recall"],
            "F1": out["f1"],
            "AUC": out["auc"],
        })
    
    
    # =========================================================
    # 5) Metrics table (FIXED ORDER)
    # =========================================================
    batch2_metrics_df = pd.DataFrame(rows_batch2)
    
    batch2_metrics_df["Model"] = pd.Categorical(
        batch2_metrics_df["Model"],
        categories=model_order,
        ordered=True
    )
    
    batch2_metrics_df = batch2_metrics_df.sort_values("Model").reset_index(drop=True)
    
    display(batch2_metrics_df)
    
    display(batch2_metrics_df.style.format({
        "Loss": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "AUC": "{:.4f}",
    }))
    
    
    # =========================================================
    # 6) 2x2 Confusion Matrix Grid (FIXED ORDER)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    for ax, model_name in zip(axes, model_order):
        out = results_batch2[model_name]
    
        plot_confusion_matrix(
            out["cm"],
            ax=ax,
            class_names=("Bad", "Good"),
            title=model_name
        )
    
    plt.tight_layout()
    plt.show()
    
    #---
    
    # model = model_rn50_ffe
    # model = model_rn50_mlp_ffe
    # model = model_rn50_mlp
    model      = model_rn50 # Change this
    model.eval()
    model_name = [k for k, v in globals().items() if v is model][0]
    
    all_preds = []
    all_labels = []
    all_images = []  # store images too
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _, labels in batch2_loader:   # <-- USE TEST LOADER
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
    
            # classification
            preds = torch.argmax(outputs, dim=1)
    
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_images.append(images.cpu())  # collect images
            all_outputs.append(outputs.cpu())
    
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_images = torch.cat(all_images)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    all_probs = torch.softmax(all_outputs, dim=1)
    
    # --- Failure cases ---
    fp_mask = (all_preds == 1) & (all_labels == 0)
    fn_mask = (all_preds == 0) & (all_labels == 1)
    tp_mask = (all_preds == 1) & (all_labels == 1)
    tn_mask = (all_preds == 0) & (all_labels == 0)
    
    
    # High heat regions imply that the feature map activations in that region had large positive gradients w.r.t to logits
    
    
    
    
    layer_index = -1
    
    all_cases = {
        "False Positive Cases": fp_mask,
        "False Negative Cases": fn_mask,
        "True Positive Cases":  tp_mask,
        "True Negative Cases":  tn_mask,
    }
    
    for cases, layer_name, use_eigen, minimal_plots in product(
        ["False Negative Cases", "False Positive Cases"],
        ["layer4"],
        [False],
        [False],
    ):
        mask = all_cases[cases]
        target_layer = getattr(model, layer_name)[layer_index]
        layer_label  = f"{layer_name}[{layer_index}]"
        cam_type     = "EigenCAM" if use_eigen    else "GradCAM"
        plot_style   = "Minimal"  if minimal_plots else "Full"
        num_images_plot = 8 if minimal_plots else 32
    
    
        fig = plot_cases_grad_eigen_cam(
            all_images[mask], all_preds[mask], all_labels[mask], all_probs[mask],
            model=model,
            title="",
            target_layer=target_layer,
            class_names=None,
            max_images=num_images_plot,
            ncols=4,
            eigen=use_eigen,
            minimal=minimal_plots,
        )
    
    #---
    
    # model = model_rn50_ffe
    # model = model_rn50_mlp_ffe
    # model = model_rn50_mlp
    model      = model_rn50 # Change this
    model.eval()
    model_name = [k for k, v in globals().items() if v is model][0]
    
    all_preds = []
    all_labels = []
    all_images = []  # store images too
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():  
        for images, _, labels in batch3_loader:   # <-- USE TEST LOADER
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
    
            # classification
            preds = torch.argmax(outputs, dim=1)
    
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_images.append(images.cpu())  # collect images
            all_outputs.append(outputs.cpu())
    
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_images = torch.cat(all_images)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    all_probs = torch.softmax(all_outputs, dim=1)
    
    # --- Failure cases ---
    fp_mask = (all_preds == 1) & (all_labels == 0)
    fn_mask = (all_preds == 0) & (all_labels == 1)
    tp_mask = (all_preds == 1) & (all_labels == 1)
    tn_mask = (all_preds == 0) & (all_labels == 0)
    
    
    # High heat regions imply that the feature map activations in that region had large positive gradients w.r.t to logits
    
    
    
    
    layer_index = -1
    
    all_cases = {
        "False Positive Cases": fp_mask,
        "False Negative Cases": fn_mask,
        "True Positive Cases":  tp_mask,
        "True Negative Cases":  tn_mask,
    }
    
    for cases, layer_name, use_eigen, minimal_plots in product(
        ["False Negative Cases", "False Positive Cases"],
        ["layer4"],
        [False],
        [False],
    ):
        mask = all_cases[cases]
        target_layer = getattr(model, layer_name)[layer_index]
        layer_label  = f"{layer_name}[{layer_index}]"
        cam_type     = "EigenCAM" if use_eigen    else "GradCAM"
        plot_style   = "Minimal"  if minimal_plots else "Full"
        num_images_plot = 4 if minimal_plots else 32
    
    
        fig = plot_cases_grad_eigen_cam(
            all_images[mask], all_preds[mask], all_labels[mask], all_probs[mask],
            model=model,
            title="",
            target_layer=target_layer,
            class_names=None,
            max_images=num_images_plot,
            ncols=4,
            eigen=use_eigen,
            minimal=minimal_plots,
        )
    
    #---
    
    # =========================================================
    # 1) Enforce model order (same as everywhere else)
    # =========================================================
    model_order = list(models_dict.keys())
    
    # =========================================================
    # 2) Evaluate models
    # =========================================================
    results_batch3 = {}
    rows_batch3 = []
    
    for model_name in model_order:
        model = models_dict[model_name]
    
        print(f"Evaluating {model_name}...")
        out = evaluate_model_b2b3(model, batch3_loader, criterion, device)
    
        results_batch3[model_name] = out
    
        rows_batch3.append({
            "Model": model_name,
            "Loss": out["loss"],
            "Accuracy": out["accuracy"],
            "Precision": out["precision"],
            "Recall": out["recall"],
            "F1": out["f1"],
            "AUC": out["auc"],
        })
    
    
    # =========================================================
    # 3) Metrics table (FIXED ORDER)
    # =========================================================
    batch3_metrics_df = pd.DataFrame(rows_batch3)
    
    batch3_metrics_df["Model"] = pd.Categorical(
        batch3_metrics_df["Model"],
        categories=model_order,
        ordered=True
    )
    
    batch3_metrics_df = batch3_metrics_df.sort_values("Model").reset_index(drop=True)
    
    display(batch3_metrics_df)
    
    display(batch3_metrics_df.style.format({
        "Loss": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "AUC": "{:.4f}",
    }))
    
    
    # =========================================================
    # 4) 2x2 Confusion Matrix Grid (FIXED ORDER)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    for ax, model_name in zip(axes, model_order):
        out = results_batch3[model_name]  
    
        plot_confusion_matrix(
            out["cm"],
            ax=ax,
            class_names=("Bad", "Good"),
            title=model_name
        )
    
    plt.tight_layout()
    plt.show()
    
    #---
    
    def get_encoder_embeddings(model, dataloader, device):
        model.eval()
        embeddings = []
        labels_list = []
    
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
    
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
    
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
    
                x = model.avgpool(x)
                x = x.flatten(start_dim=1)   # (B, 2048)
    
                embeddings.append(x.cpu())
                labels_list.append(labels.cpu())
    
        return torch.cat(embeddings, dim=0), torch.cat(labels_list, dim=0)
    
    def get_encoder_embeddings_b2b3(model, dataloader, device):
        model.eval()
        embeddings = []
        labels_list = []
    
        with torch.no_grad():
            for images, _, labels in dataloader:
                images = images.to(device)
    
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
    
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
    
                x = model.avgpool(x)
                x = x.flatten(start_dim=1)   # (B, 2048)
    
                embeddings.append(x.cpu())
                labels_list.append(labels.cpu())
    
        return torch.cat(embeddings, dim=0), torch.cat(labels_list, dim=0)
    
    
    #---
    
    embeddings_b1, labels_b1 = get_encoder_embeddings(model_rn50, test_loader, device)
    
    #---
    
    embeddings_b2,  labels_b2  = get_encoder_embeddings_b2b3(model_rn50, batch2_loader,  device)
    embeddings_b3,  labels_b3  = get_encoder_embeddings_b2b3(model_rn50, batch3_loader,  device)
    
    #---
    
    # embeddings: (N, D) array, labels: (N,) array of true classes
    def kmeans_embeddings(embeddings, labels, n_clusters=None):
        if n_clusters is None:
            n_clusters = len(np.unique(labels))
    
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_assignments = kmeans.fit_predict(embeddings)
    
        
        
        ari_clusters = adjusted_rand_score(labels, cluster_assignments)
        nmi_clusters = normalized_mutual_info_score(labels, cluster_assignments)
    
        return cluster_assignments, {'ari': ari_clusters, 'nmi': nmi_clusters}
    
    #---
    
    cluster_assg_b1_baseline, sim_scores_b1_baseline = kmeans_embeddings(embeddings_b1, labels_b1, n_clusters=2)
    cluster_assg_b2_baseline, sim_scores_b2_baseline = kmeans_embeddings(embeddings_b2, labels_b2, n_clusters=2)
    cluster_assg_b3_baseline, sim_scores_b3_baseline = kmeans_embeddings(embeddings_b3, labels_b3, n_clusters=2)
    
    print(f"{'':20} {'Batch-1':>10} {'Batch-2':>10} {'Batch-3':>10}")
    print("-" * 50)
    print(f"{'ARI':20} {sim_scores_b1_baseline['ari']:>10.3f} {sim_scores_b2_baseline['ari']:>10.3f} {sim_scores_b3_baseline['ari']:>10.3f}")
    print(f"{'NMI':20} {sim_scores_b1_baseline['nmi']:>10.3f} {sim_scores_b2_baseline['nmi']:>10.3f} {sim_scores_b3_baseline['nmi']:>10.3f}")
    
    #---
    
    # %%
    """
    # BYOL (Bootstrap Your Own Latent)
    """
    
    #---
    
    class NormalizedMSELoss(nn.Module):
        def __init__(self) -> None:
            super(NormalizedMSELoss,self).__init__()
    
        def forward(self, view1, view2):
            v1 = F.normalize(view1, dim=-1)
            v2 = F.normalize(view2, dim=-1)
            return 2 - 2 * (v1 * v2).sum(dim=-1)
    
    #---
    
    class GaussianBlur(object):
        def __init__(self, p):
            self.p = p
    
        def __call__(self, img):
            if random.random() < self.p:
                sigma = random.random() * 1.9 + 0.1
                return img.filter(ImageFilter.GaussianBlur(sigma))
            else:
                return img
    
    
    class Transform:
        def __init__(self):
            self.transform = transforms.Compose([
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(224, scale=(0.2, 0.9))],
                    p=0.8
                ),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.8, contrast=0.8)],
                    p=0.2
                ),
                transforms.RandomRotation(30),
                GaussianBlur(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
            self.transform_prime = transforms.Compose([
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(224, scale=(0.6, 0.9))],
                    p=0.2
                ),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4)],
                    p=0.8
                ),
                transforms.RandomRotation(30),
                GaussianBlur(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
        def __call__(self, x):
            y1 = self.transform(x)
            y2 = self.transform_prime(x)
            return y1, y2
        
    
    #---
    
    df = pd.read_csv("csv_folder/train.csv")
    byol_dataset = ImageCSVDataset(df, img_dir = "dataset/batch1/train", transform=Transform()) 
    byol_dataloader = DataLoader(byol_dataset, batch_size = 32, shuffle=True,
                               num_workers=NUM_WORKERS)
    
    
    #---
    
    n = 16
    
    paired_views, labels = next(iter(byol_dataloader))
    print(len(paired_views))
    print(paired_views[0].shape)
    print(paired_views[1].shape)
    out1 = torchvision.utils.make_grid(paired_views[0][:n])
    out2 = torchvision.utils.make_grid(paired_views[1][:n])
    imshow(out1, title = [class_names[x] for x in labels[:n]])
    imshow(out2, title = [class_names[x] for x in labels[:n]])
    
    #---
    
    def get_encoder_model(**kwargs):
        resnet = torchvision.models.resnet50(**kwargs)
        resnet.fc = torch.nn.Identity() 
        return resnet
        
    
    class MLP(nn.Module):
        def __init__(self, input_dim: int, projection_dim: int=128, hidden_dim: int=512):
            super(MLP,self).__init__()
    
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim)
            )
    
        def forward(self, x):
            return self.net(x)
    
    class EncoderProjecter(nn.Module):
        def __init__(self,
                     encoder: nn.Module,
                     hidden_dim: int=512,
                     projection_out_dim: int=128
                     ) -> None:
            super(EncoderProjecter, self).__init__()
    
            self.encoder = encoder # encoder is ResNet50 with classifier head removed (Linear layer) 
            self.projection = MLP(input_dim=2048, projection_dim=projection_out_dim, hidden_dim=hidden_dim)
    
        def forward(self, x):
            h = self.encoder(x)
            h = h.view(h.shape[0], h.shape[1]) # flattens
            return self.projection(h)
        
    class BYOL(nn.Module):
        def __init__(self,
                     hidden_dim: int = 512,
                     projection_out_dim: int = 128,
                     target_decay: float = 0.996
                    ) -> None:
            super(BYOL, self).__init__()
            resnet = get_encoder_model(weights=ResNet50_Weights.IMAGENET1K_V1)
    
            # freeze everything
            for param in resnet.parameters():
                param.requires_grad = False
    
            # unfreeze layer3 + layer4
            for param in resnet.layer3.parameters():
                param.requires_grad = True
    
            for param in resnet.layer4.parameters():
                param.requires_grad = True
        
        
            self.online_network = EncoderProjecter(resnet)  # encoder + projector
            self.online_predictor = MLP(input_dim=projection_out_dim, projection_dim=projection_out_dim, hidden_dim=hidden_dim)
    
            self.target_network = copy.deepcopy(self.online_network)  # independent copy
            self.target_network.load_state_dict(self.online_network.state_dict())
            
            # set target_network's weights to be untrainable
            self.target_network.eval()
            for param in self.target_network.parameters():
                param.requires_grad = False
            self.target_decay = target_decay
            self.loss_function = NormalizedMSELoss()
    
    
        @torch.no_grad()
        def soft_update_target_network(self) -> None:
            for online_p, target_p in zip(self.online_network.parameters(), self.target_network.parameters()):
                target_p.data = target_p.data * self.target_decay + online_p.data * (1. - self.target_decay)
    
    
        def forward(self, view):
            online_proj = self.online_network(view)
            target_proj = self.target_network(view)
    
            return online_proj, target_proj
    
        def loss(self, view1, view2):
            online_proj1, target_proj1 = self(view1) # v embedding
            online_proj2, target_proj2 = self(view2) # v' embedding
    
            online_prediction_1 = self.online_predictor(online_proj1)
            online_prediction_2 = self.online_predictor(online_proj2)
    
            loss1 = self.loss_function(online_prediction_1, target_proj2.detach()) # online v prediction, target v' prediction
            loss2 = self.loss_function(online_prediction_2, target_proj1.detach()) # online v' prediction, target v prediction
            return torch.mean(loss1 + loss2)
    
    #---
    
    def save_byol_checkpoint(epoch, encoder, optimizer, scheduler, path):
        checkpoint = {
            "epoch"      : epoch,
            "encoder"      : encoder.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved → {path}")
    
    #---
    
    def train_self_supervised(model, dataloader, num_epochs=50, checkpoint_dir = 'ckpts', ckpt_save_interval = 15):
        since = time.time()
        history = {"train_loss": []}
        os.makedirs(checkpoint_dir, exist_ok = True)
        checkpoint_dir_dt = os.path.join(checkpoint_dir, f"byol_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}")
        os.makedirs(checkpoint_dir_dt, exist_ok = False)
    
        optimizer = optim.SGD(list(model.online_network.parameters()) + list(model.online_predictor.parameters()), lr=0.002, momentum=0.9)
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min=0)
    
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch, _ in dataloader:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
    
    
                view_1, view_2 = batch[0], batch[1]
    
                loss = model.loss(view_1, view_2)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.soft_update_target_network()
                
            scheduler.step()
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
            if (epoch + 1) % ckpt_save_interval == 0 or (epoch + 1) == num_epochs:
                save_byol_checkpoint(
                    epoch=epoch,
                    encoder=model.online_network.encoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    path=os.path.join(checkpoint_dir_dt, f"epoch_{epoch:03d}.pth")
                )
            history["train_loss"].append(epoch_loss)
        torch.save(history, os.path.join(checkpoint_dir_dt, "history.pth"))
        elapsed = time.time() - since
        print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        return model, history
    
    
    #---
    
    byol = BYOL()
    byol.to(device);
    
    #---
    
    trained_encoder = get_encoder_model()
    byol_encoder_path = r"models/byol_encoder.pt"
    if not os.path.exists(byol_encoder_path):
        byol, history = train_self_supervised(byol,byol_dataloader,num_epochs=100)
        trained_encoder.load_state_dict(byol.online_network.encoder.state_dict())
        torch.save(trained_encoder.state_dict(), byol_encoder_path)
    else:
        trained_encoder.load_state_dict(torch.load(byol_encoder_path, map_location=device))
    
    #---
    
    class Classifier(nn.Module):
        def __init__(self, encoder_model, frozen=False):
            super(Classifier, self).__init__()
            self.encoder = copy.deepcopy(encoder_model)
            if frozen:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            self.fc = nn.Linear(in_features=2048, out_features=2)
            
    
        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.shape[0], x.shape[1])
            x = self.fc(x)
            return x
    
    trained_encoder.to(device)
    seed_clf = Classifier(trained_encoder)
    seed_clf.to(device)
    
    #---
    
    seed_clf_save_dir = r"models/byol_tuned_seed_clf.pt"
    if not os.path.exists(seed_clf_save_dir):
        optimizer = optim.SGD(
            seed_clf.parameters(),
            lr=0.001,
            momentum=0.9
        )
        # Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        seed_clf, history = train_model(
            seed_clf,
            crtrn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=50,
            checkpoint_dir='ckpts'
        )
        torch.save(seed_clf.state_dict(), seed_clf_save_dir)
    else:
        seed_clf = Classifier(get_encoder_model())
        seed_clf.load_state_dict(torch.load(seed_clf_save_dir, map_location=device))
        
    
    #---
    
    history_byol = torch.load("ckpts/byol/history.pth")
    plt.plot(history_byol["train_loss"], linewidth=2, color='#2E86AB')
    plt.title("BYOL Training Loss", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figs/byol_train_loss.png", dpi=300, bbox_inches='tight')
    
    #---
    
    print(history_byol["train_loss"][-1])
    
    #---
    
    seed_clf.to(device)
    seed_clf_results_b1 = evaluate_model_batch1(
        model = seed_clf,
        loader = test_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_results_b1
    
    #---
    
    seed_clf_results_b2 = evaluate_model_b2b3(
        model = seed_clf,
        loader = batch2_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_results_b2
    
    #---
    
    
    plt.rcParams.update({'font.size': 16})  # Set before plotting
    
    disp = ConfusionMatrixDisplay(confusion_matrix=seed_clf_results_b2['cm'], display_labels=["Bad", "Good"])
    disp = disp.plot(cmap="Blues", values_format="d", colorbar=False)
    
    disp.figure_.savefig("figs/byol_finetuned_batch2_cm.png")
    plt.show()
    
    #---
    
    seed_clf_results_b3 = evaluate_model_b2b3(
        model = seed_clf,
        loader = batch3_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_results_b3
    
    #---
    
    
    plt.rcParams.update({'font.size': 16})
    
    disp = ConfusionMatrixDisplay(confusion_matrix=seed_clf_results_b3['cm'], display_labels=["Bad", "Good"])
    disp = disp.plot(cmap="Blues", values_format="d", colorbar=False)
    
    disp.figure_.savefig("figs/byol_finetuned_batch3_cm.png")
    plt.show()
    
    #---
    
    # tuned_encoder = copy.deepcopy(seed_clf.encoder)
    
    tuned_encoder = Classifier(get_encoder_model())
    tuned_encoder.load_state_dict(torch.load("models/byol_tuned_seed_clf.pt", map_location=device))
    tuned_encoder = tuned_encoder.encoder
    tuned_encoder.to(device)
    
    
    byol_finetuned_embeddings_b1, byol_finetuned_labels_b1 = get_encoder_embeddings(tuned_encoder, dataloader = test_loader, device = device)
    byol_finetuned_embeddings_b2, byol_finetuned_labels_b2 = get_encoder_embeddings_b2b3(tuned_encoder, dataloader = batch2_loader, device = device)
    byol_finetuned_embeddings_b3, byol_finetuned_labels_b3 = get_encoder_embeddings_b2b3(tuned_encoder, dataloader = batch3_loader, device = device)
    
    
    
    #---
    
    byol_encoder_path = r"models/byol_encoder.pt"
    trained_encoder = get_encoder_model()
    trained_encoder.load_state_dict(torch.load(byol_encoder_path, map_location=device))
    
    seed_clf_probed = Classifier(trained_encoder, frozen=True)
    seed_clf_probed.to(device)
    
    #---
    
    seed_clf_probed_save_dir = r"models/byol_probed_seed_clf.pt"
    if not os.path.exists(seed_clf_probed_save_dir):
        optimizer = optim.SGD(
            seed_clf_probed.parameters(),
            lr=0.001,
            momentum=0.9
        )
        # Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        seed_clf_probed, history = train_model(
            seed_clf_probed,
            crtrn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=150,
            checkpoint_dir='ckpts'
        )
        torch.save(seed_clf_probed.state_dict(), seed_clf_probed_save_dir)
    else:
        seed_clf_probed = Classifier(get_encoder_model())
        seed_clf_probed.to(device)
        seed_clf_probed.load_state_dict(torch.load(seed_clf_probed_save_dir, map_location=device))
        
    
    #---
    
    seed_clf_probed_results_b1 = evaluate_model_batch1(
        model = seed_clf_probed,
        loader = test_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_probed_results_b1
    
    #---
    
    seed_clf_probed_results_b2 = evaluate_model_b2b3(
        model = seed_clf_probed,
        loader = batch2_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_probed_results_b2
    
    #---
    
    seed_clf_probed_results_b3 = evaluate_model_b2b3(
        model = seed_clf_probed,
        loader = batch3_loader, 
        criterion=nn.CrossEntropyLoss(),
        device=device
    )
    seed_clf_probed_results_b3
    
    #---
    
    history_probed_byol = torch.load("ckpts/byol_linear_probe/history.pth", map_location=device)
    history_tuned_byol = torch.load("ckpts/byol_fine_tuned/history.pth", map_location=device)
    
    #---
    
    directories = [
        "ckpts/byol_linear_probe",
        "ckpts/byol_fine_tuned",
        "ckpts/ResNet50"
    ]
    plot_histories(directories, "byol_training_histories")
    
    #---
    
    cluster_assg_b1_byol, sim_scores_b1_byol = kmeans_embeddings(byol_finetuned_embeddings_b1, byol_finetuned_labels_b1, n_clusters=2)
    cluster_assg_b2_byol, sim_scores_b2_byol = kmeans_embeddings(byol_finetuned_embeddings_b2, byol_finetuned_labels_b2, n_clusters=2)
    cluster_assg_b3_byol, sim_scores_b3_byol = kmeans_embeddings(byol_finetuned_embeddings_b3, byol_finetuned_labels_b3, n_clusters=2)
    
    print(f"{'':20} {'Batch-1':>10} {'Batch-2':>10} {'Batch-3':>10}")
    print("-" * 50)
    print(f"{'ARI':20} {sim_scores_b1_byol['ari']:>10.4f} {sim_scores_b2_byol['ari']:>10.4f} {sim_scores_b3_byol['ari']:>10.4f}")
    print(f"{'NMI':20} {sim_scores_b1_byol['nmi']:>10.4f} {sim_scores_b2_byol['nmi']:>10.4f} {sim_scores_b3_byol['nmi']:>10.4f}")
    
    #---
    
    trained_encoder.to(device)
    byol_pretrained_embeddings_b1, byol_pretrained_labels_b1 = get_encoder_embeddings(trained_encoder, dataloader = test_loader, device = device)
    byol_pretrained_embeddings_b2, byol_pretrained_labels_b2 = get_encoder_embeddings_b2b3(trained_encoder, dataloader = batch2_loader, device = device)
    byol_pretrained_embeddings_b3, byol_pretrained_labels_b3 = get_encoder_embeddings_b2b3(trained_encoder, dataloader = batch3_loader, device = device)
    
    #---
    
    cluster_assg_b1_byol_pre, sim_scores_b1_byol_pre = kmeans_embeddings(byol_pretrained_embeddings_b1, byol_pretrained_labels_b1, n_clusters=2)
    cluster_assg_b2_byol_pre, sim_scores_b2_byol_pre = kmeans_embeddings(byol_pretrained_embeddings_b2, byol_pretrained_labels_b2, n_clusters=2)
    cluster_assg_b3_byol_pre, sim_scores_b3_byol_pre = kmeans_embeddings(byol_pretrained_embeddings_b3, byol_pretrained_labels_b3, n_clusters=2)
    
    print(f"{'':20} {'Batch-1':>10} {'Batch-2':>10} {'Batch-3':>10}")
    print("-" * 50)
    print(f"{'ARI':20} {sim_scores_b1_byol_pre['ari']:>10.6f} {sim_scores_b2_byol_pre['ari']:>10.6f} {sim_scores_b3_byol_pre['ari']:>10.6f}")
    print(f"{'NMI':20} {sim_scores_b1_byol_pre['nmi']:>10.6f} {sim_scores_b2_byol_pre['nmi']:>10.6f} {sim_scores_b3_byol_pre['nmi']:>10.6f}")
    
    #---
    
    
    pretrained_embeddings_b1, pretrained_labels_b1 = get_encoder_embeddings(model_rn50_ffe, dataloader = test_loader, device = device)
    pretrained_embeddings_b2, pretrained_labels_b2 = get_encoder_embeddings_b2b3(model_rn50_ffe, dataloader = batch2_loader, device = device)
    pretrained_embeddings_b3, pretrained_labels_b3 = get_encoder_embeddings_b2b3(model_rn50_ffe, dataloader = batch3_loader, device = device)
    
    #---
    
    cluster_assg_b1_pre, sim_scores_b1_pre = kmeans_embeddings(pretrained_embeddings_b1, pretrained_labels_b1, n_clusters=2)
    cluster_assg_b2_pre, sim_scores_b2_pre = kmeans_embeddings(pretrained_embeddings_b2, pretrained_labels_b2, n_clusters=2)
    cluster_assg_b3_pre, sim_scores_b3_pre = kmeans_embeddings(pretrained_embeddings_b3, pretrained_labels_b3, n_clusters=2)
    
    print(f"{'':20} {'Batch-1':>10} {'Batch-2':>10} {'Batch-3':>10}")
    print("-" * 50)
    print(f"{'ARI':20} {sim_scores_b1_pre['ari']:>10.6f} {sim_scores_b2_pre['ari']:>10.6f} {sim_scores_b3_pre['ari']:>10.6f}")
    print(f"{'NMI':20} {sim_scores_b1_pre['nmi']:>10.6f} {sim_scores_b2_pre['nmi']:>10.6f} {sim_scores_b3_pre['nmi']:>10.6f}")
    
    #---
    
    summary(model_rn50, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params", "mult_adds"])
    
    #---
    
    summary(model_rn50_ffe, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params", "mult_adds"])
    
    #---
    
    summary(byol, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params", "mult_adds"])
    
    #---
    
    summary(seed_clf_probed, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params", "mult_adds"])
    
    #---
    
    summary(seed_clf, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params", "mult_adds"])


##########################################################################
# This file was converted using nb2py: https://github.com/BardiaKh/nb2py #
##########################################################################
