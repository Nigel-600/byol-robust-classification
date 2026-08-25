import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights

from utils.visualisation import visualize_model

import os
import random
import time
import copy
import datetime

from config import DEVICE as device

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

def train_model(model, all_dataloaders, crtrn, optimizer=None, scheduler=None, num_epochs=25, ckpt_save_interval=20, checkpoint_dir = "ckpts", history=None):
    dataset_sizes = dict(zip(all_dataloaders.keys(), list(map(len, all_dataloaders.values()))))

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


def get_baseline_models(model_name = "resnet50", none_weights = False):
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


def ensure_model(model_name, device, num_epochs=150, lr=0.001, momentum=0.9, step_size=10, gamma=0.1, model_dir="models"):
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    criterion = nn.CrossEntropyLoss()

    # if no checkpoint -> create, train and save
    if not os.path.exists(model_path):
        model = get_baseline_models(model_name=model_name)
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
        visualize_model(model)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    else:
        # load into a fresh model constructed with no weights (so architecture matches)
        model = get_baseline_models(model_name=model_name, none_weights=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = True

    return model

def save_byol_checkpoint(epoch, encoder, optimizer, scheduler, path):
    checkpoint = {
        "epoch"      : epoch,
        "encoder"      : encoder.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "scheduler"  : scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"  ✓ Checkpoint saved → {path}")
    
    
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