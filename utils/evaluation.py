import numpy as np
import torch
import math

import matplotlib.pyplot as plt


from sklearn.metrics import (
         accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        adjusted_rand_score,
        normalized_mutual_info_score
    )

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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