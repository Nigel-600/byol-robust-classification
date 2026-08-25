import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch, torchvision
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import ConfusionMatrixDisplay
import xml.etree.ElementTree as ET
import math
from PIL import Image, ImageDraw
import os
import csv

from config import CLASS_NAMES as class_names
from config import DEVICE


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

def visualize_model(model, all_dataloaders, device, num_images=6):
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
    
    
def plot_cases_grad_eigen_cam(images, preds, labels, probs, model, target_layer, title, 
                                class_names=None, max_images=32, ncols=4, eigen=False, minimal=False, device=DEVICE):
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


def plot_confusion_matrix(cm, ax, class_names=("Negative", "Positive"), title="Confusion Matrix"):
    cm = np.array(cm, dtype=np.int64)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    ax.set_title(title)
    
    
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
    
    
def show_batch_grid(loader, class_names, n=8, figsize=(16, 4)):
    images, paths, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(images[:n])
    print(paths[:n])
    imshow(grid, title=[class_names[x] for x in labels[:n]])