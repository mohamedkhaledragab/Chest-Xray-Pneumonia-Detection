# Chest X-Ray Pneumonia Detection

import os
import random
import io

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import (
    DataLoader, WeightedRandomSampler,
    Subset, ConcatDataset
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import kagglehub

# 2. CONFIG  (change only here)
SEED        = 42
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 3e-4
PATIENCE    = 5
VAL_SIZE    = 58
TEST_SIZE   = 862
IMG_SIZE    = 224
SAVE_PATH   = os.getenv("MODEL_PATH", "xray_classifier.pth")
CLASSES     = ["NORMAL", "PNEUMONIA"]
TTA_STEPS   = 5

# Make sure output directory exists
os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else ".", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 3. DOWNLOAD DATASET

dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

train_dir = os.path.join(dataset_path, "chest_xray", "train")
val_dir   = os.path.join(dataset_path, "chest_xray", "val")
test_dir  = os.path.join(dataset_path, "chest_xray", "test")

for p in [train_dir, val_dir, test_dir]:
    status = "True" if os.path.exists(p) else "False"
    print(f"{status} {p}")


# 4. DATASET VISUALIZATION

raw_dataset = datasets.ImageFolder(train_dir)

fig, axes = plt.subplots(4, 6, figsize=(16, 10))
sample_idx = random.sample(range(len(raw_dataset)), 24)

for ax, idx in zip(axes.flat, sample_idx):
    img, label = raw_dataset[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(raw_dataset.classes[label], fontsize=9)
    ax.axis("off")

plt.suptitle("Sample X-Ray Images", fontsize=14)
plt.tight_layout()
plt.show()

# 5. EXPLORATORY ANALYSIS

counts = [0, 0]
for _, label in raw_dataset:
    counts[label] += 1

plt.figure(figsize=(6, 4))
bars = plt.bar(CLASSES, counts, color=["steelblue", "salmon"])
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 30, str(cnt),
             ha="center", fontsize=11)
plt.title("Class Distribution (Train folder)")
plt.ylabel("Count")
plt.show()
print("Classes:", CLASSES, "| Counts:", counts)

widths, heights = [], []
for img_path, _ in raw_dataset.samples:
    w, h = Image.open(img_path).size
    widths.append(w)
    heights.append(h)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=25, color="steelblue")
axes[0].set_title("Width Distribution")
axes[0].set_xlabel("Pixels")
axes[1].hist(heights, bins=25, color="salmon")
axes[1].set_title("Height Distribution")
axes[1].set_xlabel("Pixels")
plt.tight_layout()
plt.show()

print(f"Width  → min: {min(widths)}, max: {max(widths)}")
print(f"Height → min: {min(heights)}, max: {max(heights)}")

all_pixels = []
for i in random.sample(range(len(raw_dataset)), 120):
    img, _ = raw_dataset[i]
    all_pixels.append(np.array(img.convert("L")).ravel())

pixel_data = np.concatenate(all_pixels)
plt.figure(figsize=(10, 5))
plt.hist(pixel_data, bins=60, color="gray", edgecolor="black", alpha=0.8)
plt.title("Pixel Intensity Distribution (sample of 120 images)")
plt.xlabel("Pixel Value  (0=Black → 255=White)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.show()

# 6. TRANSFORMS

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

tta_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# 7. DATASET SPLIT

ds_train = datasets.ImageFolder(train_dir, transform=train_transform)
ds_val   = datasets.ImageFolder(val_dir,   transform=test_transform)
ds_test  = datasets.ImageFolder(test_dir,  transform=test_transform)

full_pool  = ConcatDataset([ds_train, ds_val, ds_test])
all_idx    = list(range(len(full_pool)))
random.shuffle(all_idx)

train_size = len(all_idx) - VAL_SIZE - TEST_SIZE

train_indices = all_idx[:train_size]
val_indices   = all_idx[train_size : train_size + VAL_SIZE]
test_indices  = all_idx[train_size + VAL_SIZE:]

train_dataset = Subset(full_pool, train_indices)
val_dataset   = Subset(full_pool, val_indices)
test_dataset  = Subset(full_pool, test_indices)

print(f"\nTotal images : {len(full_pool)}")
print(f"Train        : {len(train_dataset)}")
print(f"Validation   : {len(val_dataset)}")
print(f"Test         : {len(test_dataset)}")

# 8. WEIGHTED SAMPLER  (handle imbalance)

def get_labels(subset: Subset) -> list:
    """Extract labels from a Subset (works with ConcatDataset inside)."""
    labels = []
    for idx in subset.indices:
        cumulative = 0
        for ds in subset.dataset.datasets:
            if idx < cumulative + len(ds):
                local_idx = idx - cumulative
                _, label = ds.samples[local_idx]
                labels.append(label)
                break
            cumulative += len(ds)
    return labels

targets      = get_labels(train_dataset)
class_count  = np.bincount(targets)
weights      = 1.0 / class_count
sample_w     = [weights[t] for t in targets]
sampler      = WeightedRandomSampler(sample_w, len(sample_w))

print(f"\nClass counts in train → NORMAL: {class_count[0]}, PNEUMONIA: {class_count[1]}")

# 9. DATALOADERS

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    sampler=sampler, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True
)
print("Dataloaders is ready.")

# 10. MODEL  (ResNet18 — partial fine-tune)

def build_model(device: torch.device) -> nn.Module:
    
    model = models.resnet18(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False
      
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features     
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128, 2),
    )

    return model.to(device)


model = build_model(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# 11. LOSS, OPTIMIZER, SCHEDULER

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW([
    {"params": model.layer4.parameters(), "lr": LR * 0.1},   
    {"params": model.fc.parameters(),     "lr": LR},          
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

print("Optimizer & Scheduler are ready.")

# 12. TRAINING LOOP

best_val_acc = 0.0
patience_counter = 0
train_losses, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    #Train
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    #Validate
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100.0 * correct / total
    val_accs.append(val_acc)
    scheduler.step()

    current_lr = optimizer.param_groups[-1]["lr"]
    print(f"Epoch {epoch:02d}/{EPOCHS} │ "
          f"Loss: {train_loss:.4f} │ "
          f"Val Acc: {val_acc:.2f}% │ "
          f"LR: {current_lr:.2e}")

    #Checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Best model saved  ({best_val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# 13. TRAINING CURVES

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, marker="o", color="steelblue")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(val_accs, marker="o", color="salmon")
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 14. TEST EVALUATION  (standard)

model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

print("\n── Standard Test Evaluation ──")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# 15. TEST-TIME AUGMENTATION (TTA)

tta_dataset = Subset(
    ConcatDataset([
        datasets.ImageFolder(train_dir, transform=tta_transform),
        datasets.ImageFolder(val_dir,   transform=tta_transform),
        datasets.ImageFolder(test_dir,  transform=tta_transform),
    ]),
    test_indices,
)
tta_loader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

model.eval()
tta_probs_sum = np.zeros(len(test_dataset))
tta_labels    = np.zeros(len(test_dataset), dtype=int)

for step in range(TTA_STEPS):
    probs_step = []
    lbls_step  = []
    with torch.no_grad():
        for images, labels in tta_loader:
            images = images.to(device)
            probs  = torch.softmax(model(images), dim=1)[:, 1]
            probs_step.extend(probs.cpu().numpy())
            lbls_step.extend(labels.numpy())
    tta_probs_sum += np.array(probs_step)
    if step == 0:
        tta_labels = np.array(lbls_step)

tta_probs = tta_probs_sum / TTA_STEPS
tta_preds = (tta_probs >= 0.5).astype(int)

print(f"\n── TTA Evaluation ({TTA_STEPS} passes) ──")
print(classification_report(tta_labels, tta_preds, target_names=CLASSES))

# 16. CONFUSION MATRIX

cm = confusion_matrix(tta_labels, tta_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (TTA)")
plt.show()

# 17. ROC CURVE

auc = roc_auc_score(tta_labels, tta_probs)
fpr, tpr, _ = roc_curve(tta_labels, tta_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (TTA)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print(f"ROC-AUC: {auc:.4f}")

# 18. MISCLASSIFICATION ANALYSIS

model.eval()
misclassified_imgs, true_lbls, pred_lbls = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = torch.max(model(images), 1)
        mask = preds != labels
        if mask.any():
            misclassified_imgs.extend(images[mask].cpu())
            true_lbls.extend(labels[mask].cpu())
            pred_lbls.extend(preds[mask].cpu())

if misclassified_imgs:
    n = min(5, len(misclassified_imgs))
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    for i, ax in enumerate(axes):
        img = misclassified_imgs[i].permute(1, 2, 0)
        img = img * torch.tensor(STD) + torch.tensor(MEAN)
        ax.imshow(img.clamp(0, 1))
        ax.set_title(
            f"True: {CLASSES[true_lbls[i]]}\nPred: {CLASSES[pred_lbls[i]]}",
            color="red"
        )
        ax.axis("off")
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    plt.show()

# 19. EXAMPLE PREDICTIONS

model.eval()
fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for ax in axes.flat:
    idx = random.randint(0, len(test_dataset) - 1)
    img, label = test_dataset[idx]
    input_img  = img.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_img), dim=1)
        pred  = torch.argmax(probs, 1).item()
        conf  = probs[0, pred].item()

    img_show = img.permute(1, 2, 0) * torch.tensor(STD) + torch.tensor(MEAN)
    ax.imshow(img_show.clamp(0, 1))
    color = "green" if pred == label else "red"
    ax.set_title(
        f"True: {CLASSES[label]}\nPred: {CLASSES[pred]} ({conf:.2%})",
        color=color, fontsize=8
    )
    ax.axis("off")

plt.suptitle("Example Predictions  (green=correct, red=wrong)", fontsize=12)
plt.tight_layout()
plt.show()

print("\n Training & Evaluation complete.")
print(f"   Best val acc : {best_val_acc:.2f}%")
print(f"   ROC-AUC (TTA): {auc:.4f}")
print(f"   Model saved to: {SAVE_PATH}")
