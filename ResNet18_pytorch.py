import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class ConcreteDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x.to(device), y.to(device)

# --------------------------------------------------
# DATA PATHS
# --------------------------------------------------
positive_dir = r"C:\Users\engin\data\Positive_tensors\Positive_tensors"
negative_dir = r"C:\Users\engin\data\Negative_tensors\Negative_tensors"

positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith(".pt")]
negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith(".pt")]

files = positive_files + negative_files
labels = [1]*len(positive_files) + [0]*len(negative_files)

# --------------------------------------------------
# TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# --------------------------------------------------
indices = np.random.permutation(len(files))
files = np.array(files)[indices]
labels = np.array(labels)[indices]

n_total = len(files)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)

train_files = files[:n_train]
train_labels = labels[:n_train]

val_files = files[n_train:n_train+n_val]
val_labels = labels[n_train:n_train+n_val]

test_files = files[n_train+n_val:]
test_labels = labels[n_train+n_val:]

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15)
])

val_transform = None
test_transform = None

# --------------------------------------------------
# DATASETS & LOADERS
# --------------------------------------------------
train_dataset = ConcreteDataset(train_files, train_labels, train_transform)
val_dataset   = ConcreteDataset(val_files, val_labels, val_transform)
test_dataset  = ConcreteDataset(test_files, test_labels, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

model = model.to(device)

# --------------------------------------------------
# LOSS & OPTIMIZER
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001,
    weight_decay=1e-4
)

# --------------------------------------------------
# TRAINING (3 EPOCHS)
# --------------------------------------------------
n_epochs = 3
train_losses = []
val_accuracies = []

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # VALIDATION
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_accuracies.append(acc)

    print(
        f"Epoch [{epoch+1}/{n_epochs}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {acc:.4f} | "
        f"P: {precision_score(all_labels, all_preds):.4f} | "
        f"R: {recall_score(all_labels, all_preds):.4f} | "
        f"F1: {f1_score(all_labels, all_preds):.4f}"
    )

# --------------------------------------------------
# TEST EVALUATION (REAL GENERALIZATION)
# --------------------------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print("\n--- TEST RESULTS ---")
print("Test Accuracy :", np.mean(np.array(all_preds) == np.array(all_labels)))
print("Test Precision:", precision_score(all_labels, all_preds))
print("Test Recall   :", recall_score(all_labels, all_preds))
print("Test F1-score :", f1_score(all_labels, all_preds))

# --------------------------------------------------
# PLOTS
# --------------------------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(val_accuracies, marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()