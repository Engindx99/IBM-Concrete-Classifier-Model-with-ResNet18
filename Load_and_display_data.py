import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# ----------------------------
# Actual folders containing .pt files
# ----------------------------
negative_dir = r"C:\Users\engin\data\Negative_tensors\Negative_tensors"
positive_dir = r"C:\Users\engin\data\Positive_tensors\Positive_tensors"

# ----------------------------
# List file paths (.pt)
# ----------------------------
negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.lower().endswith(".pt")]
negative_files.sort()
print("Negative samples:", negative_files[:3])

positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.lower().endswith(".pt")]
positive_files.sort()
print("Positive samples:", positive_files[:3])

# ----------------------------
# Combine all files
# ----------------------------
all_files = positive_files + negative_files
number_of_samples = len(all_files)
samples = range(4)  # Samples for visualization

# ----------------------------
# Create labels
# ----------------------------
Y = torch.zeros(number_of_samples, dtype=torch.long)
Y[::2] = 1  # Positive
Y[1::2] = 0  # Negative

# ----------------------------
# Visualize sample (.pt tensor)
# ----------------------------
for y, file in zip(Y, all_files[:4]):
    tensor = torch.load(file)          # shape: (3, H, W)
    tensor = tensor.permute(1, 2, 0)   # shape: (H, W, 3) → RGB format
    plt.imshow(tensor.numpy())
    plt.title("y=" + str(y.item()))
    plt.show()

# ----------------------------
# Dataset class
# ----------------------------
class ConcreteDataset(Dataset):
    def __init__(self, transform=None):
        positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.lower().endswith(".pt")]
        positive_files.sort()
        negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.lower().endswith(".pt")]
        negative_files.sort()

        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.transform = transform

        self.Y = torch.zeros(number_of_samples, dtype=torch.long)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        self.len = len(self.all_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tensor = torch.load(self.all_files[idx])
        tensor = tensor.permute(1, 2, 0)  # shape: (H, W, C)
        y = self.Y[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, y

# ----------------------------
# Dataset sample visualization
# ----------------------------
dataset = ConcreteDataset()

for sample in samples:
    tensor, label = dataset[sample]
    plt.imshow(tensor.numpy())
    plt.xlabel("y=" + str(label.item()))
    plt.title(f"Sample {sample}")
    plt.show()