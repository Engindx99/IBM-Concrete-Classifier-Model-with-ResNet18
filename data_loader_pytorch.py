import os
import zipfile
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# ----------------------------
# Extract ZIP files
# ----------------------------
negative_zip = r"C:\Users\engin\data\Negative_tensors.zip"
positive_zip = r"C:\Users\engin\data\Positive_tensors.zip"

negative_dir = r"C:\Users\engin\data\Negative_tensors"
positive_dir = r"C:\Users\engin\data\Positive_tensors"

if not os.path.exists(negative_dir):
    with zipfile.ZipFile(negative_zip, 'r') as zip_ref:
        zip_ref.extractall(negative_dir)
    print("Negative zip extracted.")
else:
    print("Negative folder already exists, zip not extracted.")

if not os.path.exists(positive_dir):
    with zipfile.ZipFile(positive_zip, 'r') as zip_ref:
        zip_ref.extractall(positive_dir)
    print("Positive zip extracted.")
else:
    print("Positive folder already exists, zip not extracted.")

# ----------------------------
# Actual folders containing .pt files
# ----------------------------
negative_file_path = r"C:\Users\engin\data\Negative_tensors\Negative_tensors"
positive_file_path = r"C:\Users\engin\data\Positive_tensors\Positive_tensors"

# ----------------------------
# List file paths (.pt)
# ----------------------------
negative_files = [os.path.join(negative_file_path, f) for f in os.listdir(negative_file_path) if f.lower().endswith(".pt")]
negative_files.sort()
print("Negative samples:", negative_files[:3])

positive_files = [os.path.join(positive_file_path, f) for f in os.listdir(positive_file_path) if f.lower().endswith(".pt")]
positive_files.sort()
print("Positive samples:", positive_files[:3])

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
    tensor = torch.load(file)           # shape: (3, 224, 224)
    tensor = tensor.permute(1, 2, 0)    # shape: (224, 224, 3) → RGB format
    plt.imshow(tensor.numpy())
    plt.title("y=" + str(y.item()))
    plt.show()

train = False
if train:
    all_files = all_files[:30000]
    Y = Y[:30000]
else:
    all_files = all_files[30000:]
    Y = Y[30000:]

# ----------------------------
# Dataset class
# ----------------------------
class ConcreteDataset(Dataset):
    def __init__(self, transform=None, train=True):
        positive_files = [os.path.join(positive_file_path, f) for f in os.listdir(positive_file_path) if f.lower().endswith(".pt")]
        positive_files.sort()
        negative_files = [os.path.join(negative_file_path, f) for f in os.listdir(negative_file_path) if f.lower().endswith(".pt")]
        negative_files.sort()

        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.transform = transform

        self.Y = torch.zeros(number_of_samples, dtype=torch.long)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        if train:
            self.Y = self.Y[:30000]
        else:
            self.Y = self.Y[30000:]

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
# Visualize Dataset sample
# ----------------------------
dataset = ConcreteDataset(train=True)

for sample in samples:
    tensor, label = dataset[sample]
    plt.imshow(tensor.numpy())
    plt.xlabel("y=" + str(label.item()))
    plt.title(f"training data, sample {int(sample)}")
    plt.show()