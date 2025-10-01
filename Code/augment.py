import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
from skimage import io,transform
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import shutil

""" 
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, common_transform=None, malignant_transform=None, protected_transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.common_transform = common_transform
        self.malignant_transform = malignant_transform
        self.protected_transform = protected_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_name = row['image_name']
        image_path = os.path.join(self.root_dir, image_name + '.jpg')

        image = io.imread(image_path)
        image = Image.fromarray(image)  # No cv2.cvtColor needed
        # Choose transform
        if row['benign_malignant'] == 'malignant':
            transform = self.malignant_transform
        elif row['protected'] == 1:
            transform = self.protected_transform
        else:
            transform = self.common_transform

        image = transform(image)  # transforms expect PIL Image

        # Inpainting logic
        image_np = np.asarray(image)
        # Ensure image_np is uint8 and 3-channel
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.ndim == 3 and image_np.shape[2] == 3:
            pass
        else:
            if image_np.shape[0] == 3 and image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.shape[2] != 3:
                raise ValueError("Image must have 3 channels for inpainting.")

        # Now safe to call cv2.inpaint
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 5)  # Lower threshold, or use edge detection for hairs
        # Optionally, use morphological operations to target thin lines only
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        inpainted = cv2.inpaint(image_np, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        inpainted = Image.fromarray(inpainted)
        inpainted = self.to_tensor(inpainted)

        y_label = torch.tensor(int(1 if 'benign' in row['benign_malignant'] else 0))

        return (inpainted, y_label) """

train = pd.read_csv('train_metadata.csv')
test = pd.read_csv('test_metadata.csv')

common_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2)], p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    transforms.ToTensor()
])

# Define malignant-specific augmentations
malignant_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2)], p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5)
])

# Define protected class (skin cancer on people of color) augmentations
protected_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2)], p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=256, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, scale=(0.01, 0.05)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5)
])


""" train_dataset = CustomDataset(csv_file='train_metadata.csv',
                               root_dir='train/',
                               common_transform=common_transforms,
                               malignant_transform=malignant_transform,
                               protected_transform=protected_transform)

test_dataset = CustomDataset(csv_file='test_metadata.csv',
                              root_dir='merge/',
                              common_transform=common_transforms,
                              malignant_transform=malignant_transform,
                              protected_transform=protected_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0) """


""" for images, labels in train_loader:
    print(images.shape, labels.shape)
    # Here you can visualize or save the images
    grid = make_grid(images, nrow=8, padding=2)
    save_image(grid, 'train_batch.png')
    break  # Remove this to process all batches """
def inpaint_image(tensor_img):
    # Convert tensor to numpy array and transpose to HWC
    image_np = tensor_img.mul(255).byte().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # C,H,W -> H,W,C

    # Ensure 3 channels
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.ndim == 3 and image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.ndim == 3 and image_np.shape[2] == 3:
        pass
    else:
        if image_np.shape[0] == 3 and image_np.shape[2] != 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.shape[2] != 3:
            raise ValueError("Image must have 3 channels for inpainting.")

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 0)  # Lower threshold, or use edge detection for hairs
    # Optionally, use morphological operations to target thin lines only
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted = cv2.inpaint(image_np, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted = Image.fromarray(inpainted)
    inpainted_tensor = transforms.ToTensor()(inpainted)
    return inpainted_tensor



os.makedirs('augmented_malignant_train', exist_ok=True)
malignant_df = train[(train['benign_malignant'] == 'malignant') & (~train['image_name'].str.contains('aug'))]
print(malignant_df.shape)
# Augment malignant cases in train
for _, row in malignant_df.iterrows():

    image_name = row['image_name']
    image_path = os.path.join('train', image_name + '.jpg')
    if not os.path.exists(image_path):
        continue
    image = Image.open(image_path).convert('RGB')
    for i in range(50):
        aug_image = malignant_transform(image)
        aug_image = inpaint_image(aug_image)
        save_path = os.path.join('augmented_malignant_train', f"{image_name}_aug{i}V2.jpg")
        save_image(aug_image, save_path)

# Augment malignant cases in test
malignant_test_df = test[(test['benign_malignant'] == 'malignant') & (~test['image_name'].str.contains('aug'))]
os.makedirs('augmented_malignant_test', exist_ok=True)
print(malignant_test_df.shape)
for _, row in malignant_test_df.iterrows():

    image_name = row['image_name']
    image_path = os.path.join('test', image_name + '.jpg')
    if not os.path.exists(image_path):
        continue
    image = Image.open(image_path).convert('RGB')
    for i in range(85):
        aug_image = malignant_transform(image)
        aug_image = inpaint_image(aug_image)
        save_path = os.path.join('augmented_malignant_test', f"{image_name}_aug{i}V2.jpg")
        save_image(aug_image, save_path)
""" 
# Augment protected cases in train
protected_train_df = train[train['protected'] == 1]
os.makedirs('augmented_protected_train', exist_ok=True)
for _, row in protected_train_df.iterrows():
    image_name = row['image_name']
    image_path = os.path.join('train', image_name + '.jpg')
    if not os.path.exists(image_path):
        continue
    image = Image.open(image_path).convert('RGB')
    for i in range(800):
        aug_image = protected_transform(image)
        aug_image = inpaint_image(aug_image)
        save_path = os.path.join('augmented_protected_train', f"{image_name}_aug{i}.jpg")
        save_image(aug_image, save_path)

# Augment protected cases in test
protected_test_df = test[test['protected'] == 1]
os.makedirs('augmented_protected_test', exist_ok=True)
for _, row in protected_test_df.iterrows():
    image_name = row['image_name']
    image_path = os.path.join('merge', image_name + '.jpg')
    if not os.path.exists(image_path):
        continue
    image = Image.open(image_path).convert('RGB')
    for i in range(1000):
        aug_image = protected_transform(image)
        aug_image = inpaint_image(aug_image)
        save_path = os.path.join('augmented_protected_test', f"{image_name}_aug{i}.jpg")
        save_image(aug_image, save_path) """