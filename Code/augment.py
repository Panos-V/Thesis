import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
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


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class CustomDataset(Dataset):
    def __init__(self,csv_file,transform = None):
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        name = os.path.join('Code/archive/train/', self.data.iloc[index,0] + '.jpg')
        image = io.imread(name)
        #image = Image.fromarray(image)
        
        y_label = torch.tensor(int(self.data.iloc[index,7]))
        
        if self.transform:
            image = self.transform(image)
        image = np.asarray(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 10)  # Detect nearly black pixels (assumed gap)

        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        inpainted = Image.fromarray(inpainted)
        tensor = transforms.ToTensor()
        inpainted = tensor(inpainted)
        return (inpainted,y_label)
    

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5)],p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=10)],p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(3)],p=0.5)
    ])

df = pd.read_csv('Code/archive/original.csv')
print(df)
augment = df[df['target'] == 1]
augment.to_csv('Code/archive/augment.csv',index=False)

dataset = CustomDataset('Code/archive/augment.csv',transform = transform)


img_num = 0
for _ in range(30):
    print('a')
    for img, label in dataset:
        path = 'Code/augmented/ISIC_AUGMENTATION_'+str(img_num)+'.jpg'
        save_image(img, path)
        df = pd.concat([pd.DataFrame([[
            'ISIC_AUGMENTATION_' + str(img_num),
            'augment',
            'male',
            '40',
            'augment',
            'melanoma',
            'malignant',
            label.item()]],
            columns=df.columns), df], ignore_index=True)
        
        img_num += 1

df.to_csv('Code/archive/metadata_with_augments.csv',index=False)