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

        name = os.path.join('archive/train/', self.data.iloc[index,0] + '.jpg')
        image = io.imread(name)
        #image = Image.fromarray(image)
        
        y_label = torch.tensor(int(self.data.iloc[index,7]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image,y_label)
    

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    #transforms.RandomCrop((28,28)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5)],p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=15,fill=127)],p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(3)],p=0.5),
    transforms.ToTensor()
    ])

dataset = CustomDataset('archive/augment.csv',transform = transform)

df = pd.read_csv('archive/augment.csv')




a = []
img_num = 0
for _ in range(30):
    print('a')
    for img, label in dataset:
        a.append(img)
        path = 'augmented/ISIC_AUGMENTATION_'+str(img_num)+'.jpg'
        save_image(img, path)
        df = pd.concat([pd.DataFrame([[
            'ISIC_AUGMENTATION_' + str(img_num)+'.jpg',
            'augment',
            'male',
            '40',
            'augment',
            'melanoma',
            'malignant',
            label.item()]],
            columns=df.columns), df], ignore_index=True)
        
        img_num += 1

df.to_csv('info.csv',index=False)
show(make_grid(a[:32]))