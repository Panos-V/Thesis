import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CustomDataset(Dataset):
    def __init__(self,path,mode,transform=None):
        self.mode = mode
        self.path = path
        self.csv_path = path + mode + "_metadata.csv"
        self.csv = pd.read_csv(self.csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        path = os.path.join(str(self.path) + self.mode+ '/' + str(self.csv.iloc[index].iloc[0] + '.jpg'))
        image = Image.open(path).convert("RGB")  # Ensure image is RGB
        label = 1 if self.csv.iloc[index].iloc[5] == 'malignant' else 0
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label)
        
        return image,label,label
    

def CreateLoader(path,transform,batch_size):
    train = CustomDataset(path,'train',transform=transform)   
    test = CustomDataset(path,'test',transform=transform)
    train_loader = DataLoader(train,batch_size=batch_size, shuffle = True,num_workers=12,pin_memory=True)
    test_loader = DataLoader(test,batch_size=batch_size, shuffle = True,num_workers=12,pin_memory=True)
    
    return train_loader,test_loader
    
    

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.ToPILImage()])
# im , _ , _ = data[0]

# img = transform(im)
# print(type(img))
# img.show()








