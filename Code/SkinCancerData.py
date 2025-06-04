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
    def __init__(self,csv,transform=None):
        self.csv = pd.read_csv(csv)
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        path = os.path.join('archive/train',str(self.csv.iloc[index].iloc[0] + '.jpg'))
        image = Image.open(path)
        label = self.csv.iloc[index].iloc[7]
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label)
        
        return image,label,label
    

def CreateLoader(path,transform,batch_size):
    data = CustomDataset(path,transform=transform)   
    train, test = torch.utils.data.random_split(data, [0.8, 0.2])
    train_loader = DataLoader(train,batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(data,batch_size=batch_size, shuffle = False)
    
    return train_loader,test_loader
    
    

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.ToPILImage()])
# im , _ , _ = data[0]

# img = transform(im)
# print(type(img))
# img.show()








