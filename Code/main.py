import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision import models

import ColorMnist
import vq_vae
import simple_classifier
import SkinCancerData
import resnet

plt.ion()  # Interactive mode on for matplotlib

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def adversarial_walk(f,h,a,model,device,steps = 4):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    e = 1e-9
    for i in range(steps):

        prediction = f(h_delta)
        entropy = -torch.special.entr(prediction).sum(dim=1).mean()

        gradient = torch.autograd.grad(entropy, h_delta)[0]

        delta = (gradient - gradient.mean()) / (gradient.std() + e)    

        h_delta = h_delta + a*delta

        _,h_delta,perplexity,_ = model.vq(h_delta)
        print(f"Step {i}: gradient mean {gradient.mean():.3e}, std {gradient.std():.3e}")
        
        h_delta = h_delta.requires_grad_(True)

    #print(h_delta)

    return h_delta,perplexity



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256))
    ])


batch_size = 64

colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=8)
colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Code/archive/"
skin_train,skin_test = SkinCancerData.CreateLoader(path, transform, batch_size)



ALPHA = 0.03
TRAIN = True
Train_f = True
LOAD_VQ = False
LOAD_F = False

epochs = 300

num_hiddens = 512
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 2056
commitment_cost = 0.35
decay = 0.99
learning_rate = 0.00001



model = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost,device).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

to_PIL = transforms.ToPILImage()

if TRAIN:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    vq_vae.train_model(model,epochs, optimizer, criterion, skin_train,load=LOAD_VQ, scheduler=scheduler)

else:
    model.load_state_dict(torch.load("vqvae.pth",weights_only=False))
    
    (im,label,_) = next(iter(skin_train))

    
    image = im.to(device)
    label = label.to(device) 
    _,recon,_ = model(image)
    
    images = make_grid(image[:32])
    outputs = make_grid(recon[:32])

    images = to_PIL(images)
    images.save('original2.png')
    outputs = to_PIL(outputs)
    outputs.save('recons.png')

f = simple_classifier.classifier(64*64*64,num_classes= 2 ,device=device).to(device)   #embdeding dimension X Height/4 X Width/4

f_optimizer = optim.SGD(f.parameters(),lr = 1e-2)
f_criterion = nn.CrossEntropyLoss()
epochs_f = 10

if Train_f:
    scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer, step_size=5, gamma=0.1)
    simple_classifier.train_classifier(model,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, scheduler, skin_train,load=LOAD_F)


else:
    f.load_state_dict(torch.load("classifier.pth",weights_only=False))
    f.eval()
    model.eval()
    
    (im,label,_) = next(iter(skin_test))
    image = im.to(device)
    label = label.to(device)
    images = make_grid(image[:32])

    h = model.encoder(image)
    h = model.pre_vq_conv(h)
    output,perplexity = adversarial_walk(f, h, ALPHA,model,device)
    recon = model.decoder(output)
    
    outputs = make_grid(recon[:32])

    images = to_PIL(images)
    images.save('original.png')
    outputs = to_PIL(outputs)
    outputs.save('outputs.png')



