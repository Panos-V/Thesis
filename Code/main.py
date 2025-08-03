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



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def adversarial_walk(f,h,a,model,device,steps = 5):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    e = 1e-12
    for i in range(steps):
        prediction = f(h_delta)
        prediction = torch.softmax(prediction, dim=1)
        entropy = -torch.special.entr(prediction + e).sum(dim=1).mean()
        gradient = torch.autograd.grad(entropy, h_delta, retain_graph=True)[0]


        delta = (gradient - gradient.mean()) / (gradient.std() + e)    

        h_delta = h_delta + a*delta

        _,h_delta,perplexity,_ = model.vq(h_delta)

        
        h_delta = h_delta.requires_grad_(True)

    #print(h_delta)

    return h_delta,perplexity



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256))
    ])


batch_size = 32

colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=8)
colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Code/archive/"
skin_train,skin_test = SkinCancerData.CreateLoader(path, transform, batch_size)



ALPHA = 0.07
TRAIN = False
Train_f = False
LOAD_VQ = True
LOAD_F = False

epochs = 100

num_hiddens = 512
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 2056
commitment_cost = 0.35
decay = 0.99
learning_rate = 0.001



vq = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost,device).to(device)

optimizer = optim.Adam(vq.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

to_PIL = transforms.ToPILImage()

if TRAIN:

    vq_vae.train_model(vq,epochs, optimizer, criterion, skin_train,load=LOAD_VQ)

else:
    vq.load_state_dict(torch.load("vqvae.pth",weights_only=False))
    
    (im,label,_) = next(iter(skin_train))

    
    image = im.to(device)
    label = label.to(device) 
    _,recon,_ = vq(image)
    
    images = make_grid(image[:32])
    outputs = make_grid(recon[:32])

    images = to_PIL(images)
    images.save('original2.png')
    outputs = to_PIL(outputs)
    outputs.save('recons.png')

f = simple_classifier.classifier(64*64*64,num_classes= 2 ,device=device).to(device)   #embdeding dimension X Height/4 X Width/4

f_optimizer = optim.SGD(f.parameters(),lr = 0.001, momentum=0.9)
f_criterion = nn.CrossEntropyLoss()
epochs_f = 10

if Train_f:

    simple_classifier.train_classifier(vq,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, skin_train,load=LOAD_F)


else:
    f.load_state_dict(torch.load("classifier.pth",weights_only=False))
    f.eval()
    vq.eval()
    
    (im,label,_) = next(iter(skin_test))
    image = im.to(device)
    label = label.to(device)
    images = make_grid(image[:32])

    h = vq.encoder(image)
    h = vq.pre_vq_conv(h)
    output,perplexity = adversarial_walk(f, h, ALPHA,vq,device)
    recon = vq.decoder(output)
    
    outputs = make_grid(recon[:32])

    images = to_PIL(images)
    images.save('original.png')
    outputs = to_PIL(outputs)
    outputs.save('outputs.png')
torch.cuda.empty_cache()
res = resnet.create_model(vq,f,skin_train, skin_test, 'biased_resnet20_a007.pth',adversarial=False, ALPHA=ALPHA)
res = resnet.inference('bias_resnet20.pth',vq,f,skin_test,transform,adversarial=False, ALPHA=ALPHA)
print(res)
