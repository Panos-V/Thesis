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

def adversarial_walk(f,h,a,model,device,steps = 2):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    for p in f.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    e = 1e-12
    for i in range(steps): 
        prediction = f(h_delta)
        prediction = torch.softmax(prediction, dim=1)
        entropy = -torch.special.entr(prediction + e).sum(dim=1).mean()
        gradient = torch.autograd.grad(entropy, h_delta,create_graph=False)[0]


        delta = (gradient - gradient.mean()) / (gradient.std() + e)    

        h_delta = (h_delta + a*delta).detach().requires_grad_(True)
    _,h_delta,perplexity,_ = model.vq(h_delta)



    torch.cuda.empty_cache()
    return h_delta,perplexity

torch.set_float32_matmul_precision('high')
transform = transforms.Compose([
    transforms.ToTensor()
#    transforms.Resize((256,256))
    ])


batch_size = 48

#colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=8)
#colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Code/archive/"
skin_train,skin_test = SkinCancerData.CreateLoader(path, transform, batch_size)



ALPHA = 0.1
TRAIN = False
Train_f = False
LOAD_VQ = True
LOAD_F = False
INFER = True

epochs = 300

num_hiddens = 512
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 2056
commitment_cost = 0.35
decay = 0.99
learning_rate = 0.0001



vq = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost,device).to(device)

optimizer = optim.Adam(vq.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

to_PIL = transforms.ToPILImage()

if TRAIN:

    vq_vae.train_model(vq,epochs, optimizer, criterion, skin_train,load=LOAD_VQ)

elif INFER:
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

f = simple_classifier.classifier(embedding_dim*64*64,2,device).to(device)

f_optimizer = optim.Adam(f.parameters(),lr = 1e-3)
f_criterion = nn.CrossEntropyLoss()
epochs_f = 50
torch.cuda.empty_cache()
if Train_f:

    simple_classifier.train_classifier(vq,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, skin_train,load=LOAD_F)


elif INFER:
    f.load_state_dict(torch.load("classifier.pth",weights_only=False))
    f.eval()
    vq.eval()
    
    (im,label,_) = next(iter(skin_test))
    image = im.to(device)
    label = label.to(device)
    images = make_grid(image[32:64])

    h = vq.encoder(image)
    h = vq.pre_vq_conv(h)
    output,perplexity = adversarial_walk(f, h, ALPHA,vq,device)
    recon = vq.decoder(output)
    
    outputs = make_grid(recon[32:64])

    images = to_PIL(images)
    images.save('original.png')
    outputs = to_PIL(outputs)
    outputs.save('outputs.png')
torch.cuda.empty_cache()


name = 'unbiased_resnet150A0005_100'
vq.load_state_dict(torch.load("vqvae.pth",weights_only=False))
f.load_state_dict(torch.load("classifier.pth",weights_only=False))
res = resnet.create_model(vq,f,skin_train, skin_test,fair=False,epoch_head=5,epoch_tune=100,
                        patience=10,name=name,adversarial=True, ALPHA=0.005)
torch.cuda.empty_cache()
""" resnet.create_model(vq,f,skin_train,skin_test,epoch_head=5,epoch_tune=100,
                        patience=10,name='unbiased_efficientnetb5A001_70',adversarial=True, ALPHA=0.01)
resnet.create_model(vq,f,skin_train,skin_test,epoch_head=5,epoch_tune=100,
                        patience=10,name='unbiased_efficientnetb5A01_70',adversarial=True, ALPHA=0.1)
resnet.create_model(vq,f,skin_train,skin_test,epoch_head=5,epoch_tune=100,
                        patience=10,name='biased_efficientnetb5A0085_70',adversarial=True, ALPHA=0.085) """
""" res = models.efficientnet_b5(weights=None)
res.classifier[1] = nn.Linear(res.classifier[1].in_features, 2).to(device)
res.load_state_dict(torch.load(name,weights_only=False)) """

res = resnet.inference(res,name,vq,f,skin_test,transform,adversarial=False, ALPHA=ALPHA)


