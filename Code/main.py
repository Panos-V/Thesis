import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import ColorMnist
import vq_vae
import simple_classifier
import SkinCancerData


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def adversarial_walk(f,h,a,model,device,steps = 6):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    e = 1e-8
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


# training_data = datasets.MNIST(root="data", train=True, download=True,
#                                   transform = transform)

# validation_data = datasets.MNIST(root="data", train=False, download=True,
#                                   transform = transform)

batch_size = 32

colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=0)
colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Code/archive/"
skin_train,skin_test = SkinCancerData.CreateLoader(path, transform, batch_size)



ALPHA = 0.1
TRAIN = True
Train_f = True

epochs = 5

num_hiddens = 512
num_residual_hiddens = 32
num_residual_layers = 4
embedding_dim = 64
num_embeddings = 2056
commitment_cost = 0.5
decay = 0.99
learning_rate = 1e-4
f_neurons = int(num_hiddens / 8)


model = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost,device).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

if TRAIN:
    vq_vae.train_model(model,epochs, optimizer, criterion, skin_train)

else:
    model.load_state_dict(torch.load("vqvae.pth",weights_only=False))
    
    (im,label,_) = next(iter(skin_test))

    
    image = im.to(device)
    label = label.to(device) 
    """    
    imshow(make_grid(image[:32]))

    _,recon,_ = model(im)
    
    imshow(make_grid(recon[:32])) """

f = simple_classifier.classifier(f_neurons*f_neurons*f_neurons,device).to(device)

f_optimizer = optim.SGD(f.parameters(),lr = 1e-2)
f_criterion = nn.BCEWithLogitsLoss()
epochs_f = 5

if Train_f:
    simple_classifier.train_classifier(model,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, skin_train)


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

    to_PIL = transforms.ToPILImage()

    images = to_PIL(images)
    images.save('original.png')
    outputs = to_PIL(outputs)
    outputs.save('outputs.png')
