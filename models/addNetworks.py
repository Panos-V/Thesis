import torch
import torch.functional as F
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, efficientnet_b0, densenet121

from models import vq_vae, simple_classifier
from torch.optim import lr_scheduler

def define_net(args):
    if args.train == 'vqvae':
        nets = vq_vae.model(), None
    elif args.train == 'classifier':
        nets = None, simple_classifier.model(num_classes=args.n_class)
    elif args.train == 'full':
        nets = vq_vae.model(), simple_classifier.model(num_classes=args.n_class)

    return nets

def define_strong_net(args):
    if args.strong_classifier == 'base_resnet18':
        net = base_resnet18(num_classes=args.n_class)
    elif args.strong_classifier == 'base_efficientnet_b0':
        net = base_efficientnet_b0(num_classes=args.n_class)
    elif args.strong_classifier == 'base_densenet121':
        net = base_densenet121(num_classes=args.n_class)
    else:
        raise NotImplementedError(f"Unknown strong classifier: {args.strong_classifier}")
    
    return net

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented' % args.lr_policy)
    return scheduler

##################### Strong classifiers #####################

class Base_Grad_model(nn.Module):
    def __init__(self):
        super(Base_Grad_model, self).__init__()
        self.gradients = None
        self.features_conv = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self,x):
        return self.features_conv(x)

class base_resnet18(Base_Grad_model):
    def __init__(self):
        super(base_resnet18, self).__init__()
        res = resnet18(weights=resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT))
        
        self.features_conv = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4
        )

        self.avgpool = res.avgpool

        self.classifier = res.fc

        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class base_efficientnet_b0(Base_Grad_model):
    def __init__(self):
        super(base_efficientnet_b0, self).__init__()
        eff = efficientnet_b0(weights=efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT))
        self.features_conv = eff.features
        self.avg_pool = eff.avgpool
        self.classifier = eff.classifier
        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
        
        x = F.relu(x, inplace=False) # Use False to avoid gradient issues
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class base_densenet121(Base_Grad_model):
    def __init__(self):
        super(base_densenet121, self).__init__()
        dn = densenet121(pretrained=True)
        self.features_conv = dn.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = dn.classifier
        self.gradients = None


    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
        
        x = F.relu(x, inplace=False) # Use False to avoid gradient issues
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x