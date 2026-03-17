import models
from torch.optim import lr_scheduler

def define_net(args):
    if args.strong_classifier == 'base_resnet18':
        net = models.addNetworks.base_resnet18(num_classes=args.n_class)
    elif args.strong_classifier == 'base_efficientnet_b0':
        net = models.addNetworks.base_efficientnet_b0(num_classes=args.n_class)
    elif args.strong_classifier == 'base_densenet121':
        net = models.addNetworks.base_densenet121(num_classes=args.n_class)
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