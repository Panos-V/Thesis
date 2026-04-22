import os
from argparse import ArgumentParser
from thop import profile



import utils
from models.trainer import Trainer



def train(args):
    """ Train the model. """
    dataloaders = utils.get_dataloaders(args)
    model = Trainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    pass


if __name__ == '__main__':


    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='-1',
                help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='fitzpatrick17k', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=224, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--train', default="full", type=str, help='full (whole network) |' \
                                'vqvae (only train vqvae) | classifier (only train classifier)')
    parser.add_argument('--strong_classifier', default='base_resnet18', type=str,
                        help='base_resnet18 | base_efficientnet_b0 | base_densenet121')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--accumulation_steps', default=0, type=int)
    parser.add_argument('--vis_dir', default='vis', type=str)

    args = parser.parse_args()
    
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
