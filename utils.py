import data_config
from datasets import SkinCancerData

def get_dataloaders(args):
    data_config = data_config.DataConfig().get_data_config(args.data_name)
    if args.dataset == 'Standard':
        dataloaders = SkinCancerData.Fitzpatrick(data_config=data_config, args=args).get_dataloaders()
    else:
        raise TypeError('%s has not defined' % args.dataset)
    return dataloaders