import argparse
import os
import yaml

from basicsr.archs.d_RMBN_arch import *
from basicsr.archs.t_RMBN_arch import *

parser = argparse.ArgumentParser(description='RMBN convertor')

## yaml configuration files
parser.add_argument('--config', type=str, default=None, help='pre-config file for training')

## paramters for rmbn
parser.add_argument('--scale', type=int, default=4, help='scale for sr network')
parser.add_argument('--colors', type=int, default=3, help='1(Y channls of YCbCr), 3(RGB)')
parser.add_argument('--m_rmbm', type=int, default=10, help='number of rmbm')
parser.add_argument('--c_rmbm', type=int, default=32, help='channels of rmbm')
parser.add_argument('--idt_rmbm', type=int, default=0, help='incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='prelu', help='prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default=r'./',
                    help='path of pretrained model')
parser.add_argument('--output_folder', type=str, default='experiments/convert', help='output folder')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    device = torch.device('cpu')

    t_rmbn = t_RMBN(module_nums=args.m_rmbm, channel_nums=args.c_rmbm, with_idt=args.idt_rmbm,
                    act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    d_rmbn = d_RMBN(module_nums=args.m_rmbm, channel_nums=args.c_rmbm, act_type=args.act_type,
                    scale=args.scale,
                    colors=args.colors).to(device)

    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        t_rmbn.load_state_dict(torch.load(args.pretrain)['params'], strict=True)
    else:
        raise ValueError('the pretrain path is invalud!')

    ## copy weights from t_rmbn to d_rmbn
    depth = len(t_rmbn.backbone)
    for d in range(depth):
        module = t_rmbn.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        d_rmbn.backbone[d].conv3x3.weight.data = RK
        d_rmbn.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':
            pass
        elif act_type == 'rrelu':
            pass
        elif act_type == 'prelu':
            d_rmbn.backbone[d].act.weight.data = module.act.weight.data
        else:
            raise ValueError('invalid type of activation!')

    d_rmbn.conv0.weight.data = t_rmbn.conv0.weight.data.clone()
    d_rmbn.conv0.bias.data = t_rmbn.conv0.bias.data.clone()
    d_rmbn.conv1.weight.data = t_rmbn.conv1.weight.data.clone()
    d_rmbn.conv1.bias.data = t_rmbn.conv1.bias.data.clone()

    torch.save(d_rmbn.state_dict(), '%s/convert.pth' % args.output_folder)
