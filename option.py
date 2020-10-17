import argparse

parser = argparse.ArgumentParser(description='Image Super Resolution')

# Train or Test
parser.add_argument('--phase', type=str, default='train',
                    help='chose the phase for the model, train or test',
                    choices=['train', 'test'])

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--hr_train_path', type=str, default=None,
                    help='train high resolution images path')
parser.add_argument('--lr_train_path', type=str, default=None,
                    help='train low resolution images path')

parser.add_argument('--hr_val_path', type=str, default=None,
                    help='validate high resolution images path')
parser.add_argument('--lr_val_path', type=str, default=None,
                    help='validate low resolution images path')
parser.add_argument('--suffix', type=str, default='png',
                    help='image file name suffix for input')

parser.add_argument('--test_file', type=str, default=None,
                    help='the low resolution image file for test to high resolution')

parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# data normalization type:
# 0 - no normalization
# 1 - just divide the image data by 255.0 to make it's values in 0~1
# 2 - standard gaussian distribute normalize (DIV2K mean and std)
parser.add_argument('--normalization', type=int, default=1,
                    help='number of color channels to use',
                    choices=[0, 1, 2])

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--continue_train', action='store_true',
                    help='continue train from last train status')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function after conv')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size')
parser.add_argument('--stride', type=int, default=1,
                    help='conv stride size')
parser.add_argument('--bn', type=bool, default=False,
                    help='chose batch normalization or not')

parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--last_act', type=type, default=None,
                    help='chose the last layer activation function')

parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')

parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--D', type=int, default=20,
                    help='default number of RDBs')
parser.add_argument('--G', type=int, default=32,
                    help='default number filters of RDB')
parser.add_argument('--C', type=int, default=6,
                    help='default number convs of RDB')

# AFN CVPR2020
parser.add_argument('--n_l3', type=int, default=3,
                    help='default number of AFN_L3')

# DDBPN CVPR2018
parser.add_argument('--nr', type=int, default=32,
                    help='default number of nr in ddbpn')
parser.add_argument('--n_depths', type=int, default=6,
                    help='default number of projection depths in ddbpn')
parser.add_argument('--n_iters', type=int, default=3,
                    help='default number of recurrent iterations in dbpn-mr')
parser.add_argument('--global_res', action='store_true',
                    help='apply the global residual connection or not')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_rg', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--n_rcab', type=int, default=20,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance normalization')

# Training specifications
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--augment', action='store_true',
                    help='use data augmentation, random horizontal flips and 90 rotations')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Test specification
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')

parser.add_argument('--decay_step', type=int, default=100,
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')

parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='L1',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--log_root', type=str, default='logs',
                    help='the root of the log, including the tblogs/ weights/ loggers/ train_visulization')

args = parser.parse_args()