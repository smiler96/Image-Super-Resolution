import random
import torch
from option import args
from train import train
from test import test

if __name__ == "__main__":

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.phase == 'train':
        with torch.autograd.detect_anomaly():
            train(args)
    else:
        test(args)