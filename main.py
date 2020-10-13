import random
import torch
from option import args
from train import train

if __name__ == "__main__":

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.phase == 'train':
        with torch.autograd.detect_anomaly():
            train(args)
    else:
        pass