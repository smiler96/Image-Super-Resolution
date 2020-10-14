import torch
import os
import numpy as np
import time
import cv2
from importlib import import_module
from loguru import logger
from utils import denormalize_, normalize_

def check_edsr_logs(args):
    log_root = args.log_root
    # model weight .pth
    args.weight_pth = log_root + f'/weight/{args.model}_{args.act}_{args.n_resblocks}_{args.n_feats}_{args.last_act}.pth'
    # result save root
    args.result_root = log_root + f'/result/{args.model}/'
    os.makedirs(args.result_root, exist_ok=True)

def get_input_image(args):
    img = cv2.imread(args.test_file)
    file_name = os.path.basename(args.test_file).split('.')[0]

    # if args.normalization == 0:
    #     pass
    # elif args.normalization == 1:
    #     img = np.float32(img) / 255.0
    # else:
    #     raise NotImplementedError
    img = normalize_(img, type=args.normalization)

    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    return {'img': img, 'fn': file_name}

def test(args):

    # chech log
    if args.model == 'EDSR':
        check_edsr_logs(args)

    # check the model
    module = import_module('model.' + args.model.lower())
    model = module.wrapper(args)

    pretrained_dict = torch.load(args.weight_pth)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    #         myNet.load_state_dict(model_dict)

    model.load_state_dict(model_dict)
    logger.info(f"Load model with model/{args.weight_pth}.pth for testing.")

    model.eval()
    with torch.no_grad():
        start = time.time()
        data = get_input_image(args)
        y_hat = model(data['img'])
        y_hat = y_hat[0].cpu().numpy()
        y_hat = np.transpose(y_hat, (1, 2, 0))

        # if args.normalization == 1:
        #     y_hat = y_hat * 255.0
        y_hat = denormalize_(y_hat, args.normalization)

        # clip is really important, otherwise the anomaly rgb noise data exists
        y_hat = np.clip(y_hat, 0.0, 255.0)

        y_hat = np.uint8(y_hat)
        end = time.time()
        dt = end - start
        logger.info(f"Test file {args.test_file} costs {dt}ms.")
        cv2.imwrite(os.path.join(args.result_root, data['fn'] + f'_{args.last_act}.png'), y_hat)
        logger.info(f"Write {os.path.join(args.result_root, data['fn'] + '_sr.png')}.")

if __name__ == "__main__":
    from option import args

    args.test_file = 'images/0829x8.png'
    args.scale = 8
    args.res_scale = 0.1

    # no normalization
    args.last_act = None
    args.normalization = 2

    # divided by 255.0
    # args.last_act = 'sigmoid'
    # args.normalization = 1

    test(args)



