import torch
import torch.nn as nn
import os
import shutil
import numpy as np
import cv2
from loguru import logger
from importlib import import_module
from dataloader import SRDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from utils import denormalize_

def check_edsr_logs(args):
    log_root = args.log_root
    os.makedirs(log_root, exist_ok=True)

    # tensorboard log root
    args.tblog = log_root + f'/tblog/{args.model}_{args.act}_{args.n_resblocks}_{args.n_feats}_{args.last_act}/'
    if os.path.exists(args.tblog):
        # os.rmdir(args.tblog)
        shutil.rmtree(args.tblog)
    os.makedirs(args.tblog, exist_ok=True)

    # model weight .pth
    args.weight_pth = log_root + f'/weight/{args.model}_{args.act}_{args.n_resblocks}_{args.n_feats}_{args.last_act}.pth'
    os.makedirs(log_root + f'/weight/', exist_ok=True)

    # train visulization root
    args.log_img_root = log_root + f'/val_result/{args.model}_{args.act}_{args.n_resblocks}_{args.n_feats}_{args.last_act}/'
    if os.path.exists(args.log_img_root):
        # os.rmdir(args.log_img_root)
        shutil.rmtree(args.log_img_root)
    os.makedirs(args.log_img_root, exist_ok=True)

    # logger file
    args.logger = log_root + f'/logger/{args.model}_{args.act}_{args.n_resblocks}_{args.n_feats}_{args.last_act}.txt'
    if os.path.exists(args.logger):
        os.remove(args.logger)
    logger.add(args.logger, rotation="200 MB", backtrace=True, diagnose=True)
    logger.info(str(args))

def check_hardware(args):
    if args.cpu:
        device = torch.device('cpu')
        logger.info("Use Cpu")
    else:
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        logger.info("CUDA visible devices: " + str(torch.cuda.device_count()))
        logger.info("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
    return device

def check_optimizer_(args, model):
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=args.betas,
                                     eps=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def check_loss_(args):
    if args.loss == 'L1':
        criterion = nn.L1Loss()
    elif args.loss == 'L2':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError
    return criterion

def train(args):
    # make dataset for train and validation
    assert args.lr_train_path is not None
    assert args.hr_train_path is not None
    assert args.lr_val_path is not None
    assert args.hr_val_path is not None
    # patch the train data for training
    train_dataset = SRDataset(lr_path=args.lr_train_path, hr_path=args.hr_train_path, patch_size=args.patch_size,
                              scale=args.scale, aug=args.augment, normalization=args.normalization,
                              need_patch=True, suffix=args.suffix)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    val_dataset = SRDataset(lr_path=args.lr_val_path, hr_path=args.hr_val_path, patch_size=args.patch_size,
                            scale=args.scale, normalization=args.normalization, need_patch=True,
                            suffix=args.suffix)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads)

    # chech log
    if args.model == 'EDSR':
        check_edsr_logs(args)
    writer = SummaryWriter(log_dir=args.tblog)
    # check for gpu
    device = check_hardware(args)
    # check the model
    module = import_module('model.' + args.model.lower())
    model = module.wrapper(args)
    if not args.cpu:
        model = model.to(device)
    # check the optimizer
    optimizer = check_optimizer_(args, model)
    # check the lr schedule
    lr_schedule = StepLR(optimizer, args.decay_step, args.gamma)
    # check the loss
    criterion = check_loss_(args)

    # for iteration to train the model and validation for every epoch
    best_val_psnr = -1.0
    best_val_loss = 1e8
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        train_loss = 0.0
        model.train()
        for batch, data in enumerate(train_dataloader):
            x = data['lr']
            y = data['hr']
            x = x.to(device)
            y = y.to(device)

            # perform forward calculation
            y_hat = model(x)
            loss_ = criterion(y_hat, y)
            train_loss += loss_.item()
            logger.info("Epoch-%d-Batch-%d, train loss: %.4f" % (epoch, batch, loss_.item()))
            writer.add_scalar(f'Train/Batchloss', loss_.item(), global_step=epoch*(len(train_dataloader))+batch)

            # perform backward calculation
            optimizer.zero_grad()
            loss_.backward()
            # perform gradient clipping
            if args.gclip > 0:
                nn.utils.clip_grad_value_(model.parameters(), args.gclip)
            optimizer.step()
        train_loss = train_loss / (batch + 1)
        logger.info("Epoch-%d, train loss: %.4f" % (epoch, train_loss))
        writer.add_scalar(f'Train/Epochloss', train_loss, global_step=epoch)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_psnr = 0.0
            for batch, data in enumerate(val_dataloader):
                x = data['lr']
                y = data['hr']
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                loss_ = criterion(y_hat, y)
                val_loss += loss_.item()

                # save the intermedia result for visualization
                y = y[0].detach().cpu().numpy()
                y_hat = y_hat[0].detach().cpu().numpy()
                y = np.transpose(y, (1, 2, 0))
                y_hat = np.transpose(y_hat, (1, 2, 0))
                # if args.normalization == 1:
                #     y = y * 255.0
                #     y_hat = y_hat * 255.0
                y = denormalize_(y, args.normalization)
                y_hat = denormalize_(y_hat, args.normalization)
                # clip is really important, otherwise the anomaly rgb noise data exists
                y = np.clip(y, 0.0, 255.0)
                y_hat = np.clip(y_hat, 0.0, 255.0)

                _res = np.concatenate([y_hat, y], axis=1).astype(np.uint8)
                cv2.imwrite(os.path.join(args.log_img_root, f'{epoch}_{batch}.png'), _res)

            val_loss = val_loss / (batch + 1)
            logger.info("Epoch-%d, validation loss: %.4f" % (epoch, val_loss))
            writer.add_scalar(f'Val/loss', val_loss, global_step=epoch)

        # adjust the learning rate
        lr_schedule.step(epoch=epoch)
        writer.add_scalar(f'Train/lr', lr_schedule.get_lr()[0], global_step=epoch)

        # save the best validation psnr model parameters
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            model.eval().cpu()
            torch.save(model.state_dict(), args.weight_pth)
            model.to(device).train()

if __name__ == "__main__":
    from option import args

    args.hr_train_path = 'D:/Dataset/DIV2K/DIV2K_train_HR/'
    args.lr_train_path = 'D:/Dataset/DIV2K/DIV2K_train_LR_x8/'
    args.hr_val_path = 'D:/Dataset/DIV2K/DIV2K_valid_HR/'
    args.lr_val_path = 'D:/Dataset/DIV2K/DIV2K_valid_LR_x8/'
    args.scale = 8
    args.res_scale = 0.1
    args.last_act = None
    args.augment = True

    args.normalization = 2

    with torch.autograd.detect_anomaly():
        train(args)