import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import gc

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet

############# using mix-up

def rand_bbox(width, height, lam):
  ##lambda is mask size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(width * cut_rat)
    cut_h = np.int(height * cut_rat)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def cutmix(inputs, alpha=1.0):
    '''Returns mixed inputs, pairs of targets (not one-hot), and lambda'''
    ################################################################################
    # TODO: CutMix function.
    # Output shape: mixup_data ([batch_size, C, H, W])       
    #               labels_1 ([batch_size, ])
    #               labels_2 ([batch_size, ])
    #               lam (float)                                         
    ################################################################################

    lam = np.random.beta(alpha, alpha)#rand lambda

    batch_size, width, height = inputs.shape
    index = torch.randperm(batch_size)#.to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(width, height, lam)
    inputs[:, bbx1:bbx2, bby1:bby2] = inputs[index, bbx1:bbx2, bby1:bby2]#cut mix
    
    mixup_data = inputs
    
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))

    #labels_1, labels_2 = labels, labels[index]


    return mixup_data, index, bbx1, bby1, bbx2, bby2    

def cutmix_val(inputs, index_, bbx1_, bby1_, bbx2_, bby2_):
    '''Returns mixed inputs, pairs of targets (not one-hot), and lambda'''
    ################################################################################
    # TODO: CutMix function.
    # Output shape: mixup_data ([batch_size, C, H, W])       
    #               labels_1 ([batch_size, ])
    #               labels_2 ([batch_size, ])
    #               lam (float)                                         
    ################################################################################

    #lam = lam_

    batch_size, width, height = inputs.shape
    index = index_

    bbx1, bby1, bbx2, bby2 = bbx1_, bby1_, bbx2_, bby2_

    inputs[:, bbx1:bbx2, bby1:bby2] = inputs[index, bbx1:bbx2, bby1:bby2]
    
    mixup_data = inputs
    
   # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))#잘라서 쓴 만큼으로 lam을 수정 박스 커지면 2번째애의 것이 커지니까 1-라네염 쩐당 ㅇㅅㅇ

   # labels_1, labels_2 = labels, labels[index]


    return mixup_data

#############

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):

        input, target, maximum, _, _ = data

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        ################################

        input, index_, bbx1_, bby1_, bbx2_, bby2_ = cutmix(input)

        target = cutmix_val(target, index_, bbx1_, bby1_, bbx2_, bby2_)

        ################################

        output = model(input)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    model = Unet(in_chans = args.in_chans, out_chans = args.out_chans).to(device=device)

    #model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    # checkpoint = torch.load(args.exp_dir / 'model.pt')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])


    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)#학습용 data
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)#정답 data
    
    # ################################

    # train_loader, index_, bbx1_, bby1_, bbx2_, bby2_ = cutmix(train_loader)

    # val_loader = cutmix_val(val_loader, index_, bbx1_, bby1_, bbx2_, bby2_)

    # ################################

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
        gc.collect()
        torch.cuda.empty_cache()
   
    del model
