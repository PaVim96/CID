from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn


from CID.models import model_dict
from dataset.cifar100 import get_cifar_dataloaders
from CID.helper.util import adjust_learning_rate
from CID.helper.util import set_seed

from CID.helper.util import AverageMeter, accuracy
import sys


def parse_option():

    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=50, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'], help='dataset')

    # model
    parser.add_argument('--model_t', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_10_10','wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
                                 'ShuffleV2', 'ResNet34', 'wrn_16_4', 'wrn_40_4', 'wrn_16_10', 'ResNet10'])
    


    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('-NT', '--net_T', type=float, default=4, help='net Tempereture')
    parser.add_argument('-s', '--seed', type=int, default=1, help='seed')
    parser.add_argument('-u', '--cu', type=float, default=0, help='moving average cofficient')

    opt = parser.parse_args()

    # set different learning rate fro these models
    if opt.model_t in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    

    # set the path according to the environment
    opt.model_path = './save/teacher_model'
  
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    opt.model_name = 'S:{}_{}_{}'.format(opt.model_t, opt.dataset, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def train_teacher(epoch, train_loader, model, criterion, optimizer, opt):
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):
        input, target, index = data 
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        logit = model(input)
        loss = criterion(logit, target)

        acc1, acc5 = accuracy(logit, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    #P.V: Since this does not use context, I guess this does not contain the interventional loss?
    return top1.avg, losses.avg




def validate_teacher(val_loader, model, opt): 
    """validation"""
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            input, target, index = data 
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            # compute output
            output = model(input)
            acc1_new, acc5_new = accuracy(output, target, topk=(1, 5))
            top1.update(acc1_new[0], input.size(0))
            top5.update(acc5_new[0], input.size(0))           
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Test Acc@1 {top1_new.avg:.3f} Acc@5 {top5_new.avg:.3f}'
              .format(top1_new=top1, top5_new=top5))
        
    return top1.avg, top5.avg


def main_train():
    best_acc = 0
    opt = parse_option()
    print(opt)
    
    set_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
  
        train_loader, val_loader, n_data = get_cifar_dataloaders(100, batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
    elif opt.dataset == "cifar10":
        train_loader, val_loader, n_data = get_cifar_dataloaders(10, batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 10
    

    model = model_dict[opt.model_t](num_classes = n_cls)

    criterion_cls = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD([{'params': model.parameters()}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=True)
    
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_teacher(epoch, train_loader, model, criterion_cls, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

      
        test_acc, tect_acc_top5 = validate_teacher(val_loader, model, opt)


        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_e{}_best.pth'.format(opt.model_t, epoch))
            print('saving the best model!')
            torch.save(state, save_file )

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
 
    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc.cpu().numpy())

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


main_train()