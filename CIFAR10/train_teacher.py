import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models import *
from dataloader import *


def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

    # base
    parser.add_argument('-n', '--num_epochs', default=300, type=int, help='number of epochs for knowledge distillation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
    parser.add_argument('--gpu_id', default='1, 2', type=str, help='gpu device')
    # optimization
    parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training')
    parser.add_argument('-lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # dataset
    parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT',
                        help='10 for cifar10,100 for cifar100 (default: 10)')
    # model
    parser.add_argument('--arch', metavar='ANN ARCH', default='wrn_16_2', choices='pyramidnet20, resnet19, wrn_28_4')
    # save and log
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--log', action='store_true', help='logging switch')
    parser.add_argument('--save', action='store_true', help='save training model')
    parser.add_argument('--note', default='',
                        type=str, help='training notes')

    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    return args


def main():
    args = parser_args()
    seed_all(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    best_prec = 0

    if args.log:
        log_fir = os.path.join('log/ann')
        if not os.path.exists(log_fir):
            os.makedirs(log_fir)
        log_name = fdir = args.arch
        log_dir = os.path.join(log_fir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        f = open(os.path.join(log_dir, log_name + '.log'), 'a', buffering=1)
        writer = SummaryWriter(log_dir=os.path.join('./summary_writer/ann', log_name))
    else:
        f = sys.stdout

    f.write('\n\n========>>>>>Start time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    f.write('\nNote: {}'.format(args.note))

    if args.save:
        if not os.path.exists('result/ann'):
            os.makedirs('result/ann')
        fdir = os.path.join('result/ann', log_name)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

    if args.arch == 'pyramidnet20':
        model = PyramidNet(depth=20, alpha=200)
    elif args.arch == 'pyramidnet110':
        model = PyramidNet(depth=110, alpha=64)
    elif args.arch == 'resnet19':
        model = resnet19()
    elif args.arch == 'wrn_28_4':
        model = wrn_28_4()
    elif args.arch == 'wrn_16_2':
        model = wrn_16_2()
    else:
        f.write('Architecture not support!')
        return

    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    f.write('\t=> loading cifar10 data...\n')
    trainloader, testloader = cifar10_dataloaders_parallel(batch_size=args.batch_size)

    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(epoch=epoch, adjust_list=args.lr_decay_epochs, lr_decay_rate=args.lr_decay_rate, optimizer=optimizer)
        torch.cuda.empty_cache()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        f.write('\n\tcurrent learning rate: {}\n'.format(current_lr))
        train_acc, train_loss, data_time = training(epoch, trainloader, model, criterion, optimizer, args, f)

        f.write('\n\ttrain_acc: {top1:.3f}%'.format(top1=train_acc))

        test_acc = validate(testloader, model, criterion, args, f)
        if args.log:
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
        # remember the best precision and save checkpoint
        is_best = test_acc > best_prec
        best_prec = max(test_acc, best_prec)
        f.write('\n\tbest acc: {:1f}\n'.format(best_prec))
        save_state = {'epoch': epoch,
                      'state_dict': model.state_dict(),
                      'best_prec': best_prec,
                      'optimizer': optimizer.state_dict()}
        if args.save:
            save_checkpoint_ann(save_state, fdir, is_best)

    f.write("\nEnd time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    if args.log:
        writer.flush()


def training(epoch, trainloader, model, criterion, optimizer, args, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()

    for idx, (input, target) in enumerate(trainloader):
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """====================== log info ======================="""
        if idx % args.print_freq == 0:
            f.write('\n\tEpoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(epoch, idx, len(trainloader),
                                                                    batch_time=batch_time, data_time=data_time,
                                                                    loss=losses, top1=top1))
    return top1.avg, losses.avg, data_time.avg


def validate(val_loader, model, criterion, args, f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                f.write('\n\tTest: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        f.write('\n\t * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
