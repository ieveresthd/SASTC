import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from models import *
from distiller import *
from dataloader import *


def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

    # base
    parser.add_argument('-n', '--num_epochs', default=240, type=int, help='number of epochs for knowledge distillation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu device')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--address', default='tcp://127.0.0.1:2322', type=str, help='server socket address')
    # optimization
    parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training')
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam'], help='optimizer type')
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    # dataset
    parser.add_argument('-ct', '--cifar-type', default='10', type=str, metavar='CT',
                        help='10 for cifar10,100 for cifar100 (default: 10)')
    # model
    parser.add_argument('--ann_arch', metavar='ANN ARCH', default='pyramidnet20',
                        choices='pyramidnet20, resnet19, wrn_28_4')
    parser.add_argument('--snn_arch', metavar='ARCH', default='wrn_16_2',
                        choices='vgg11, resnet18, wrn_16_2')
    parser.add_argument('--T', default=7, type=int, help='the bit-width of the quantized network')
    parser.add_argument('--kd_path', help='checkpoint for distilled SNN', type=str, default='')
    parser.add_argument('--ta_path', type=str, default='result/ann/pyramidnet20/ann_model_best.pth.tar',
                        help='initialize form pre-trained teacher floating point model',
                        choices=['result/ann/resnet19/ann_model_best.pth.tar',
                                 'result/ann/wrn_28_4/ann_model_best.pth.tar',
                                 'result/ann/pyramidnet20/ann_model_best.pth.tar'])
    # knowledge distillation
    parser.add_argument('--distill_type', type=str, default='KD', choices=['KD', 'SASTC'])
    parser.add_argument('--kd_T', type=float, default=4, help='knowledge distillation temperature')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('--beta', type=float, default=100, help='weight balance for other losses')
    parser.add_argument('--mlp_dim', type=int, default=128, help='dimension of MLP')
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


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    seed_all(args.seed)
    cudnn.deterministic = True

    best_prec = 0

    dist.init_process_group(backend='nccl',
                            init_method=args.address,
                            world_size=args.nprocs,
                            rank=local_rank)

    if args.log:
        log_name = (args.snn_arch + '_T' + str(args.T) + '_lr' + str(args.learning_rate) + '_beta' + str(args.beta)
                    + '_ANN' + args.ann_arch)
        log_dir = os.path.join('log', args.note, args.distill_type, log_name)
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        except:
            pass
        f = open(os.path.join(log_dir, log_name + '.log'), 'a', buffering=1)
        writer_fir = os.path.join('./summary_writer', args.note, args.distill_type, log_name)
        writer = SummaryWriter(log_dir=writer_fir)
    else:
        f = sys.stdout

    if args.save:
        fdir = os.path.join('./result', args.note, args.distill_type, log_name)
        try:
            if not os.path.exists(fdir):
                os.makedirs(fdir)
        except:
            pass
    f.write('\n\n========>>>>>Start time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    f.write('\nNote: {}'.format(args.note))
    f.write('\nTime steps: {}'.format(args.T))
    f.write('\nOptimizer: {}'.format(args.optimizer))
    f.write('\nLearning rate: {:.4f}'.format(args.learning_rate))
    f.write('\nlr_decay_epochs: {}'.format(args.lr_decay_epochs))
    f.write('\nknowledge distillation temperature: {}'.format(args.kd_T))
    f.write('\ngamma: {}'.format(args.gamma))
    f.write('\nalpha: {}'.format(args.alpha))
    f.write('\nbeta: {}'.format(args.beta))
    f.write('\n\t=> Building model...\n')

    student_model, teacher_model = init_models(args, f)
    torch.cuda.set_device(local_rank)
    student_model.cuda(local_rank)
    teacher_model.cuda(local_rank)
    args.batch_size = int(args.batch_size / args.nprocs)
    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[local_rank])
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[local_rank])

    # get feature sizes of student and teacher
    student_model.eval()
    teacher_model.eval()
    pseudo_data = torch.randn(10, 3, 32, 32).cuda()
    ta_features, _ = teacher_model(pseudo_data, get_feature=True)
    st_features, _ = student_model(pseudo_data, get_feature=True)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_div = DistillKL(T=args.kd_T)

    if args.distill_type == 'KD':
        criterion_kd = DistillKL(T=args.kd_T)
    elif args.distill_type == 'SASTC':
        st_chl = [ft.shape[2] for ft in st_features[1:-1]]
        ta_chl = [ft.shape[1] for ft in ta_features[1:-1]]
        criterion_kd = SASTCLoss()
        kd_module = SelfAttentionTime(len(st_features) - 2, len(ta_features) - 2, args.mlp_dim, st_chl, ta_chl)
    else:
        criterion_kd = None

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    if args.distill_type == 'SASTC':
        kd_module = kd_module.cuda(local_rank)
    criterion_list = criterion_list.cuda(local_rank)

    student_model, teacher_model = load_models(student_model, teacher_model, args, f)
    if args.evaluate and os.path.isfile(args.kd_path):
        args.teacher = ''
        f.write("\t=> loading distilled SNN checkpoint\n")
        snn_ckp = torch.load(args.kd_path)
        student_model.load_state_dict(snn_ckp["state_dict"], strict=True)

    f.write('\t=> loading cifar10 test data...\n')
    trainloader, testloader = cifar_dataloaders_distribute(data_type=args.cifar_type, batch_size=args.batch_size)
    if args.evaluate:
        f.write('\t=> Evaluation\n')
        validate(testloader, student_model, criterion_list[0], local_rank, args, f)
        f.write('\tEnd')
        return

    """================================ Distilling ==========================================="""
    if args.distill_type == 'SASTC':
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam([{"params": student_model.parameters()},
                                         {"params": kd_module.parameters()}],
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD([{"params": student_model.parameters()},
                                         {"params": kd_module.parameters()}],
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        else:
            f.write('Error: No optimizer!')
            sys.exit()
    else:
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(student_model.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            f.write('Error: No optimizer!')
            sys.exit()

    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(epoch=epoch, adjust_list=args.lr_decay_epochs, lr_decay_rate=args.lr_decay_rate, optimizer=optimizer)
        torch.cuda.empty_cache()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        f.write('\n\tcurrent learning rate: {}\n'.format(current_lr))
        if args.distill_type == 'SASTC':
            train_acc, train_loss, data_time = train_distill(epoch, trainloader, student_model, teacher_model,
                                                             criterion_list, optimizer, local_rank, args.nprocs, args, f, kd_module)
        else:
            train_acc, train_loss, data_time = train_distill(epoch, trainloader, student_model, teacher_model,
                                                             criterion_list, optimizer, local_rank, args.nprocs, args, f)
        f.write('\n\ttrain_acc: {top1:.3f}%'.format(top1=train_acc))

        test_acc = validate(testloader, student_model, criterion_ce, local_rank, args, f)
        if args.log:
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
        # remember the best precision and save checkpoint
        is_best = test_acc > best_prec
        best_prec = max(test_acc, best_prec)
        f.write('\n\tbest acc: {:1f}\n'.format(best_prec))
        save_state = {'epoch': epoch,
                     'state_dict': student_model.state_dict(),
                     'best_prec': best_prec,
                     'optimizer': optimizer.state_dict()}
        if args.save:
            save_checkpoint_snn(save_state, fdir, is_best)

    f.write("\nEnd time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    if args.log:
        writer.flush()


def init_models(args, f):
    if args.snn_arch == 'vgg11':
        snn = VGG11SNN(args.T)
    elif args.snn_arch == 'resnet18':
        snn = resnet18snn()
        snn.T = args.T
    elif args.snn_arch == 'wrn_16_2':
        snn = wrn_16_2_snn()
        snn.T = args.T
    else:
        f.write('\n\tSNN architecture not support!')
        sys.exit()

    if args.ann_arch == 'pyramidnet20':
        ann = PyramidNet(depth=20, alpha=200)
    elif args.ann_arch == 'resnet19':
        ann = resnet19()
    elif args.ann_arch == 'wrn_28_4':
        ann = wrn_28_4()
    else:
        f.write('\n\tANN architecture not support!')
        sys.exit()
    return snn, ann


def load_models(snn, ann, args, f):
    if os.path.isfile(args.kd_path):
        f.write("\n=> loading pre-trained SNN model")
        checkpoint = torch.load(args.kd_path, map_location=torch.device('cpu'))
        snn.load_state_dict(checkpoint['state_dict'], strict=True)
        f.write('\n\t <== pre-snn ft best test acc: {:.3f} ==>'.format(checkpoint['best_prec']))

    if os.path.isfile(args.ta_path):
        f.write("\n\t=> loading pre-trained ANN model\n")
        ann_ckp = torch.load(args.ta_path, map_location=torch.device('cpu'))

        # Remove DataParallel wrapper 'module'
        # for name in list(ann_ckp['state_dict'].keys()):
        #     ann_ckp['state_dict'][name[7:]] = ann_ckp['state_dict'].pop(name)
        ann.load_state_dict(ann_ckp['state_dict'], strict=True)
        f.write('\n\t <== Teacher ANN test acc: {}\n'.format(ann_ckp['best_prec']))
    else:
        f.write('\n\tNo pre-trained model found !\n')
        sys.exit()
    return snn, ann


def train_distill(epoch, trainloader, snn, ann, criterion_list, optimizer, local_rank, local_size, args, f, kd_module=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    snn.train()
    ann.eval()

    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    for idx, (input, target) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if args.distill_type == 'SASTC' and input.size(0) < args.batch_size:
            continue
        input = input.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        with torch.no_grad():
            ta_features, ta_output = ann(input, get_feature=True)
            ta_features = [fea.detach() for fea in ta_features]
        st_features, st_output = snn(input, get_feature=True)
        mean_st_output = st_output.mean(1)

        loss_ce = criterion_ce(mean_st_output, target)
        loss_div = criterion_div(mean_st_output, ta_output)
        loss_kd = 0

        if args.distill_type == 'KD':
            loss_kd = 0
        elif args.distill_type == 'SASTC':
            st_value_tmp, fea_target_tmp, kd_weight_tmp = kd_module(st_features[1:-1], ta_features[1:-1])
            loss_kd = criterion_kd(st_value_tmp, fea_target_tmp, kd_weight_tmp)

        loss = args.gamma * loss_ce + args.alpha * loss_div + args.beta * loss_kd
        prec = accuracy(mean_st_output, target)[0]

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, local_size)
        reduced_prec = reduce_mean(prec, local_size)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        """====================== backward propagation ===================="""
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


def validate(val_loader, model, criterion, local_rank, args, f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(input)
            mean_output = output.mean(1)
            loss = criterion(mean_output, target)

            # measure accuracy and record loss
            prec = accuracy(mean_output, target)[0]
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
