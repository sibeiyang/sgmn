import os.path as osp
import sys
import numpy as np
import random
import torch
import time

import _init_paths
from opt import parse_opt
from datasets.factory import get_db
from utils.logging import Logger
from utils.meter import AverageMeter
from utils.osutils import mkdir_if_missing, save_checkpoint, load_checkpoint
from utils import to_numpy
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# dga model import
from datasets.refdataset import RefDataset as RefDataset
from dga_models.chain_reason import CR
from models.model_utils import clip_gradient
from crits.criterion import TripletLoss

best_prec = 0
args = parse_opt()
opt = vars(args)


def main():
    global best_prec
    global opt

    if opt['id'] != '':
        model_id = opt['id']
    else:
        model_id = time.strftime("%m_%d_%H-%M-%S")

    sys.stdout = Logger(osp.join(opt['log_dir'], 'log.' + model_id + '.txt'))

    # initialize
    checkpoint_dir = osp.join(opt['checkpoint_dir'], model_id)
    mkdir_if_missing(checkpoint_dir)

    # check gpu
    assert opt['gpus'] is not None

    # set random seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])


    # load imdb
    train_refdb = get_db('refvg_train_' + opt['model_method'])
    vocab = train_refdb.load_dictionary()
    opt['vocab_size'] = len(vocab)
    val_refdb = get_db('refvg_val_'+opt['model_method'])

    # model, criterion, optimizer
    model = CR(opt)
    model = torch.nn.DataParallel(model).cuda()
    criterion = TripletLoss(opt['margin']).cuda()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()),
                                 lr=opt['learning_rate'],
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'])

    scheduler = ReduceLROnPlateau(optimizer, factor=0.1,
                                  patience=3, mode='max')

    if opt['evaluate']:
        if osp.isfile(opt['model']):
            model, criterion = load_checkpoint(model, criterion, opt['model'])
            test_refdb = get_db('refvg_test_' + opt['model_method'])
            test_dataset = RefDataset(test_refdb, vocab, opt)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=opt['batch_size'], shuffle=False,
                num_workers=opt['workers'], pin_memory=True)
            test_loss, test_prec = validate(test_loader, model, criterion)
            print(test_prec)
        else:
            print("=> no checkpoint found at '{}'".format(opt['model']))
        return

    # start training
    epoch_cur = 0
    train_dataset = RefDataset(train_refdb, vocab, opt)
    val_dataset = RefDataset(val_refdb, vocab, opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt['batch_size'], shuffle=True,
        num_workers=opt['workers'], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt['batch_size'], shuffle=False,
        num_workers=opt['workers'], pin_memory=True)

    for epoch in range(epoch_cur, opt['max_epochs']):
        train(train_loader, model, criterion, optimizer, epoch)
        val_loss, prec = validate(val_loader, model, criterion, epoch)
        scheduler.step(prec)
        for i, param_group in enumerate(optimizer.param_groups):
            print(float(param_group['lr']))

        is_best = prec >= best_prec
        best_prec = max(best_prec, prec)
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'crit_state_dict': criterion.state_dict(),
            'optimizer': optimizer.state_dict()}, is_best, checkpoint_dir, str(epoch))


def train(train_loader, model, criterion, optimizer, epoch):
    global opt
    losses = AverageMeter()

    # switch to train mode
    model.train()
    criterion.train()

    step = epoch * len(train_loader)
    pred_gt_same = []
    for i, (box, cls, feature, lfeat, lrel,
            sents, sents_gt, gt_boxes, img_ids, sent_ids) in enumerate(train_loader):

        step += 1
        if opt['gpus'] is not None:
            box = box.cuda()
            cls = cls.cuda()
            feature = feature.cuda()
            lfeat = lfeat.cuda()
            lrel = lrel.cuda()
            sents = sents.cuda()
            sents_gt = sents_gt.cuda()

        # compute output
        score = model(feature, cls, lfeat, lrel, sents)
        loss, score = criterion(score, box, cls, sents_gt)

        losses.update(loss.item())

        cls = to_numpy(cls)
        final_score = to_numpy(score.detach())
        final_score[cls == -1] = -999
        pred_ind = np.argmax(final_score, 1)
        sents_gt = to_numpy(sents_gt)
        for j in range(pred_ind.size):
            if sents_gt[j] == pred_ind[j]:
                pred_gt_same.append(1)
            else:
                pred_gt_same.append(0)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, opt['grad_clip'])
        optimizer.step()

        if i % args.print_freq == 0:
            if i != 0:
                same = np.sum(pred_gt_same[-args.print_freq*opt['batch_size']:]) / float(args.print_freq*opt['batch_size'])
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {same:.4f}'.format(
                       epoch, i, len(train_loader), loss=losses, same=same))


def validate(val_loader, model, criterion, epoch=-1):
    global opt
    losses = AverageMeter()

    # switch to eval mode
    model.eval()
    criterion.eval()

    pred_gt_same = []
    with torch.no_grad():
        for i, (box, cls, feature, lfeat, lrel,
                sents, sents_gt, gt_boxes, img_ids, sent_ids) in enumerate(val_loader):

            if opt['gpus'] is not None:
                box = box.cuda()
                cls = cls.cuda()
                feature = feature.cuda()
                lfeat = lfeat.cuda()
                lrel = lrel.cuda()
                sents = sents.cuda()
                sents_gt = sents_gt.cuda()

            # compute output
            score = model(feature, cls, lfeat, lrel, sents)
            loss, score = criterion(score, box, cls, sents_gt)
            losses.update(loss.item())

            cls = to_numpy(cls)
            final_score = to_numpy(score.detach())
            final_score[cls == -1] = -999
            pred_ind = np.argmax(final_score, 1)
            sents_gt = to_numpy(sents_gt)
            for j in range(pred_ind.size):
                if sents_gt[j] == pred_ind[j]:
                    pred_gt_same.append(1)
                else:
                    pred_gt_same.append(0)

            if i % args.print_freq == 0:
                if i != 0:
                    same = np.sum(pred_gt_same[-args.print_freq * opt['batch_size']:]) / float(
                        args.print_freq * opt['batch_size'])
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec {same:.4f}'.format(
                           epoch, i, len(val_loader), loss=losses, same=same))

        same = np.sum(pred_gt_same) / float(len(pred_gt_same))
        print('Epoch: [{0}]\t'
              'Loss {1:.4f}\t'
              'Prec {2:.4f}'.format(epoch, losses.avg, same))

    return losses.avg, same


if __name__ == '__main__':
    main()