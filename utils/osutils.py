import torch
import os
import errno
import shutil
import collections


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, is_best, filedir, filepre, filename='_checkpoint.pth.tar'):
    torch.save(state, os.path.join(filedir, filepre + filename))
    if is_best:
        shutil.copyfile(os.path.join(filedir, filepre + filename), os.path.join(filedir, 'model_best.pth.tar'))


def load_checkpoint(model, criterion, checkpoint_pth):
    print("=> loading checkpoint '{}'".format(checkpoint_pth))
    checkpoint = torch.load(checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['crit_state_dict'])
    print("=> loaded checkpoint '{}'".format(checkpoint_pth))

    return model, criterion
