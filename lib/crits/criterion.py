import torch
import torch.nn as nn
import numpy as np
from utils import to_numpy, to_torch
from torch.nn.parameter import Parameter


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.selector = Selector(margin=self.margin, cpu=False)

    def forward(self, scores, box, cls, sent_gt):
        bs = scores.size(0)
        loss = torch.zeros((bs,), requires_grad=False).cuda()
        for i in range(bs):
            negative_ind = self.selector.get_negative_ind(scores[i].detach(), sent_gt[i], box[i],
                                                          cls[i])
            if negative_ind is not None:
                loss[i] = torch.clamp(self.margin + scores[i, negative_ind]
                                      - scores[i, sent_gt[i]], min=0)

        return torch.mean(loss).view(-1), scores


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    if len(hard_negatives) == 0:
        return None
    return np.random.choice(hard_negatives, replace=False)


class Selector(object):
    def __init__(self, margin, cpu=True):
        super(Selector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.select_fn = random_hard_negative

    def get_negative_ind(self, scores, positive_ind, box, cls):
        if self.cpu:
            scores = scores.cpu()
            box = box.cpu()
            positive_ind = positive_ind.cpu()

        positive_score = scores[positive_ind]
        loss_value = self.margin + scores - positive_score
        loss_value[positive_ind] = 0.0
        loss_value[cls == -1] = -1.0
        loss_value = to_numpy(loss_value)
        negative_ind = self.select_fn(loss_value)
        return negative_ind if negative_ind is not None else None


class ScaleLayer(nn.Module):
    def __init__(self, init_value=20.0):
        super(ScaleLayer, self).__init__()
        self.scale = Parameter(torch.FloatTensor(1,),)
        self.reset_parameter(init_value)

    def reset_parameter(self, init_value):
        self.scale.data.fill_(init_value)

    def forward(self, x):
        x = x * self.scale
        return x


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.scale_fun = ScaleLayer(20)

    def forward(self, score, cls, sent_gt):
        bs, n = score.size(0), score.size(1)
        x = self.scale_fun(score)

        loss = torch.zeros((bs, ), requires_grad=False).cuda()
        logits = torch.zeros((bs, n), requires_grad=False).cuda()
        for i in range(bs):
            label = np.ones((n, ), dtype=np.float32)
            label[to_numpy((cls[i])) == -1.0] = 0
            valid_label = to_torch(label).cuda()
            s = torch.exp(x[i, :]) * valid_label
            s = s / torch.sum(s)
            loss[i] = -torch.log(s[sent_gt[i].long()])
            logits[i, :] = s

        return torch.mean(loss), logits
