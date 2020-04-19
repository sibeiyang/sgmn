import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class Controller(nn.Module):

    def __init__(self, dim_word_output, T_ctrl):
        super(Controller, self).__init__()
        ctrl_dim = dim_word_output

        # define c_0 and reset_parameters
        self.c_init = Parameter(torch.FloatTensor(1, ctrl_dim))
        self.reset_parameters()

        # define fc operators
        self.encode_que_list = nn.ModuleList([nn.Sequential(nn.Linear(ctrl_dim, ctrl_dim),
                                                            nn.Tanh(),
                                                            nn.Linear(ctrl_dim, ctrl_dim))])
        for i in range(T_ctrl - 1):
            self.encode_que_list.append(nn.Sequential(nn.Linear(ctrl_dim, ctrl_dim),
                                                      nn.Tanh(),
                                                      nn.Linear(ctrl_dim, ctrl_dim)))
        self.fc1 = nn.Linear(2*ctrl_dim, ctrl_dim)
        self.fc2 = nn.Linear(ctrl_dim, 1)
        self.fc3 = nn.Linear(2*ctrl_dim, ctrl_dim)

        self.T_ctrl = T_ctrl

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.c_init.size(1))
        self.c_init.data.uniform_(-stdv, stdv)

    def forward(self, lstm_seq, q_encoding, attn_mask):

        c_prev = self.c_init.expand(q_encoding.size(0), self.c_init.size(1))

        words_weight_list = []
        control_vector_list = []

        for t in range(self.T_ctrl):
            q_i = self.encode_que_list[t](q_encoding)
            q_i_c = torch.cat([q_i, c_prev], dim=1)
            cq_i = self.fc1(q_i_c)

            cq_i_reshape = cq_i.unsqueeze(1).expand(-1, lstm_seq.size(1), -1)
            interactions = cq_i_reshape * lstm_seq
            interactions = torch.cat([interactions, lstm_seq], dim=2)
            interactions = F.tanh(self.fc3(interactions))

            logits = self.fc2(interactions).squeeze(2)
            mask = (1.0 - attn_mask.float()) * (-1e30)
            logits = logits + mask
            logits = F.softmax(logits, dim=1)
            norm_cv_i = logits * attn_mask.float()
            norm_cv_i_sum = torch.sum(norm_cv_i, dim=1).unsqueeze(1).expand(logits.size(0), logits.size(1))
            norm_cv_i[norm_cv_i_sum != 0] = norm_cv_i[norm_cv_i_sum != 0] / norm_cv_i_sum[norm_cv_i_sum != 0]

            words_weight_list.append(norm_cv_i)

            c_i = torch.sum(
                norm_cv_i.unsqueeze(2).expand(norm_cv_i.size(0), norm_cv_i.size(1), lstm_seq.size(2)) * lstm_seq, dim=1)
            c_prev = c_i
            control_vector_list.append(c_prev)

        return words_weight_list, control_vector_list
