import torch
import torch.nn as nn
from cmrin_models.model_utils import NormalizeScale
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphR(nn.Module):
    def __init__(self, opt):
        super(GraphR, self).__init__()

        dim_word_embed = opt['word_embedding_size']
        dim_word_output = opt['rnn_hidden_size'] * (2 if opt['bidirectional'] else 1)

        self.feat_normalizer = NormalizeScale(opt['dim_input_vis_feat'], opt['vis_init_norm'])
        dim_input_vis_feat = opt['dim_input_vis_feat']
        self.word_normalizer = NormalizeScale(dim_word_embed, opt['word_init_norm'])

        self.nrel_l = opt['num_location_relation']
        self.edge_gate = nn.Sequential(nn.Linear(dim_word_embed, opt['dim_edge_gate']),
                                       nn.ReLU(),
                                       nn.Dropout(opt['edge_gate_drop_out']),
                                       nn.Linear(opt['dim_edge_gate'], opt['num_location_relation']),
                                       nn.Softmax(dim=2))
        self.node_word_match = nn.Sequential(nn.Linear(dim_input_vis_feat + dim_word_embed, opt['dim_edge_gate']),
                                             nn.Tanh(),
                                             nn.Linear(opt['dim_edge_gate'], 1),
                                             nn.Softmax(dim=2))

        self.T_ctrl = opt['T_ctrl']

        dim_reason = opt['dim_reason']
        # fusion model
        dim_fusion_input = dim_input_vis_feat + dim_word_embed
        self.fuse_fc = nn.Linear(dim_fusion_input, dim_reason)

        # update feature
        self.w1 = Parameter(torch.FloatTensor(dim_reason, dim_reason))
        self.w3 = Parameter(torch.FloatTensor(dim_reason, dim_reason))
        self.rel_bias = Parameter(torch.FloatTensor(self.nrel_l, dim_reason))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        self.w3.data.uniform_(-stdv, stdv)
        self.rel_bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, words_weight_list, control_vector_list, embed, context_weight, cls, rel):

        x = self.feat_normalizer(feature)
        words = self.word_normalizer(embed)

        word_edge_weights_expand = context_weight[:, :, 0].unsqueeze(2).expand(context_weight.size(0),
                                                                               context_weight.size(1),
                                                                               self.nrel_l)
        words_expand = words.unsqueeze(2).expand(words.size(0), words.size(1), x.size(1), words.size(2))
        x_expand = x.unsqueeze(1).expand(x.size(0), words.size(1), x.size(1), x.size(2))
        word_node_weights_expand = context_weight[:, :, 1].unsqueeze(2).expand(context_weight.size(0),
                                                                               context_weight.size(1),
                                                                               x.size(1))

        # obtain initial attention
        words_input_edge_gate = words
        edge_type_weight = self.edge_gate(words_input_edge_gate)
        edge_weight_per_type_per_word = edge_type_weight * word_edge_weights_expand

        attn_word_node = self.node_word_match(torch.cat([words_expand, x_expand], 3)).squeeze(3)
        is_not_pad_node = (cls != -1.0).float()
        attn_word_node = attn_word_node * (is_not_pad_node.unsqueeze(1).expand(is_not_pad_node.size(0),
                                                                               attn_word_node.size(1),
                                                                               is_not_pad_node.size(1)))
        attn_word_node_sum = attn_word_node.sum(2).unsqueeze(2).expand(attn_word_node.size(0),
                                                                       attn_word_node.size(1),
                                                                       attn_word_node.size(2))
        attn_word_node[attn_word_node_sum != 0] = attn_word_node[attn_word_node_sum != 0] / \
                                                  attn_word_node_sum[attn_word_node_sum != 0]
        node_weight_per_word = attn_word_node * word_node_weights_expand

        # fuse language context on node
        word_feat_per_node = torch.bmm(node_weight_per_word.transpose(1, 2), words)
        fusion_x = self.fuse_fc(torch.cat([word_feat_per_node, x], dim=2))

        # iterator
        x_i = fusion_x
        node_gate = torch.zeros((x.size(0), x.size(1)), requires_grad=False).cuda()
        # edge_type_gate = torch.zeros((edge_type_weight.size(0), edge_type_weight.size(2)), requires_grad=False).cuda()
        x_list = []
        for i in range(self.T_ctrl):
            word_weights = words_weight_list[i]
            control_vector = control_vector_list[i]

            # node
            i_node_weight_per_word = node_weight_per_word * word_weights.unsqueeze(2).expand(word_weights.size(0),
                                                                                             word_weights.size(1),
                                                                                             node_weight_per_word.size(2))
            i_node_weight = torch.sum(i_node_weight_per_word, 1)
            # init edge: simplify the edge update as opening all the edges for Ref-Reasoning
            i_edge_weight_per_type_per_word = edge_weight_per_type_per_word
            i_edge_weight_per_type = torch.sum(i_edge_weight_per_type_per_word, 1)

            if i == 0:
                node_gate, x_i = self.go_node(i_node_weight, x_i)
            else:
                node_gate, x_i = self.go(node_gate, i_node_weight, x_i, i_edge_weight_per_type, cls, rel)
            x_list.append(x_i)

        return x_list[-1]

    def go(self, last_node_gate, node_weight, last_x, edge_weight_per_type_per_sent, cls, rel):
        rel.requires_grad = False
        l_last_x = torch.matmul(last_x, self.w1)
        x3 = torch.matmul(last_x, self.w3)
        x1_t = torch.zeros((l_last_x.size(0), l_last_x.size(1), l_last_x.size(2)), requires_grad=False).cuda()

        last_node_gate_expand = last_node_gate.unsqueeze(2).expand(last_node_gate.size(0), last_node_gate.size(1), last_node_gate.size(1))
        for i in range(self.nrel_l):
            adj1_un = (rel == i).detach()
            adj1 = adj1_un.transpose(2, 1).float()
            gate_matrix_1 = edge_weight_per_type_per_sent[:, i].unsqueeze(1).unsqueeze(2).expand(adj1.size(0),
                                                                                                 adj1.size(1),
                                                                                                 adj1.size(2))
            gate_adj1 = gate_matrix_1 * adj1
            gate_adj1 = gate_adj1 * last_node_gate_expand
            x1_t = x1_t + torch.bmm(gate_adj1, l_last_x) + \
                                    self.rel_bias[i].unsqueeze(0).unsqueeze(1).expand(adj1.size(0),
                                                                                      adj1.size(1),
                                                                                      self.rel_bias.size(1)) * \
                                    gate_adj1.sum(2).unsqueeze(2).expand(adj1.size(0), adj1.size(1),
                                                                         self.rel_bias.size(1))

        adj3 = torch.ones((x3.size(0), x3.size(1)), requires_grad=False).cuda()
        adj3[cls == -1] = 0.0
        adj3 = adj3.unsqueeze(2)
        adj3_weight = adj3
        x3_t = x3 * adj3_weight.expand(adj3.size(0), adj3.size(1), x3.size(2))

        x_new = F.relu(x1_t + x3_t)

        # update gate
        new_gate = node_weight
        new_gate_expand = new_gate.unsqueeze(2).expand(-1, -1, x1_t.size(2))
        total_gate = last_node_gate + new_gate
        total_gate_expand = total_gate.unsqueeze(2).expand(-1, -1, x1_t.size(2))
        x_combine = new_gate_expand * x_new + last_node_gate.unsqueeze(2).expand(-1, -1, last_x.size(2)) * last_x
        x_combine[total_gate_expand != 0] = x_combine[total_gate_expand != 0] / total_gate_expand[total_gate_expand != 0]

        return total_gate, x_combine

    def go_node(self, node_weight, x_ini):
        return node_weight, x_ini

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + '->' \
               + str(self.out_features) + ')'