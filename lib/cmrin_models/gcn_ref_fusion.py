import torch
import torch.nn as nn
from cmrin_models.model_utils import NormalizeScale
from models.language import RNNEncoder
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from cmrin_models.matching import Matching


class GCNRefFusion(nn.Module):

    def __init__(self, opt):
        super(GCNRefFusion, self).__init__()

        # language model
        self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0,
                                      pretrain=True)
        dim_word_output = opt['rnn_hidden_size'] * (2 if opt['bidirectional'] else 1)
        num_cls_word = 4
        self.word_judge = nn.Sequential(nn.Linear(dim_word_output, opt['dim_hidden_word_judge']),
                                        nn.ReLU(),
                                        nn.Dropout(opt['word_judge_drop']),
                                        nn.Linear(opt['dim_hidden_word_judge'], num_cls_word),
                                        nn.Softmax(dim=2))

        self.feat_normalizer = NormalizeScale(opt['dim_input_vis_feat'], opt['vis_init_norm'])
        dim_input_vis_feat = opt['dim_input_vis_feat']
        self.word_normalizer = NormalizeScale(dim_word_output, opt['word_init_norm'])
        # absolute location
        self.locate_encoder = LocationEncoder(opt['vis_init_norm'], opt['dim_location'])

        self.nrel_l = opt['num_location_relation']
        self.edge_gate = nn.Sequential(nn.Linear(dim_word_output, opt['dim_edge_gate']),
                                       nn.ReLU(),
                                       nn.Dropout(opt['edge_gate_drop_out']),
                                       nn.Linear(opt['dim_edge_gate'], opt['num_location_relation']),
                                       nn.Softmax(dim=2))
        self.node_word_match = nn.Sequential(nn.Linear(dim_input_vis_feat + dim_word_output, opt['dim_edge_gate']),
                                             nn.Tanh(),
                                             nn.Linear(opt['dim_edge_gate'], 1),
                                             nn.Softmax(dim=2))

        # fusion model
        dim_gcn_input = dim_input_vis_feat + dim_word_output

        self.gcn_encoder = GCNFusionEncoder(opt['num_hid_location_gcn'],
                                            opt['num_location_relation'],
                                            opt['gcn_drop_out'],
                                            dim_gcn_input)

        self.matching = Matching(opt['num_hid_location_gcn'][-1]+opt['dim_location'],
                                 dim_word_output, opt['jemb_dim'], opt['jemb_drop_out'])

    def forward(self, feature, cls, lfeat, lrel, sents):
        # language
        context, hidden, embeded, max_length = self.rnn_encoder(sents)
        input_gcnencoder_sents = sents[:, 0:max_length]
        context_weight = self.word_judge(context)
        is_not_pad_sents = (input_gcnencoder_sents != 0).float()
        context_weight = context_weight * is_not_pad_sents.unsqueeze(2).expand(is_not_pad_sents.size(0),
                                                                               is_not_pad_sents.size(1),
                                                                               context_weight.size(2))

        x = self.feat_normalizer(feature)
        words = self.word_normalizer(context)

        # obtain edge gate
        word_edge_weights_expand = context_weight[:, :, 0].unsqueeze(2).expand(context_weight.size(0),
                                                                               context_weight.size(1),
                                                                               self.nrel_l)
        words_input_edge_gate = words
        edge_type_weight = self.edge_gate(words_input_edge_gate)
        edge_weight_per_type_per_word = edge_type_weight * word_edge_weights_expand
        edge_weight_per_type_per_sent = torch.sum(edge_weight_per_type_per_word, 1)

        # obtain note gate
        words_expand = words.unsqueeze(2).expand(words.size(0), words.size(1), x.size(1), words.size(2))
        x_expand = x.unsqueeze(1).expand(x.size(0), words.size(1), x.size(1), x.size(2))
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
        word_node_weights_expand = context_weight[:, :, 1].unsqueeze(2).expand(context_weight.size(0),
                                                                               context_weight.size(1),
                                                                               x.size(1))
        node_weight_per_word = attn_word_node * word_node_weights_expand
        word_feat_per_node = torch.bmm(node_weight_per_word.transpose(1, 2), words)
        node_weight_per_sent = torch.sum(node_weight_per_word, 1)

        vis_feat_per_node = x

        # visual location
        location_feature = self.locate_encoder(lfeat)  # num_image, num_box, dim_location_feature

        fusion_feat_per_node = torch.cat([word_feat_per_node, vis_feat_per_node], dim=2)

        gcn_feature = self.gcn_encoder(fusion_feat_per_node, cls, lrel, edge_weight_per_type_per_sent, node_weight_per_sent)

        # judge word context
        attn = torch.sum(context_weight[:, :, 0:3], 2)
        attn_sum = attn.sum(1).unsqueeze(1).expand(attn.size(0), attn.size(1))
        attn[attn_sum != 0] = attn[attn_sum != 0] / (attn_sum[attn_sum != 0])
        attn3 = attn.unsqueeze(1)
        lang_cxt = torch.bmm(attn3, words)
        lang_cxt = lang_cxt.squeeze(1)

        # matching
        vis_cxt = torch.cat([gcn_feature, location_feature], dim=2)
        score_cos = self.matching(vis_cxt, lang_cxt)

        return score_cos


class LocationEncoder(nn.Module):
    def __init__(self, init_norm, dim):
        super(LocationEncoder, self).__init__()
        self.lfeat_normalizer = NormalizeScale(5, init_norm)
        self.fc = nn.Linear(5, dim)

    def forward(self, lfeats):
        loc_feat = self.lfeat_normalizer(lfeats)
        output = self.fc(loc_feat)
        return output


class GCNFusionEncoder(nn.Module):
    def __init__(self, nhid_l, nrel_l, dropout, dim_input_vis):
        super(GCNFusionEncoder, self).__init__()
        self.nrel_l = nrel_l

        self.l_gcn = nn.ModuleList([GCNFusionUnit(dim_input_vis, nhid_l[0], nrel_l)])
        for i in range(len(nhid_l)-1):
            self.l_gcn.append(GCNFusionUnit(nhid_l[i], nhid_l[i+1], nrel_l))
        self.dropout = dropout

    def forward(self, input_x, cls, rel_l, edge_weight_per_type_per_sent, node_weight_per_sent):
        xl = F.relu(self.l_gcn[0](input_x, cls, rel_l, edge_weight_per_type_per_sent, node_weight_per_sent))
        for i in range(len(self.l_gcn) - 1):
            xl = F.relu(self.l_gcn[i+1](xl, cls, rel_l, edge_weight_per_type_per_sent, node_weight_per_sent))
        xl = F.dropout(xl, self.dropout, training=self.training)

        return xl


class GCNFusionUnit(nn.Module):
    def __init__(self, num_in_vis, num_out, num_type):
        super(GCNFusionUnit, self).__init__()
        self.in_features = num_in_vis
        self.out_features = num_out
        self.num_type = num_type
        self.w1 = Parameter(torch.FloatTensor(num_in_vis, num_out))
        self.w2 = Parameter(torch.FloatTensor(num_in_vis, num_out))
        self.w3 = Parameter(torch.FloatTensor(num_in_vis, num_out))
        self.rel_bias = Parameter(torch.FloatTensor(num_type, num_out))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        self.w3.data.uniform_(-stdv, stdv)
        self.rel_bias.data.uniform_(-stdv, stdv)

    def forward(self, x, cls, rel, edge_weight_per_type_per_sent, node_weight_per_sent):
        rel.requires_grad = False
        x3 = torch.matmul(x, self.w3)
        x = node_weight_per_sent.unsqueeze(2).expand(node_weight_per_sent.size(0),
                                                     node_weight_per_sent.size(1),
                                                     x.size(2)) * x
        x1 = torch.matmul(x, self.w1)
        x2 = torch.matmul(x, self.w2)
        x1_t = torch.zeros((x1.size(0), x1.size(1), x1.size(2)), requires_grad=False).cuda()
        x2_t = torch.zeros((x2.size(0), x2.size(1), x2.size(2)), requires_grad=False).cuda()

        for i in range(self.num_type):
            adj1_un = (rel == i).detach()
            adj2_un = adj1_un.transpose(2, 1)
            adj1 = adj2_un.float()
            adj2 = adj1_un.float()
            gate_matrix_1 = edge_weight_per_type_per_sent[:, i].unsqueeze(1).unsqueeze(2).expand(adj1.size(0),
                                                                                                 adj1.size(1),
                                                                                                 adj1.size(2))
            gate_adj1 = gate_matrix_1 * adj1
            x1_t = x1_t + torch.bmm(gate_adj1, x1) + \
                                    self.rel_bias[i].unsqueeze(0).unsqueeze(1).expand(adj1.size(0),
                                                                                      adj1.size(1),
                                                                                      self.rel_bias.size(1)) * \
                                    gate_adj1.sum(2).unsqueeze(2).expand(adj1.size(0), adj1.size(1),
                                                                         self.rel_bias.size(1))

            gate_matrix_2 = edge_weight_per_type_per_sent[:, i].unsqueeze(1).unsqueeze(2).expand(adj2.size(0),
                                                                                                 adj2.size(1),
                                                                                                 adj2.size(2))
            gate_adj2 = gate_matrix_2 * adj2
            x2_t = x2_t + torch.bmm(gate_adj2, x2) + \
                                    self.rel_bias[i].unsqueeze(0).unsqueeze(1).expand(adj2.size(0),
                                                                                      adj2.size(1),
                                                                                      self.rel_bias.size(1)) * \
                                    gate_adj2.sum(2).unsqueeze(2).expand(adj2.size(0), adj2.size(1),
                                                                         self.rel_bias.size(1))

        adj3 = torch.ones((x3.size(0), x3.size(1)), requires_grad=False).cuda()
        adj3[cls == -1] = 0.0
        adj3 = adj3.unsqueeze(2)
        adj3_weight = adj3
        x3_t = x3 * adj3_weight.expand(adj3.size(0), adj3.size(1), x3.size(2))

        x_new = x1_t + x2_t + x3_t

        return x_new

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + '->' \
               + str(self.out_features) + ')'
