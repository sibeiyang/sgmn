import torch
import torch.nn as nn
import numpy as np

from models.language import RNNEncoder, ModuleInputAttention
from models.modules import AttendRelationModule, AttendLocationModule, AttendNodeModule
from models.modules import MergeModule, TransferModule, NormAttnMap


class SGReason(nn.Module):

    def __init__(self, opt):
        super(SGReason, self).__init__()

        # language
        self.seq_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0,
                                      pretrain=True)
        dim_word_emb = opt['word_embedding_size']
        dim_word_cxt = opt['rnn_hidden_size'] * (2 if opt['bidirectional'] else 1)
        # judge module weight for seq (node, relation, location)
        self.weight_module_spo = nn.Sequential(nn.Linear(dim_word_cxt, 3),
                                               nn.Sigmoid())
        # module input attention
        self.node_input_encoder = ModuleInputAttention(dim_word_cxt)
        self.relation_input_encoder = ModuleInputAttention(dim_word_cxt)
        self.location_input_encoder = ModuleInputAttention(dim_word_cxt)
        self.obj_input_encoder = ModuleInputAttention(dim_word_cxt)

        dim_vis_feat = opt['dim_input_vis_feat']
        # module
        self.node_module = AttendNodeModule(dim_vis_feat, opt['vis_init_norm'], opt['jemb_dim'],
                                            dim_word_emb, opt['jemb_drop_out'])
        self.relation_module = AttendRelationModule(dim_vis_feat, opt['vis_init_norm'], opt['jemb_dim'],
                                                    dim_word_emb, opt['jemb_drop_out'])
        self.location_module = AttendLocationModule(opt['vis_init_norm'], opt['jemb_dim'],
                                                    dim_word_emb, opt['jemb_drop_out'])

        self.min_value, self.max_value = -1, 1
        self.sum_module = MergeModule()
        self.sum_relation_module = TransferModule()
        self.elimination = opt['elimination']
        self.norm_fun = NormAttnMap()

        self.need_location = False  # expressions in Ref-Reasoning do not describe the absolute location

    def forward(self, feature, cls, lfeat,
                      seq, seq_weight, seq_type, seq_rel, com_mask,
                      cxt_idx, cxt_idx_mask, cxt_lfeats):
        ''' language seq: seq(bs, num_seq, len_sent); seq_type(bs, num_seq){-1: None, 0: SPO, 1: S, 2:ALL};
                          seq_rel(bs, num_seq, num_seq){-1:None, 0:SS, 1:SO, 2:OS, 3:OO}
        '''
        bs, num_seq = seq.size(0), seq.size(1)
        n = feature.size(1)

        # cxt_feats (bs, n, 5, dim_feat)
        num_cxt = cxt_idx.size(2)
        offset_idx = torch.tensor(np.array(range(bs)) * n, requires_grad=False).cuda().long()
        offset_idx = offset_idx.unsqueeze(1).unsqueeze(2).expand(bs, n, num_cxt)
        cxt_feats = torch.index_select(feature.view(bs*n, -1), 0, (offset_idx+cxt_idx).view(-1))
        cxt_feats = cxt_feats.view(bs, n, num_cxt, -1) * cxt_idx_mask.unsqueeze(3).float()

        context, hidden, embeded, max_length = self.seq_encoder(seq.view(bs*num_seq, -1))
        seq = seq[:, :, 0:max_length]
        seq_weight = seq_weight[:, :, 0:max_length]
        context = context.view(bs, num_seq, max_length, -1)
        hidden = hidden.view(bs, num_seq, -1)
        embeded = embeded.view(bs, num_seq, max_length, -1)
        real_num_seq = torch.sum((seq_type != -1).float(), dim=1).long()
        max_num_seq = torch.max(real_num_seq)

        # module weights of each seq
        weights_spo = self.weight_module_spo(hidden) # bs, num_seq, 3
        weights_spo_expand = weights_spo.unsqueeze(2).expand(bs, num_seq, n, 3)

        # attn each part
        if self.elimination:
            input_labels = (seq != 0).float() * (seq_weight == 1).float()
        else:
            input_labels = (seq != 0).float()
        node_input_emb, node_input_attn = self.node_input_encoder(context, embeded, input_labels) # bs, num_seq, dim_word_embed
        attn_node = self.node_module(feature, node_input_emb, cls) # bs, num_seq, n
        relation_input_emb, relation_input_attn = self.relation_input_encoder(context, embeded, input_labels) # bs, num_seq, dim_word_embed
        attn_relation = self.relation_module(cxt_feats, cxt_lfeats, relation_input_emb) # bs, num_seq, n, num_cxt
        location_input_emb, location_input_attn = self.location_input_encoder(context, embeded, input_labels)
        attn_location = self.location_module(lfeat, location_input_emb, cls) # bs, num_seq, n
        obj_input_emb, obj_input_attn = self.obj_input_encoder(context, embeded, input_labels)
        attn_obj = self.node_module(feature, obj_input_emb, cls)

        global_sub_attn_map = torch.zeros((bs, num_seq, n), requires_grad=False).float().cuda()
        global_obj_attn_map = torch.zeros((bs, num_seq, n), requires_grad=False).float().cuda()
        for i in range(max_num_seq):
            clone_global_sub_attn_map = global_sub_attn_map.clone()
            clone_global_obj_attn_map = global_obj_attn_map.clone()
            # seq type: S
            s_attn_node_iter = weights_spo_expand[:, i, :, 0] * attn_node[:, i, :]
            if self.need_location:
                s_attn_location_iter = weights_spo_expand[:, i, :, 1] * attn_location[:, i, :]
                s_attn_iter_s = s_attn_node_iter + s_attn_location_iter
                s_attn_iter_o = s_attn_node_iter + s_attn_location_iter
                s_attn_iter_s, s_attn_iter_s_norm = self.norm_fun(s_attn_iter_s)
                s_attn_iter_o, s_attn_iter_o_norm = self.norm_fun(s_attn_iter_o)
            else:
                s_attn_iter_s = s_attn_node_iter
                s_attn_iter_o = s_attn_iter_s
                s_attn_iter_s, s_attn_iter_s_norm = self.norm_fun(s_attn_iter_s)
                s_attn_iter_o, s_attn_iter_o_norm = self.norm_fun(s_attn_iter_o)

            # seq type: SPO
            spo_attn_node_iter = s_attn_node_iter
            if self.need_location:
                spo_attn_location_iter = s_attn_location_iter
            spo_attn_relation, spo_attn_obj = self.sum_relation_module(attn_relation[:, i, :, :], cxt_idx,
                                                                       clone_global_sub_attn_map,
                                                                       (seq_rel[:, i, :] == 2).float(),
                                                                       clone_global_obj_attn_map,
                                                                       (seq_rel[:, i, :] == 3).float(),
                                                                       attn_obj = attn_obj[:, i, :])

            spo_attn_relation_iter = weights_spo_expand[:, i, :, 2] * spo_attn_relation
            if self.need_location:
                spo_attn_iter_s = spo_attn_node_iter + spo_attn_location_iter + spo_attn_relation_iter
                spo_attn_iter_s, spo_attn_iter_s_norm = self.norm_fun(spo_attn_iter_s)
                spo_attn_iter_o = spo_attn_obj * (seq_type[:, i] == 0).float().unsqueeze(1).expand(bs, n)
            else:
                spo_attn_iter_s = spo_attn_node_iter + spo_attn_relation_iter
                spo_attn_iter_s, spo_attn_iter_s_norm = self.norm_fun(spo_attn_iter_s)
                spo_attn_iter_o = spo_attn_obj * (seq_type[:, i] == 0).float().unsqueeze(1).expand(bs, n)

            # combine
            seq_type_s_expand = (seq_type[:, i] == 1).float().unsqueeze(1).expand(bs, n)
            seq_type_spo_expand = (seq_type[:, i] == 0).float().unsqueeze(1).expand(bs, n)
            attn_iter_s = s_attn_iter_s * seq_type_s_expand + spo_attn_iter_s * seq_type_spo_expand
            attn_iter_o = s_attn_iter_o * seq_type_s_expand + spo_attn_iter_o * seq_type_spo_expand

            # after rel with sub
            attn_iter_s = self.sum_module(attn_iter_s, clone_global_sub_attn_map, clone_global_obj_attn_map,
                                          (seq_rel[:, i, :] == 0).float(), (seq_rel[:, i, :] == 1).float())

            attn_iter_s[(seq_type[:, i] == -1).unsqueeze(1).expand(bs, n)] = self.min_value
            attn_iter_o[(seq_type[:, i] == -1).unsqueeze(1).expand(bs, n)] = self.min_value
            attn_iter_s[(seq_type[:, i] == 2).unsqueeze(1).expand(bs, n)] = 0
            attn_iter_o[(seq_type[:, i] == 2).unsqueeze(1).expand(bs, n)] = 0

            global_sub_attn_map[:, i, :] = attn_iter_s
            global_obj_attn_map[:, i, :] = attn_iter_o

        com_mask_expand = (com_mask == 1).unsqueeze(2).expand(bs, num_seq, n).float()
        score = torch.sum(com_mask_expand * global_sub_attn_map, dim=1)

        return score
