import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_utils import NormalizeScale


class AttendRelationModule(nn.Module):
    def __init__(self, dim_vis_feat, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout):
        super(AttendRelationModule, self).__init__()
        self.vis_feat_normalizer = NormalizeScale(dim_vis_feat, visual_init_norm)
        self.lfeat_normalizer = NormalizeScale(5, visual_init_norm)
        self.fc = nn.Linear(dim_vis_feat + 5, jemb_dim)
        self.matching = RelationMatching(jemb_dim, dim_lang_feat, jemb_dim, jemb_dropout, -1)

    def forward(self, cxt_feats, cxt_lfeats, lang_feats):
        # cxt_feats: (bs, n, num_cxt, dim_vis_feat); cxt_lfeats: (bs, n, num_cxt, 5); lang_feats: (bs, num_seq, dim_lang)
        # compute masks first
        masks = (cxt_lfeats.sum(3) != 0).float()  # bs, n, num_cxt

        # compute joint encoded context
        batch, n, num_cxt = cxt_feats.size(0), cxt_feats.size(1), cxt_feats.size(2)
        cxt_feats = self.vis_feat_normalizer(cxt_feats.view(batch * n * num_cxt, -1))
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.view(batch * n * num_cxt, -1))

        # joint embed
        concat = torch.cat([cxt_feats, cxt_lfeats], 1)
        rel_feats = self.fc(concat)
        rel_feats = rel_feats.view(batch, n, num_cxt, -1)  # bs, n, 10, jemb_dim

        attn = self.matching(rel_feats, lang_feats, masks)

        return attn


class RelationMatching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_dropout, min_value=-1):
        super(RelationMatching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.min_value = min_value

    def forward(self, vis_input, lang_input, masks):
        # vis_input: (bs, n, num_cxt, vim_dim); lang_input: (bs, num_seq, lang_dim);  mask(bs, n, num_cxt)
        bs, n, num_cxt = vis_input.size(0), vis_input.size(1), vis_input.size(2)
        num_seq = lang_input.size(1)
        vis_emb = self.vis_emb_fc(vis_input.view(bs * n * num_cxt, -1))
        lang_emb = self.lang_emb_fc(lang_input.view(bs * num_seq, -1))

        # l2-normalize
        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)
        vis_emb_normalized = vis_emb_normalized.view(bs, n, num_cxt, -1)
        lang_emb_normalized = lang_emb_normalized.view(bs, num_seq, -1)

        # compute cossim
        cossim = torch.bmm(lang_emb_normalized,
                           vis_emb_normalized.view(bs, n * num_cxt, -1).transpose(1, 2))  # bs, num_seq, n*num_cxt
        cossim = cossim.view(bs, num_seq, n, num_cxt)

        # mask cossim
        mask_expand = masks.unsqueeze(1).expand(bs, num_seq, n, num_cxt)
        cossim = mask_expand * cossim
        cossim[mask_expand == 0] = self.min_value

        cossim = F.relu(cossim)

        return cossim


class AttendLocationModule(nn.Module):
    def __init__(self, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout):
        super(AttendLocationModule, self).__init__()
        self.lfeat_normalizer = NormalizeScale(5, visual_init_norm)
        self.fc = nn.Linear(5, jemb_dim)
        self.matching = Matching(jemb_dim, dim_lang_feat, jemb_dim, jemb_dropout, -1)

    def forward(self, lfeats, lang_feats, cls):
        # lfeats: (bs, n, 5); lang_feats: (bs, num_seq, dim_lang_feat)
        bs, n = lfeats.size(0), lfeats.size(1)

        lfeats = self.lfeat_normalizer(lfeats.view(bs * n, -1))
        loc_feats = self.fc(lfeats).view(bs, n, -1)
        attn = self.matching(loc_feats, lang_feats, (cls != -1).float())

        return attn


class AttendNodeModule(nn.Module):
    def __init__(self, dim_vis_feat, visual_init_norm, jemb_dim, dim_lang_feat, jemb_dropout):
        super(AttendNodeModule, self).__init__()
        self.matching = Matching(dim_vis_feat, dim_lang_feat, jemb_dim, jemb_dropout, -1)
        self.feat_normalizer = NormalizeScale(dim_vis_feat, visual_init_norm)

    def forward(self, vis_feats, lang_feats, cls):
        bs, n = vis_feats.size(0), vis_feats.size(1)
        vis_feats = self.feat_normalizer(vis_feats.view(bs * n, -1)).view(bs, n, -1)

        attn = self.matching(vis_feats, lang_feats, (cls != -1).float())

        return attn


class Matching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_dropout, min_value):
        super(Matching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_dropout),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_dropout),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.min_value = min_value

    def forward(self, vis_input, lang_input, mask):
        # vis_input (bs, n, vis_dim); lang_input (bs, num_seq, lang_dim); mask (bs, n)
        bs, n = vis_input.size(0), vis_input.size(1)
        num_seq = lang_input.size(1)
        vis_emb = self.vis_emb_fc(vis_input.view(bs * n, -1))
        lang_emb = self.lang_emb_fc(lang_input.view(bs * num_seq, -1))

        vis_emb_normalized = F.normalize(vis_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)
        vis_emb_normalized = vis_emb_normalized.view(bs, n, -1)
        lang_emb_normalized = lang_emb_normalized.view(bs, num_seq, -1)

        cossim = torch.bmm(lang_emb_normalized, vis_emb_normalized.transpose(1, 2))  # bs, num_seq, n
        mask_expand = mask.unsqueeze(1).expand(bs, num_seq, n).float()
        cossim = cossim * mask_expand
        cossim[mask_expand == 0] = self.min_value

        return cossim


class MergeModule(nn.Module):
    def __init__(self, norm_type='cossim', need_norm=True):
        super(MergeModule, self).__init__()
        self.norm_type = norm_type
        self.need_norm = need_norm
        self.norm_fun = NormAttnMap(norm_type)

    def forward(self, attn_map, global_sub_attn_maps, global_obj_attn_maps, mask_sub, mask_obj):
        # attn_map(bs, n); global_attn_maps(bs, num_seq, n); mask(bs, num_seq)
        bs, num_seq, n = global_sub_attn_maps.size(0), global_sub_attn_maps.size(1), global_sub_attn_maps.size(2)

        mask_sub_expand = (mask_sub == 1).float().unsqueeze(2).expand(bs, num_seq, n)
        sub_attn_map_sum = torch.sum(mask_sub_expand * global_sub_attn_maps, dim=1)
        mask_obj_expand = (mask_obj == 1).float().unsqueeze(2).expand(bs, num_seq, n)
        obj_attn_map_sum = torch.sum(mask_obj_expand * global_obj_attn_maps, dim=1)
        attn_map_sum = sub_attn_map_sum + obj_attn_map_sum + attn_map
        if self.need_norm:
            attn, norm = self.norm_fun(attn_map_sum)
        else:
            attn = attn_map_sum

        return attn


class TransferModule(nn.Module):
    def __init__(self, norm_type='cossim', need_norm=True):
        super(TransferModule, self).__init__()
        self.norm_type = norm_type
        self.need_norm = need_norm
        self.norm_fun = NormAttnMap(norm_type)

    def forward(self, attn_relation, relation_ind, global_sub_attn_maps, sub_mask, global_obj_attn_maps, obj_mask,
                attn_obj):
        # attn_relation(bs, n, num_cxt), relation_ind(bs, n, num_cxt)
        # global_attn_maps(bs, num_seq, n), mask(bs, num_seq)
        bs, n, num_cxt = attn_relation.size(0), attn_relation.size(1), attn_relation.size(2)
        num_seq = global_sub_attn_maps.size(1)

        # first son or no son
        sub_num_rel = torch.sum(sub_mask, dim=1)
        # get sub son attn
        sub_mask_expand = (sub_mask == 1).float().unsqueeze(2).expand(bs, num_seq, n)
        sub_son_map = torch.sum(sub_mask_expand * global_sub_attn_maps, dim=1)
        sub_num_rel_expand = sub_num_rel.unsqueeze(1).expand(bs, n)
        sub_son_map[sub_num_rel_expand == 0] = 0  # bs, n

        # get obj son attn
        obj_num_rel = torch.sum(obj_mask, dim=1)
        obj_mask_expand = (obj_mask == 1).float().unsqueeze(2).expand(bs, num_seq, n)
        obj_son_map = torch.sum(obj_mask_expand * global_obj_attn_maps, dim=1)
        obj_num_rel_expand = obj_num_rel.unsqueeze(1).expand(bs, n)
        obj_son_map[obj_num_rel_expand == 0] = 0

        #total son
        son_map = sub_son_map + obj_son_map
        num_rel_expand = sub_num_rel_expand + obj_num_rel_expand
        if self.need_norm:
            son_map, norm = self.norm_fun(son_map)
        son_map = son_map * (num_rel_expand != 0).float() + attn_obj * (num_rel_expand == 0).float()

        offset_idx = torch.tensor(np.array(range(bs)) * n, requires_grad=False).cuda()
        offset_idx = offset_idx.unsqueeze(1).unsqueeze(2).expand(bs, n, num_cxt)
        select_idx = (relation_ind != -1).long() * relation_ind + offset_idx
        select_attn = torch.index_select(son_map.view(bs * n, 1), 0, select_idx.view(-1))
        select_attn = select_attn.view(bs, n, num_cxt)
        select_attn = (relation_ind != -1).float() * select_attn

        attn_map_sum = torch.sum(attn_relation * select_attn, dim=2)

        if self.need_norm:
            attn, norm = self.norm_fun(attn_map_sum)
        else:
            attn = attn_map_sum

        return attn, son_map


class NormAttnMap(nn.Module):
    def __init__(self, norm_type='cossim'):
        super(NormAttnMap, self).__init__()
        self.norm_type = norm_type

    def forward(self, attn_map):
        if self.norm_type != 'cosssim':
            norm = torch.max(attn_map, dim=1, keepdim=True)[0].detach()
        else:
            norm = torch.max(torch.abs(attn_map), dim=1, keepdim=True)[0].detach()
        norm[norm <= 1] = 1
        attn = attn_map / norm

        return attn, norm