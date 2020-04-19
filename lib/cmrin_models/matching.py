import torch
import torch.nn as nn
import torch.nn.functional as F


class Matching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(Matching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))

    def forward(self, visual_input, lang_input):

        assert visual_input.size(0) == lang_input.size(0)

        visual_feat = visual_input.view((visual_input.size(0) * visual_input.size(1), -1))
        lang_feat = lang_input
        visual_emb = self.vis_emb_fc(visual_feat)
        lang_emb = self.lang_emb_fc(lang_feat)

        # l2-normalize
        visual_emb_normalized = F.normalize(visual_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)

        block_visual_emb_normalized = visual_emb_normalized.view((visual_input.size(0), visual_input.size(1), -1))
        block_lang_emb_normalized = lang_emb_normalized.unsqueeze(1).expand((visual_input.size(0),
                                                                             visual_input.size(1),
                                                                             lang_emb_normalized.size(1)))
        cossim = torch.sum(block_lang_emb_normalized * block_visual_emb_normalized, 2)

        return cossim