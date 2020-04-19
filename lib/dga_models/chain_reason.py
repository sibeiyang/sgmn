import torch
import torch.nn as nn
from cmrin_models.model_utils import NormalizeScale
from models.language import RNNEncoder
from dga_models.controller import Controller
from dga_models.one_step import GraphR
from cmrin_models.matching import Matching


class CR(nn.Module):

    def __init__(self, opt):
        super(CR, self).__init__()

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

        num_cls_word = 4  # for comparison with cmrin, 2->4 and word embedding->word context
        self.word_judge = nn.Sequential(nn.Linear(dim_word_output, opt['dim_hidden_word_judge']),
                                        nn.ReLU(),
                                        nn.Dropout(opt['word_judge_drop']),
                                        nn.Linear(opt['dim_hidden_word_judge'], num_cls_word),
                                        nn.Softmax(dim=2))

        # control on language
        self.controller = Controller(dim_word_output, opt['T_ctrl'])

        self.updater = GraphR(opt)

        # for comparison with cmrin, encode location feats with learned cxt feats
        self.locate_encoder = LocationEncoder(opt['vis_init_norm'], opt['dim_location'])

        self.matching = Matching(opt['dim_reason']+opt['dim_location'],
                                 dim_word_output, opt['jemb_dim'], opt['jemb_drop_out'])

    def forward(self, feature, cls, lfeat, lrel, sents):
        context, hidden, embeded, max_length = self.rnn_encoder(sents)
        input_gcnencoder_sents = sents[:, 0:max_length]
        is_not_pad_sents = (input_gcnencoder_sents != 0).float()

        context_weight = self.word_judge(context)
        context_weight = context_weight * is_not_pad_sents.unsqueeze(2).expand(is_not_pad_sents.size(0),
                                                                               is_not_pad_sents.size(1),
                                                                               context_weight.size(2))

        words_weight_list, control_vector_list = self.controller(context, hidden, is_not_pad_sents)

        x = self.updater(feature, words_weight_list, control_vector_list, embeded, context_weight, cls, lrel)

        location_feature = self.locate_encoder(lfeat)
        final_x = torch.cat([x, location_feature], dim=2)
        score_cos = self.matching(final_x, hidden)

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
