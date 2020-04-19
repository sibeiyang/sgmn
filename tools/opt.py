from pprint import pprint
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # data input setting
    parser.add_argument('--model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data/', help='data root')
    parser.add_argument('--workers', type=int, default=0, help='the num of jobs')

    # visual feats setting
    parser.add_argument('--dim_input_vis_feat', default=2048, type=int, help='the dim of input vis feat')
    parser.add_argument('--vis_init_norm', default=20.0, type=float, help='the l2 norm of feature')
    parser.add_argument('--dim_location', default=512, type=int, help='the dim of location embed')

    # language encoder setting
    parser.add_argument('--word_embedding_size', type=int, default=300, help='the encoding size of each token')
    parser.add_argument('--word_drop_out', type=float, default=0.2, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0.2, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode')
    parser.add_argument('--elimination', type=bool, default=False, help='eliminate the unnecessary words')
    parser.add_argument('--word_init_norm', type=float, default=20, help='.')

    parser.add_argument('--dim_hidden_word_judge', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--word_judge_drop', type=float, default=0.2, help='dropout for word judge')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')

    # joint embedding setting
    parser.add_argument('--jemb_dim', default=1024, type=int, help='the dim of joint embedding')    #512 cmrin
    parser.add_argument('--jemb_drop_out', default=0.2, type=float, help='joint embedding drop out') # jemb_dropout

    # loss setting
    parser.add_argument('--margin', type=float, default=0.1, help='margin for ranking loss')

    # optimization
    parser.add_argument('--max_epochs', type=int, default=10, help='max number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in number of images per batch')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')

    # evaluation/checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='output/checkpoints', help='directory to save models')
    parser.add_argument('--log_dir', type=str, default='experiments/logs', help='where to output log')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--id', type=str, default='', help='an id identifying this run/job.')
    parser.add_argument('--seed', type=int, default=9, help='random number generator seed to use')
    parser.add_argument('--gpus', type=list, default=[0, 1], help='which gpus to use, -1 = use CPU')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluate or not')

    # method
    parser.add_argument('--model_method', type=str, default='sgmn')

    # controller
    parser.add_argument('--T_ctrl', type=int, default=5, help='The iterator num of controller')
    parser.add_argument('--dim_reason', type=int, default=1024, help='The dim of reasoning operator')

    # gcn
    parser.add_argument('--num_hid_location_gcn', default=[1024, 1024], type=list, help='number of gcn layer')
    parser.add_argument('--num_location_relation', default=11, type=int, help='number of location relation')
    parser.add_argument('--gcn_drop_out', default=0.2, type=float)
    parser.add_argument('--num_embed_gcn', default=512, type=int)
    parser.add_argument('--dim_edge_gate', default=512, type=int)
    parser.add_argument('--edge_gate_drop_out', default=0.1, type=float)

    # parse
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()