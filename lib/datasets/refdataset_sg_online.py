import torch.utils.data as data
import numpy as np
from scipy.spatial.distance import squareform, pdist


class RefDatasetSG(data.Dataset):
    def __init__(self, refdb, vocab, opt):
        self.refdb = refdb
        self.vocab = vocab
        self.opt = opt
        self.ids = self.refdb.image_ids
        self.sent_ids = self.refdb.sent_ids
        self.lfeats = self._compute_lfeats()
        self.num_cxt = 5
        self.sent_ind_to_ids = self._compute_refgt()
        self._compute_reftoken()

    def __getitem__(self, index):
        sent_id = self.sent_ind_to_ids[index]
        sent = self.refdb.ref_db.load_sent(sent_id)
        img_id = self.refdb.ref_db.get_imgIds([sent_id])[0]

        gt_box = sent['gt_box']
        sent_to_box_idx = sent['bbox_ind']
        max_length = self.max_length
        word_idx = np.zeros((max_length,), dtype=np.float32)
        word_idx[0:len(sent['vocab_idx'])] = sent['vocab_idx']

        seq_ind, seq_weight, seq_type, seq_rel, com_mask, seq_subtree, num_seq = self.get_sg_data(sent_id, max_length)
        seq = np.zeros((max_length, max_length), dtype=np.float32)
        seq[:, 0] = 1  # unk
        for i in range(num_seq):
            for j in range(max_length):
                if seq_ind[i, j] != -1:
                    seq[i, j] = word_idx[seq_ind[i, j]]

        # vis feature
        rois = self.refdb.rois_db[img_id].copy()
        num_box = min(rois['box'].shape[0], self.refdb.max_num_box)
        box = np.zeros((self.refdb.max_num_box, 4), dtype=np.float32)
        cls = np.ones((self.refdb.max_num_box, ), dtype=np.float32) * -1.0
        file = rois['file']
        idx = rois['idx']
        roi_feature = self.refdb.h5_files[file]['features'][idx]
        feature = np.zeros((self.refdb.max_num_box, roi_feature.shape[1]), dtype=np.float32)
        feature[0:num_box] = roi_feature[0:num_box].copy()
        box[0:num_box, :] = rois['box'].copy()
        cls[0:num_box] = rois['cls'].copy()

        # dis_ind
        o_box = rois['box'].copy()
        o_pair_dis = compute_distance_pairs(o_box)
        o_sorted_ind = o_pair_dis.argsort(axis=1)

        # cxt feature
        cxt_idx = np.zeros((feature.shape[0], self.num_cxt), dtype=np.int)
        cxt_idx_mask = np.zeros((feature.shape[0], self.num_cxt), dtype=np.int)
        cxt_lfeats = np.zeros((feature.shape[0], self.num_cxt, 5), dtype=np.float32)
        for i in range(num_box):
            j = 0
            b1 = o_box[i]
            for idx in o_sorted_ind[i]:
                if idx == i:
                    continue
                b2 = o_box[idx]
                cxt_lfeats[i, j] = compute_dif_lfeat(b1, b2)
                cxt_idx[i, j] = idx
                cxt_idx_mask[i, j] = 1
                j += 1
                if j == self.num_cxt:
                    break

        lfeat = np.zeros((self.refdb.max_num_box, 5), dtype=np.float32)
        lfeat[0:num_box, :] = self.lfeats[img_id].copy()

        return box, cls, feature, lfeat, \
               word_idx, sent_to_box_idx, gt_box, img_id, sent_id, \
               seq, seq_weight, seq_type, seq_rel, com_mask,\
               cxt_idx, cxt_idx_mask, cxt_lfeats

    def __len__(self):
        return len(list(self.sent_ind_to_ids.keys()))

    def _compute_lfeats(self):
        lfeats = {}
        for img_id in self.ids:
            iw = self.refdb.rois_db[img_id]['size'][0]
            ih = self.refdb.rois_db[img_id]['size'][1]
            boxes = self.refdb.rois_db[img_id]['box']
            num_objs = self.refdb.rois_db[img_id]['num_objs']
            lfeats[img_id] = np.zeros((num_objs, 5), dtype=np.float32)
            for j in range(num_objs):
                box = boxes[j]
                x = box[0]
                y = box[1]
                w = box[2] - x + 1
                h = box[3] - y + 1
                lfeats[img_id][j] = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)], dtype=np.float32)
        return lfeats

    def _compute_refgt(self):
        sent_ind_to_id = {}
        for i, sent_id in enumerate(self.sent_ids):
            img_id = self.refdb.ref_db.get_imgIds([sent_id])[0]
            boxes = self.refdb.rois_db[img_id]['box']
            ref = self.refdb.ref_db.load_sent(sent_id)
            bbox = ref['bbox']
            x0 = bbox[0]
            x1 = bbox[2] + x0 - 1
            y0 = bbox[1]
            y1 = bbox[3] + y0 - 1
            iou, ind, ious = max_overlap([x0, y0, x1, y1], boxes)
            ref['bbox_ind'] = ind
            ref['gt_box'] = np.array([x0, y0, x1, y1], dtype=np.float32)
            sent_ind_to_id[i] = sent_id

        return sent_ind_to_id

    def _compute_reftoken(self):
        self.max_length = 0
        for sent_id in self.refdb.sent_ids:
            sent = self.refdb.ref_db.load_sent(sent_id)
            sg = self.refdb.ref_sg[sent_id]
            tokens = sent['token']
            self.max_length = max(len(tokens), self.max_length)
            sent['vocab_idx'] = []
            for i, wd in enumerate(tokens):
                wd = wd.lower()
                if wd in ['it', 'he', 'she']:
                    split = sg['words_info'][i][2]
                    if split in sg['co_index']:
                        co_split = sg['co_index'][split]
                        split_to_word, split_to_head = get_split_projection(sg['words_info'])
                        wd = tokens[split_to_head[co_split][-1]]
                if wd in self.vocab:
                    sent['vocab_idx'].append(self.vocab[wd])
                else:
                    sent['vocab_idx'].append(self.vocab['<unk>'])

    def get_sg_data(self, sent_id, max_length):
        seqrel_to_label = {'SUBANDSUB': 0, 'SUBANDOBJ': 1, 'OBJANDSUB': 2, 'OBJANDOBJ': 3}
        seqtype_to_label = {'SPO': 0, 'S': 1, 'ALL': 2}
        seq_eliminate = ['and', 'or', 'while', ',', '.']
        sg = self.refdb.ref_sg[sent_id]
        sg_seq = self.refdb.ref_sg_seq[sent_id]
        seq = np.ones((max_length, max_length), dtype=np.int) * -1
        seq_weight = np.zeros((max_length, max_length), dtype=np.int)
        # -1: none, 0: node, 1: node-edge-node, 2: all
        seq_type = np.ones((max_length,), dtype=np.int) * -1
        seq_connection = np.ones((max_length, max_length), dtype=np.int) * -1
        split_to_word, split_to_head = get_split_projection(sg['words_info'])
        num_valid_words_in_seq = []
        seq_subtrees = np.zeros((max_length, max_length), dtype=np.int)
        seqset_list = []
        for i, s in enumerate(sg_seq['seq_sg']):
            seqset_list.append(set([i]))
            t = 0
            num_words_has_weight = 0
            for split in s['seq']:
                words = split_to_word[split]
                for w in words:
                    if w <= max_length:
                        seq[i, t] = w
                        w_info = sg['words_info'][w]
                        if (w_info[0] != 0) and (w_info[1] != 'det') and (w_info[3] not in seq_eliminate):
                            seq_weight[i, t] = 1
                            num_words_has_weight += 1
                        t += 1
            for sr in s['seq_rel']:
                assert sr[0] == i
                seq_connection[i, sr[1]] = seqrel_to_label[sr[2]]
                seqset_list[i] = seqset_list[i] | seqset_list[sr[1]]
            if num_words_has_weight == 0:
                seq_type[i] = seqtype_to_label['ALL']
                if t == 0:
                    seq[i, 0] = 0
            else:
                seq_type[i] = seqtype_to_label[s['type']]
            num_valid_words_in_seq.append(num_words_has_weight)
            for j in seqset_list[i]:
                seq_subtrees[i, j] = 1
        com_mask = np.zeros((max_length,), dtype=np.float32)
        com_seq = sg_seq['com_seq']
        com_seq.sort(reverse=True)
        for com_seq_ind in com_seq:
            if num_valid_words_in_seq[com_seq_ind] > 0:
                com_mask[com_seq_ind] = 1
                break
        num_seq = sg_seq['num_seq']
        if np.sum(com_mask) == 0:
            com_mask[com_seq[0]] = 1
            num_seq = 1
            seq = np.ones((max_length, max_length), dtype=np.int) * -1
            seq[0, 0] = 0

        return seq, seq_weight, seq_type, seq_connection, com_mask, seq_subtrees, num_seq


def compute_IoU(b1, b2):
    iw = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
    if iw <= 0:
        return 0
    ih = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
    if ih <= 0:
        return 0
    ua = float((b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1) + (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1) - iw*ih)
    return iw * ih / ua


def max_overlap(b1, boxes):
    max_value = 0
    max_ind = -1
    ious = []
    for i in range(boxes.shape[0]):
        iou = compute_IoU(b1, boxes[i, :])
        ious.append(iou)
        if iou > max_value:
            max_value = iou
            max_ind = i
    return max_value, max_ind, ious


def get_split_projection(words_info):
    split_to_head = {}
    split_to_word = {}
    for i, word in enumerate(words_info):
        if word[2] in split_to_word:
            split_to_word[word[2]].append(i)
        else:
            split_to_word[word[2]] = [i]
        if word[1] == 'head':
            if word[2] in split_to_head:
                split_to_head[word[2]].append(i)
            else:
                split_to_head[word[2]] = [i]
    return split_to_word, split_to_head


def compute_dif_lfeat(b1, b2):
    rcx, rcy, rw, rh = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2, b1[2] - b1[0] + 1, b1[3] - b1[1] + 1
    cx, cy, w, h = b2[0], b2[1], b2[2] - b2[0] + 1, b2[3] - b2[1] + 1
    return np.array([(cx-rcx)/rw, (cy-rcy)/rh, (cx+w-rcx)/rw, (cy+h-rcy)/rh, w*h/(rw*rh)])


def compute_distance_pairs(b):
    n = b.shape[0]
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    x = np.zeros((n,2), dtype=np.float32)
    x[:, 0] = cx
    x[:, 1] = cy
    return squareform(pdist(x, metric='sqeuclidean'))