import torch.utils.data as data
import numpy as np


class RefDataset(data.Dataset):
    def __init__(self, refdb, vocab, opt):
        self.refdb = refdb
        self.vocab = vocab
        self.opt = opt
        self.ids = self.refdb.image_ids
        self.sent_ids = self.refdb.sent_ids
        self.lfeats = self._compute_lfeats()
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

        lfeat = np.zeros((self.refdb.max_num_box, 5), dtype=np.float32)
        roi_lrel = self.refdb.h5_lrel_files[file]['lrel'][idx]
        lrel = np.ones((self.refdb.max_num_box, self.refdb.max_num_box), dtype=np.float32) * -1.0
        lrel[0:num_box, 0:num_box] = roi_lrel[0:num_box, 0:num_box]
        lfeat[0:num_box, :] = self.lfeats[img_id].copy()

        return box, cls, feature, lfeat, lrel, \
               word_idx, sent_to_box_idx, gt_box, img_id, sent_id

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
            tokens = sent['token']
            self.max_length = max(len(tokens), self.max_length)
            sent['vocab_idx'] = []
            for i, wd in enumerate(tokens):
                wd = wd.lower()
                if wd in self.vocab:
                    sent['vocab_idx'].append(self.vocab[wd])
                else:
                    sent['vocab_idx'].append(self.vocab['<unk>'])


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