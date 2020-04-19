import os.path as osp
import json


class Refer(object):

    def __init__(self, data_root, dataset, split):
        print "Loading dataset %s %s into memory..." % (dataset, split)
        self.data_dir = osp.join(data_root, dataset)

        ref_file = osp.join(self.data_dir, split+'_expressions.json')

        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = self._load_data(ref_file)
        self.add_token()
        print 'number of refs:', len(self.data['refs'])

    def _load_data(self, ref_file):
        return json.load(open(ref_file, 'r'))

    def get_sentIds(self, img_ids=None):
        if img_ids is None:
            return self.data['refs'].keys()
        else:
            sent_ids = []
            for sent_id in sent_ids:
                if self.data['refs'][sent_id]['image_id'] in img_ids:
                    sent_ids.append(sent_id)
            return sent_ids

    def get_imgIds(self, sent_ids):
        img_ids = []
        for sent_id in sent_ids:
            img_ids.append(self.data['refs'][sent_id]['image_id'])
        img_ids = set(img_ids)
        img_ids = list(img_ids)
        return img_ids

    def load_sent(self, sent_id):
        return self.data['refs'][sent_id]

    def add_token(self):
        for sent_id in self.data['refs']:
            self.data['refs'][sent_id]['token'] = self.data['refs'][sent_id]['expression'].split(' ')
            self.data['refs'][sent_id]['sent'] = self.data['refs'][sent_id]['expression']