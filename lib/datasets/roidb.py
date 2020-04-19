import os.path as osp
import numpy as np
from opt import parse_opt
import h5py
from glob import glob
import json


opt = parse_opt()
opt = vars(opt)


class Roidb(object):

    def __init__(self, image_set, model_method):
        self.image_set = image_set
        self.model_method = model_method
        self._data_path = osp.join(opt['data_root'], 'gt_objects')

        self._image_ids, self._roidb, self._h5_files, self._h5_lrel_files = self._load_roidb()

    def _load_roidb(self):
        info_file = osp.join(self._data_path, 'gt_objects_info.json')
        num_files = len(glob(osp.join(self._data_path, 'gt_objects_*.h5')))
        h5_paths = [osp.join(self._data_path, 'gt_objects_%d.h5' % n)
                    for n in range(num_files)]
        h5_lrel_paths = [osp.join(self._data_path, 'lrel_gt_objs_%d.h5' % n)
                        for n in range(num_files)]

        with open(info_file) as f:
            all_info = json.load(f)
        h5_files = [h5py.File(path, 'r') for path in h5_paths]

        image_ids = []
        data = {}
        for img_id in all_info:
            info = all_info[img_id]
            file, idx, num = info['file'], info['idx'], info['objectsNum']
            bbox = h5_files[file]['bboxes'][idx]
            if 'cls' in h5_files[file]:
                cls = h5_files[file]['cls'][idx]
            else:
                cls = np.ones((num,), dtype=np.int) * 999999
            width = info['width']
            height = info['height']

            image_ids.append(img_id)
            data[img_id] = {'size': np.array([width, height], dtype=np.float32),
                            'num_objs': num,
                            'cls': np.array(cls[0:num], dtype=np.float32),
                            'box': np.array(bbox[0:num,:], dtype=np.float32),
                            'file': file,
                            'idx': idx}

        if self.model_method in ['cmrin', 'dga']:
            h5_lrel_files = [h5py.File(path, 'r') for path in h5_lrel_paths]
            return image_ids, data, h5_files, h5_lrel_files
        else:
            return image_ids, data, h5_files, None

    @property
    def image_ids(self):
        return self._image_ids

    @property
    def roidb(self):
        return self._roidb

    @property
    def num_images(self):
        return len(self.image_id)

    @property
    def h5_files(self):
        return self._h5_files

    @property
    def h5_lrel_files(self):
        return self._h5_lrel_files