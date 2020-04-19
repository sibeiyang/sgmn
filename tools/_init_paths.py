import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


#  path
this_dir = osp.dirname(__file__)

# refer path
refer_dir = osp.join(this_dir, '..', 'data', 'ref')
sys.path.insert(0, refer_dir)

# lib path
sys.path.insert(0, osp.join(this_dir, '..'))
sys.path.insert(0, osp.join(this_dir, '..', 'lib'))
sys.path.insert(0, osp.join(this_dir, '..', 'tools'))
sys.path.insert(0, osp.join(this_dir, '..', 'utils'))