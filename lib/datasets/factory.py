__sets = {}

from datasets.refvg import Refvg

for split in ['train', 'val', 'test']:
    for model_method in ['cmrin', 'dga', 'sgmn']:
        name = 'refvg_{}_{}'.format(split, model_method)
        __sets[name] = (lambda split=split, model_method=model_method: Refvg(split, model_method))


def get_db(name):
    """Get an imdb by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()