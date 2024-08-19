from .od_feat import ODFeat
from .weight_feat import SSSPFeat
from .space_syntax import attach_space_syntax

def get_all_feats():
    return (SSSPFeat, ODFeat)

def get_feat_index(key):
    index = 0
    for feat in get_all_feats():
        if key is feat:
            return index
        if key.feat_type == feat.feat_type:
            index += 1
    raise RuntimeError(f'Feat {key} not found.')
            

def parse_feat_strs(feat_strs):
    ret_list = []
    for feat_cls in get_all_feats():
        for feat_str in feat_strs:
            if feat_cls.__name__[:-4].lower() == feat_str.lower() and feat_cls not in ret_list:
                ret_list.append(feat_cls)
    return ret_list
