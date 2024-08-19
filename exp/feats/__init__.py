from .od_feat import ODFeat
from .weight_feat import SSSPFeat
from .node_heat_feat import NodeHeatFeat

def get_all_feats():
    return (SSSPFeat, ODFeat, NodeHeatFeat)

def get_feat_indexes(key):
    index = 0
    for feat in get_all_feats():
        if key is feat:
            return list(range(index, index + feat.size))
        if key.feat_type == feat.feat_type:
            index += feat.size
    raise RuntimeError(f'Feat {key} not found.')
            

def parse_feat_strs(feat_strs, print_result=False):
    node_feats = []
    edge_feats = []
    for feat_cls in get_all_feats():
        for feat_str in feat_strs:
            if feat_cls.__name__[:-4].lower() == feat_str.lower():
                if feat_cls.feat_type == 'node' and feat_cls not in node_feats:
                    node_feats.append(feat_cls)
                elif feat_cls.feat_type == 'edge' and feat_cls not in edge_feats:
                    edge_feats.append(feat_cls)
    if print_result:
        print(f"Using Node feats: {[feat.__name__ for feat in node_feats]}, Edge Feats: {[feat.__name__ for feat in edge_feats]}")
    return node_feats, edge_feats
