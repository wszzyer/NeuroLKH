import pickle
import torch
import numpy as np
from feats import get_feat_indexes
from utils import get_problem_default_node_feat_dim
from torch.utils.data import Dataset


class LaDeDataset(Dataset):
    def __init__(self, file_path, extra_node_feats, edge_feats, problem="tsp"):
        self.file_path = file_path
        self.problem = problem
        with open(self.file_path, "rb") as f:
            self.dataset = pickle.load(f)
        
        # we use BatchNorm instead of standarization.
        # feature selection
        default_node_dim = get_problem_default_node_feat_dim(problem)
        if extra_node_feats:
            self.dataset["node_feat"] = np.concatenate([self.dataset["node_feat"][..., :default_node_dim],
                                                        self.dataset["node_feat"][..., default_node_dim + np.concatenate([get_feat_indexes(feat) for feat in extra_node_feats])]], axis=-1)
        else:
            self.dataset["node_feat"] = self.dataset["node_feat"][..., :default_node_dim]
        if edge_feats:
            self.dataset["edge_feat"] = self.dataset["edge_feat"][..., np.concatenate([get_feat_indexes(feat) for feat in edge_feats])]
        else:
            raise RuntimeError("At least one edge feat must be enabled.")
        
        self.max_node_num = self.dataset["node_feat"].shape[1]
        self.dataset["pad_mask"] = np.stack([
                np.concatenate((
                    np.ones((n, ), dtype=np.bool_),
                    np.zeros((self.max_node_num - n, ), dtype=np.bool_),
            )) for n in self.dataset["node_num"]])

        self.key_list = ["node_feat", "edge_feat", "label", "edge_index", "pad_mask"]

    def __iter__(self):
        return iter(zip([self.dataset[key] for key in self.key_list]))
    
    def __len__(self):
        return self.dataset["node_feat"].shape[0]
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self.dataset.__getitem__(index)
        else:
            return [self.dataset[key][index] for key in self.key_list]
    
    def __getitems__(self, indice):
        return [self.dataset[key][indice] for key in self.key_list]

    def collate_fn(self, samples):
        node_feat = torch.tensor(samples[0], dtype=torch.float32) # B x N x feat_num
        edge_feat = torch.tensor(samples[1], dtype=torch.float32).flatten(1, -2) # B x (N x N) x feat_num
        label = torch.tensor(samples[2], dtype=torch.long) # B x (2) x 1000
        edge_index = torch.tensor(samples[3], dtype=torch.int32) # B x N x N x edge_num
        pad_mask = torch.tensor(samples[4], dtype=torch.bool)
        return node_feat, edge_feat, label, edge_index, pad_mask

class LaDeTestDataset(Dataset):
    def __init__(self, problem, node_feat, edge_feat, edge_index, node_num, extra_node_feats_class, edge_feats_class):
        self.problem = problem
        self.size = node_feat.shape[0]
        self.max_node_num = node_feat.shape[1]

        # feature selection
        default_node_dim = get_problem_default_node_feat_dim(problem)
        if extra_node_feats_class:
            node_feat = np.concatenate([node_feat[..., :default_node_dim], 
                                        node_feat[..., default_node_dim + np.concatenate([get_feat_indexes(feat) for feat in extra_node_feats_class])]], axis=-1)
        else:
            node_feat = node_feat[..., :default_node_dim]
        if edge_feats_class:
           edge_feat = edge_feat[..., np.concatenate([get_feat_indexes(feat) for feat in edge_feats_class])]
        else:
            raise RuntimeError("At least one edge feat must be enabled.")
        self.node_feat = torch.tensor(node_feat, dtype=torch.float32)
        self.edge_feat = torch.tensor(edge_feat, dtype=torch.float32).flatten(1, -2) # B x (N * N) x feat_num
        self.edge_index = torch.tensor(edge_index, dtype=torch.long) # B x N x N x edge_num
        self.pad_mask = torch.tensor(np.stack([
                np.concatenate((
                    np.ones((n, ), dtype=np.bool_),
                    np.zeros((self.max_node_num - n, ), dtype=np.bool_),
            )) for n in node_num]))
           
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.__getitems__(index)
    
    def __getitems__(self, indice):
        return [self.node_feat[indice], self.edge_feat[indice], self.edge_index[indice], self.pad_mask[indice]]
    
    def collate_fn(self, samples):
        return samples
