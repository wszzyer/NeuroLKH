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

        if self.problem == "pdp" or self.problem == "cvrptw":
            self.key_list = ["node_feat", "edge_feat", "label1", "label2", "edge_index", "inverse_edge_index", "alpha_values"]
        elif self.problem == "cvrp":
            self.key_list = ["node_feat", "edge_feat", "label", "edge_index", "inverse_edge_index", "alpha_values"]
        else:
            raise RuntimeError(f"Cannot recognize problem type: {self.problem}")

    def __iter__(self):
        return iter(zip([self.dataset[key] for key in self.key_list]))
    
    def __len__(self):
        return self.dataset["node_feat"].shape[0]
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self.dataset.__getitem__(index)
        else:
            return [self.dataset[key][index] for key in self.key_list]
    
    def __getitems__(self, indexes):
        return [self.dataset[key][indexes] for key in self.key_list]

    def collate_fn(self, samples):
        node_feat = torch.tensor(samples[0], dtype=torch.float32) # B x N x feat_num
        edge_feat = torch.tensor(samples[1], dtype=torch.float32).flatten(1, -2) # B x 20N x feat_num
        edge_index = torch.tensor(samples[-3], dtype=torch.long).flatten(1) # B x 1000
        inverse_edge_index = torch.tensor(samples[-2], dtype=torch.long).flatten(1) # B x 1000
        alpha_values = torch.tensor(samples[-1], dtype=torch.long) # No,do not flatten me.
        if self.problem == "pdp" or self.problem == "cvrptw":
            label1 = torch.tensor(samples[2], dtype=torch.long).flatten(1) # B x 1000
            label2 = torch.tensor(samples[3], dtype=torch.long).flatten(1) # B x 1000
            return node_feat, edge_feat, label1, label2, edge_index, inverse_edge_index
        elif self.problem == "cvrp":
            label = torch.tensor(samples[2], dtype=torch.long).flatten(1) # B x 1000
            return node_feat, edge_feat, label, edge_index, inverse_edge_index, alpha_values

class LaDeTestDataset(Dataset):
    def __init__(self, problem, node_feat, edge_feat, edge_index, inverse_edge_index, extra_node_feats_class, edge_feats_class):
        self.problem = problem
        self.size = node_feat.shape[0]

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
        self.edge_feat = torch.tensor(edge_feat, dtype=torch.float32).flatten(1, -2) # B x 20N x feat_num
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).flatten(1) # B x 1000
        self.inverse_edge_index = torch.tensor(inverse_edge_index, dtype=torch.long).flatten(1) # B x 1000
           
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.__getitems__(index)
    
    def __getitems__(self, indexes):
        return [self.node_feat[indexes], self.edge_feat[indexes], self.edge_index[indexes], self.inverse_edge_index[indexes]]
    
    def collate_fn(self, samples):
        return samples
