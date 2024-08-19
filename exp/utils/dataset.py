import pickle
import torch
from feats import get_feat_index
from torch.utils.data import Dataset

class LaDeDataset(Dataset):
    def __init__(self, file_path, used_feats, problem="tsp"):
        self.file_path = file_path
        self.problem = problem
        index_map = []
        for feat in used_feats:
            index_map.append(get_feat_index(feat))
        with open(self.file_path, "rb") as f:
            self.dataset = pickle.load(f)
        # we use BatchNorm instead of standardization.
        # standardization
        # bias = {
        #     "cvrp": [4.1662e+06, 8.7300e+05, 0., 0.],
        #     "cvrptw": [4.1662e+06, 8.7300e+05, 0., 0., 0., 0.]
        # }
        # std = {
        #     "cvrp": [2.5976e+03, 5.0038e+03, 1., 1.],
        #     "cvrptw": [2.5976e+03, 5.0038e+03, 1., 1., 1e4, 1e4]
        # }
        self.dataset["node_feat"] = self.dataset["node_feat"][...]
        self.dataset["edge_feat"] = self.dataset["edge_feat"][..., index_map]

        if self.problem == "pdp" or self.problem == "cvrptw":
            self.key_list = ["node_feat", "edge_feat", "label1", "label2", "edge_index", "inverse_edge_index"]
        elif self.problem == "cvrp":
            self.key_list = ["node_feat", "edge_feat", "label", "edge_index", "inverse_edge_index"]

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
        node_feat = torch.tensor(samples[0], dtype=torch.float32) # B x N x 2
        edge_feat = torch.tensor(samples[1], dtype=torch.float32).flatten(1, -2) # B x 20N x feat_num
        edge_index = torch.tensor(samples[-2], dtype=torch.long).flatten(1) # B x 1000
        inverse_edge_index = torch.tensor(samples[-1], dtype=torch.long).flatten(1) # B x 1000
        if self.problem == "pdp" or self.problem == "cvrptw":
            label1 = torch.tensor(samples[2], dtype=torch.long).flatten(1) # B x 1000
            label2 = torch.tensor(samples[3], dtype=torch.long).flatten(1) # B x 1000
            return node_feat, edge_feat, label1, label2, edge_index,  inverse_edge_index
        elif self.problem == "cvrp":
            label = torch.tensor(samples[2], dtype=torch.long).flatten(1) # B x 1000
            return node_feat, edge_feat, label, edge_index,  inverse_edge_index
