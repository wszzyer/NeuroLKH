import os
import argparse
import numpy as np
from tqdm import tqdm
from net.sgcn_model import SparseGCNModel
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader
from feats import parse_feat_strs
from utils.dataset import LaDeDataset

def get_args():  
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--problem', default='CVRP', choices=['CVRP', 'CVRPTW'], help='')
    parser.add_argument('--file_path', default='train', help='')
    parser.add_argument('--eval_file_path', default='val', help='')
    parser.add_argument('--n_epoch', type=int, default=10000, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--eval_interval', type=int, default=1, help='')
    parser.add_argument('--eval_batch_size', type=int, default=20, help='')
    parser.add_argument('--n_hidden', type=int, default=128, help='')
    parser.add_argument('--n_gcn_layers', type=int, default=30, help='')
    parser.add_argument('--n_mlp_layers', type=int, default=3, help='')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='')
    parser.add_argument('--save_interval', type=int, default=25, help='')
    parser.add_argument('--save_dir', type=str, default="saved/exp1/", help='')
    parser.add_argument('--load_pt', type=str, default="", help='')
    parser.add_argument('--device', type=str, default="cuda:0", help='')
    parser.add_argument('--early_stop_thres', type=int, default=15)
    parser.add_argument('--ramuda', type=float, default=0.02)
    parser.add_argument('--l_a', type=float, default=0.15)
    parser.add_argument('--use_feats', type=str, action='extend', default=["sssp"], nargs='+', help='')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    args.problem = args.problem.lower()

    torch.manual_seed(114514)
    np.random.seed(114514)

    N_EDGES = 20
    MAGIC = 16

    node_feats, edge_feats = parse_feat_strs(args.use_feats,  print_result=True)
    net = SparseGCNModel(problem=args.problem,
                         n_mlp_layers=4,
                         node_extra_dim=sum(map(lambda cls:cls.size, node_feats)),
                         edge_dim=sum(map(lambda cls:cls.size, edge_feats)))
    net.to(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_dataset = LaDeDataset(file_path=args.file_path, extra_node_feats=node_feats, edge_feats=edge_feats, problem=args.problem)
    val_dataset = LaDeDataset(file_path=args.eval_file_path, extra_node_feats=node_feats, edge_feats=edge_feats, problem=args.problem)
    alpha_loss = torch.nn.CrossEntropyLoss()

    start_epoch  = 0
    best_loss = 1e7
    worse_count = 0
    if args.load_pt:
        saved = torch.load(args.load_pt)
        start_epoch = saved["epoch"] + 1
        best_loss = saved["best_loss"]
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
    for epoch in range(start_epoch, args.n_epoch):
        statistics = {"train_loss": [],
                     "edge_loss": [],
                    "train_sample_count": 0,
                    "val_loss": [],
                    "val_sample_count": 0}
        rank_train = [[] for _ in range(MAGIC)]
        net.train()
        
        edge_labels = train_dataset.dataset["label"].flatten() if "label" in train_dataset.dataset else train_dataset.dataset["label1"].flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edge_cw = torch.tensor(edge_cw, dtype=torch.float32, device=args.device)
        pbar = tqdm(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn))
        for index, batch in enumerate(pbar):
            if args.problem == "cvrp":
                node_feat, edge_feat, label, edge_index, inverse_edge_index, alpha_values = map(lambda t: t.to(args.device), batch)
            else:
                assert args.problem == "cvrptw"
                node_feat, edge_feat, label1, label2, edge_index,  inverse_edge_index = map(lambda t: t.to(args.device), batch)
            batch_size = node_feat.shape[0]
            if args.problem == "cvrp":
                y_edges, edge_loss, reg_loss,  y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, N_EDGES)
                logsoft_y_edges = y_edges[:, :, 1].squeeze().reshape(batch_size, -1, N_EDGES)
                alpha_prob = alpha_values.max(dim=-1, keepdim=True)[0] * 1.2 - alpha_values + 1
                alpha_prob = (alpha_prob / alpha_prob.sum(dim=-1, keepdim=True)).detach()
                loss = edge_loss.mean() + args.ramuda * reg_loss.mean() + args.l_a * alpha_loss(logsoft_y_edges, alpha_prob).mean()
            else:
                assert args.problem == "cvrptw"
                y_edges1, y_edges2, edge1_loss, edge2_loss, reg_loss = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, label1, label2, edge_cw, N_EDGES)
                loss = (edge1_loss.mean() + edge2_loss.mean() + args.ramuda * reg_loss.mean()) / 2
                edge_loss = edge1_loss + edge2_loss

            n_nodes = node_feat.size(1)
            loss.backward()
            statistics["edge_loss"].append(edge_loss.mean().detach().cpu().numpy() * batch_size)
            statistics["train_loss"].append(loss.detach().cpu().numpy() * batch_size)
            statistics["train_sample_count"] += batch_size
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"train_loss": loss.item()})
        print (f"Epoch {epoch} loss {np.sum(statistics['train_loss'])/statistics['train_sample_count']:.6f}",
               f"edge_loss {np.sum(statistics['edge_loss'])/statistics['train_sample_count']:.6f}")    
        scheduler.step()

        if (epoch + 1) % args.eval_interval == 0:
            net.eval()
            eval_results = []
            dataset_rank = []
            dataset_norms = []

            for val_batch in DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=val_dataset.collate_fn):
                if args.problem == "cvrp":
                    node_feat, edge_feat, label, edge_index, inverse_edge_index, alpha_values = map(lambda t: t.to(args.device), val_batch)
                else:
                    assert args.problem == "cvrptw"
                    node_feat, edge_feat, label1, label2, edge_index,  inverse_edge_index = map(lambda t: t.to(args.device), val_batch)

                with torch.no_grad():
                    batch_size = node_feat.shape[0]
                    n_nodes = node_feat.size(1)

                    if args.problem == "cvrp":
                        y_edges, edge_loss, reg_loss, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, N_EDGES)
                        logsoft_y_edges = y_edges[:, :, 1].squeeze().reshape(batch_size, -1, N_EDGES)
                        alpha_prob = alpha_values.max(dim=-1, keepdim=True)[0] * 1.2 - alpha_values + 1
                        alpha_prob = (alpha_prob / alpha_prob.sum(dim=-1, keepdim=True)).detach()
                        loss = edge_loss.mean() + args.ramuda * reg_loss.mean() + args.l_a * alpha_loss(logsoft_y_edges, alpha_prob).mean()
                    else:
                        assert args.problem == "cvrptw"
                        y_edges1, y_edges2, edge1_loss, edge2_loss, reg_loss = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, label1, label2, edge_cw, N_EDGES)
                        edge_loss = edge1_loss + edge2_loss
                        loss = (edge1_loss.mean() + edge2_loss.mean() + args.ramuda * reg_loss.mean())/2
                    
                    if args.problem == "cvrp":
                        y_edges = y_edges.detach().cpu().numpy()
                        label = label.cpu().numpy()
                        rank_batch = np.zeros((batch_size * n_nodes, N_EDGES))
                        rank_batch[np.arange(batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edges[:, :, 1].reshape(-1, N_EDGES))] = np.tile(np.arange(N_EDGES), (batch_size * n_nodes, 1))
                        dataset_rank.append((rank_batch.reshape(-1) * label.reshape(-1)).sum() / label.sum())
                    statistics["val_loss"].append(loss.detach().cpu().numpy() * batch_size)
                    statistics["edge_loss"].append(edge_loss.mean().detach().cpu().numpy() * batch_size)
                    statistics["val_sample_count"] += batch_size
            avg_loss = np.sum(statistics["val_loss"])/statistics['val_sample_count']
            print (f"{args.eval_file_path} loss {avg_loss:.7f} edge_loss {np.sum(statistics["val_loss"])/statistics['val_sample_count']:.7f}" + 
                   (f" Avg rank: {np.mean(dataset_rank):3f}" if dataset_rank else ""))
            if avg_loss < best_loss:
                best_loss = avg_loss
                worse_count = 0
                torch.save(net.state_dict(), args.save_dir + f"/best.pth")
            else:
                worse_count += 1
                if worse_count >  args.early_stop_thres:
                    print("Early stop triggered, stop training.")
                    break

        if (epoch + 1) % args.save_interval == 0:
            torch.save({"epoch": epoch, 
                        "best_loss": best_loss,
                        "model": net.state_dict(), 
                        "optimizer": optimizer.state_dict()
                        },args.save_dir + f"/{epoch}.pth")
