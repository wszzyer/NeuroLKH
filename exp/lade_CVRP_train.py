import os
import argparse
import numpy as np
from tqdm import tqdm
from net import SparseGCNModel, GraphTransformer
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import DataLoader
from feats import parse_feat_strs
from utils.dataset import LaDeDataset
import logging

logger = logging.getLogger()
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
    parser.add_argument('--save_dir', type=str, default='saved/exp1/', help='')
    parser.add_argument('--load_pt', type=str, default='', help='')
    parser.add_argument('--device', type=str, action='extend', nargs='+', help='')
    parser.add_argument('--early_stop_thres', type=int, default=15)
    parser.add_argument('--lambda_1', type=float, default=0.1)
    parser.add_argument('--lambda_2', type=float, default=1)
    parser.add_argument('--lambda_3', type=float, default=0.01)
    parser.add_argument('--use_feats', type=str, action='extend', default=['sssp'], nargs='+', help='')
    parser.add_argument('--log_path', type=str, default='')
    return parser.parse_args()

def calculate_loss(problem, y_pred_nodes, y_pred_edges, node_label, edge_label, label_weight, loss_mask, node_num):
    batch_size = y_pred_edges.size(0)
    node_count = y_pred_edges.size(1)
    if problem == 'cvrp':
        node_loss = nn.CrossEntropyLoss(reduction="none").forward(y_pred_nodes.squeeze(), node_label) * torch.sqrt(node_num / 1000)
        node_loss = node_loss.mean()
        # FIXME: Use log_softmax
        # p_edges = nn.functional.softmax(y_pred_edges, dim=-1).view(batch_size, -1, 2)
        # log_p_edges = torch.log(p_edges + 1e-5)
        log_p_edges = nn.functional.log_softmax(y_pred_edges, dim=-1).view(batch_size, -1, 2)
        edge_loss = nn.NLLLoss(label_weight, reduction="none").forward(log_p_edges.transpose(1, 2), edge_label.flatten(-2))
        edge_loss = edge_loss.reshape(batch_size, node_count, -1)[loss_mask]
        edge_loss = edge_loss.mean()
        reg_loss = torch.linalg.vector_norm(log_p_edges[..., 1].squeeze(), dim=1, ord=2).mean()
    else:
        raise NotImplementedError(problem)
    return node_loss, edge_loss, reg_loss

if __name__ == "__main__":
    args = get_args()
    args.problem = args.problem.lower()
    if not args.device:
        args.device.append('cuda:0')

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.set_num_threads(16)
    logging.basicConfig(format='[%(asctime)s][%(levelname)s]%(message)s', datefmt='%I:%M:%S', level=logging.INFO)
    if args.log_path:
       handler = logging.FileHandler(args.log_path)
       handler.setFormatter(logging.Formatter(fmt='[%(asctime)s][%(levelname)s]%(message)s', datefmt='%I:%M:%S'))
       logger.addHandler(handler)

    node_feats, edge_feats = parse_feat_strs(args.use_feats,  print_result=True)
    node_extra_dim=sum(map(lambda cls:cls.size, node_feats))
    edge_dim=sum(map(lambda cls:cls.size, edge_feats))
    net = GraphTransformer(problem=args.problem, 
                        node_extra_dim=node_extra_dim, 
                        edge_dim=edge_dim,
                        node_hidden_dim=128,
                        n_encoder_layers=6,
                        devices=args.device)
    net.to(args.device[0])
    os.makedirs(args.save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_dataset = LaDeDataset(file_path=args.file_path, extra_node_feats=node_feats, edge_feats=edge_feats, problem=args.problem, label_type='ours')
    val_dataset = LaDeDataset(file_path=args.eval_file_path, extra_node_feats=node_feats, edge_feats=edge_feats, problem=args.problem, label_type='ours')
    logger.info(f"Using {args.file_path} as training dataset, {args.eval_file_path} as validation set")

    start_epoch  = 0
    best_loss = 1e7
    worse_count = 0
    edge_labels = train_dataset.dataset["edge_label"].flatten() 
    label_weight = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    label_weight = torch.tensor(label_weight, dtype=torch.float32, device=args.device[-1])
    if args.load_pt:
        saved = torch.load(args.load_pt)
        start_epoch = saved["epoch"] + 1
        best_loss = saved["best_loss"]
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
    logger.info(f"bs = {args.batch_size}, lambda_1 = {args.lambda_1}, lambda_2 = {args.lambda_2}, lambda_3 = {args.lambda_3}")
    for epoch in range(start_epoch, args.n_epoch):
        statistics = {"train_loss": [],
                     "node_loss": [],
                     "edge_loss": [],
                    "train_sample_count": 0,
                    "val_loss": [],
                    "val_sample_count": 0}
        net.train()
        
        pbar = tqdm(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn))
        for index, batch in enumerate(pbar):
            node_feat, edge_feat, node_label, edge_label, edge_index, pad_mask, node_num = batch
            node_feat, edge_feat, edge_index, pad_mask = map(lambda t: t.to(args.device[0]), (node_feat, edge_feat, edge_index, pad_mask))
            node_label = node_label.to(args.device[-1])
            edge_label = edge_label.to(args.device[-1])
            batch_size = node_feat.size(0)
            if args.problem == "cvrp":
                y_node, y_edge = net.forward(node_feat, edge_feat, edge_index, pad_mask)
                node_loss, edge_loss, reg_loss = calculate_loss(args.problem, y_node, y_edge, node_label, edge_label, label_weight, pad_mask.to(args.device[-1]), node_num.to(args.device[-1]))
                loss = args.lambda_1 * node_loss + args.lambda_2 * edge_loss + args.lambda_3 * reg_loss
            else:
                raise NotImplementedError()

            n_nodes = node_feat.size(1)
            loss.backward()
            statistics["node_loss"].append(args.lambda_1 * node_loss.detach().cpu().numpy() * batch_size)
            statistics["edge_loss"].append(args.lambda_2 * edge_loss.detach().cpu().numpy() * batch_size)
            statistics["train_loss"].append(loss.detach().cpu().numpy() * batch_size)
            statistics["train_sample_count"] += batch_size
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"train_loss": loss.item()})
        logger.info(f"Epoch {epoch} loss {np.sum(statistics['train_loss'])/statistics['train_sample_count']:.6f} " + 
               f"node_loss {np.sum(statistics['node_loss'])/statistics['train_sample_count']:.6f} "
               f"edge_loss {np.sum(statistics['edge_loss'])/statistics['train_sample_count']:.6f}")    
        scheduler.step()

        if (epoch + 1) % args.eval_interval == 0:
            net.eval()
            dataset_rank = []

            for val_batch in DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=val_dataset.collate_fn):
                node_feat, edge_feat, node_label, edge_label, edge_index, pad_mask, node_num  = batch
                node_feat, edge_feat, edge_index, pad_mask = map(lambda t: t.to(args.device[0]), (node_feat, edge_feat, edge_index, pad_mask))
                node_label = node_label.to(args.device[-1])
                edge_label = edge_label.to(args.device[-1])
                with torch.no_grad():
                    batch_size = node_feat.size(0)
                    n_nodes = node_feat.size(1)
                    n_edges = edge_feat.size(1) // n_nodes

                    if args.problem == "cvrp":
                        y_node, y_edge = net.forward(node_feat, edge_feat, edge_index, pad_mask)
                        node_loss, edge_loss, reg_loss = calculate_loss(args.problem, y_node, y_edge, node_label, edge_label, label_weight, pad_mask.to(args.device[-1]), node_num.to(args.device[-1]))
                        loss = args.lambda_1 * node_loss + args.lambda_2 * edge_loss + args.lambda_3 * reg_loss
                    else:
                        raise NotImplementedError()
                    
                    if args.problem == "cvrp":
                        y_edge = y_edge.detach().cpu().numpy()
                        edge_label = edge_label.cpu().numpy()
                        rank_batch = np.zeros((batch_size * n_nodes, n_edges))
                        rank_batch[np.arange(batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edge[..., 1].reshape(-1, n_edges))] = np.tile(np.arange(n_edges), (batch_size * n_nodes, 1))
                        dataset_rank.append((rank_batch.reshape(-1) * edge_label.reshape(-1)).sum() / edge_label.sum())
                    statistics["val_loss"].append(loss.detach().cpu().numpy() * batch_size)
                    statistics["val_sample_count"] += batch_size
            avg_loss = np.sum(statistics["val_loss"])/statistics['val_sample_count']
            logger.info(f"{args.eval_file_path} loss {avg_loss:.7f}" +
                   (f" Avg rank: {np.mean(dataset_rank):3f}" if dataset_rank else ""))
            if avg_loss < best_loss:
                best_loss = avg_loss
                worse_count = 0
                torch.save(net.state_dict(), args.save_dir + f"/best.pth")
            else:
                worse_count += 1
                if worse_count > args.early_stop_thres:
                    logger.info("Early stop triggered, stop training.")
                    break

        if (epoch + 1) % args.save_interval == 0:
            torch.save({"epoch": epoch, 
                        "best_loss": best_loss,
                        "model": net.state_dict(), 
                        "optimizer": optimizer.state_dict()
                        },args.save_dir + f"/{epoch}.pth")
