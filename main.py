import torch
import dgl
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score



def load_dataset(name, self_loop=False):
    root = "dataset"
    g = dgl.load_graphs(f"{root}/{name}.bin")[0][0]

    att = g.ndata["feat"]
    adj = g.adj()
    if self_loop:
        adj = adj + dgl.sparse.identity(shape=adj.shape)
    y = g.ndata["label"].to(torch.int64)

    # Convert a dgl sparse matrix to a torch sparse tensor
    indices = torch.from_numpy(
        np.vstack(adj.coo()).astype(np.int64))
    adj = torch.sparse.FloatTensor(indices, adj.val, adj.shape)

    return att, adj, y


def main(args):
    att, adj, y_true = load_dataset(args.dataset)
    print(adj.shape, att.shape, y_true.shape)

    # perform tfgad
    _, _, V_att = torch.svd_lowrank(att, q=args.k1, niter=10)
    _, _, V_adj = torch.svd_lowrank(adj, q=args.k2, niter=10)

    att_rec_err = (att - att @ V_att @ V_att.T).pow(2).sum(dim=1)
    adj_prj_len = (adj @ V_adj).pow(2).sum(dim=1)
    y_score = att_rec_err + adj_prj_len / args.eta

    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="inj_cora", help='dataset name')
    parser.add_argument("--k1", type=int, default=1)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--eta", type=float, default=10)
    args = parser.parse_args()

    main(args)
