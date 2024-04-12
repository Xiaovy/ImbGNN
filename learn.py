from utils import *
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def train(encoder, classifier, data_loader, optimizer_e, optimizer_c, args):
    encoder.train()
    classifier.train()

    total_loss = 0
    for i, batch in enumerate(data_loader):
        batch_to_gpu(batch, args.device)
        data, train_idx = batch['data'], batch['train_idx']

        knn_adj_t, aug_adj_ts, aug_xs = batch['knn_adj_t'], batch['aug_adj_ts'], batch['aug_xs']

        H_augs, logit_aug_props = [], []

        for i in range(args.aug_num):
            H_augs.append(encoder(aug_xs[i], aug_adj_ts[i], data.ptr))

            H_knn = H_augs[-1]
            for k in range(args.knn_layer):
                H_knn = torch.sparse.mm(knn_adj_t, H_knn)
            logit_aug_props.append(classifier(H_knn)[train_idx])

        loss = 0
        for i in range(args.aug_num):
            loss += F.nll_loss(logit_aug_props[i], data.y[train_idx])
        loss = loss / args.aug_num

        loss = loss + consis_loss(logit_aug_props, temp=args.temp)

        optimizer_e.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_e.step()
        optimizer_c.step()

        total_loss += (loss * train_idx.shape[0]).item()

    return total_loss / (i + 1)


def eval(encoder, classifier, data_loader, args):
    encoder.eval()
    classifier.eval()

    pred, truth = [], []
    deg, size = [], []
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch_to_gpu(batch, args.device)
            data, train_idx = batch['data'], batch['train_idx']

            knn_adj_t, aug_adj_ts, aug_xs = batch['knn_adj_t'], batch['aug_adj_ts'], batch['aug_xs']

            H_augs, logit_aug_props = [], []

            for i in range(args.aug_num):
                H_augs.append(encoder(aug_xs[i], aug_adj_ts[i], data.ptr))

                H_knn = H_augs[-1]
                for k in range(args.knn_layer):
                    H_knn = torch.sparse.mm(knn_adj_t, H_knn)
                logit_aug_props.append(classifier(H_knn)[train_idx])

            loss = 0
            for i in range(args.aug_num):
                loss += F.nll_loss(logit_aug_props[i], data.y[train_idx])
            loss = loss / args.aug_num

            loss = loss + consis_loss(logit_aug_props, temp=args.temp)

            logits = torch.stack(logit_aug_props).mean(dim = 0)

            total_loss += (loss * train_idx.shape[0]).item()
            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y[train_idx].tolist())

    acc_c = f1_score(truth, pred, labels=np.arange(
        0, 2), average=None, zero_division=0)
    acc = (np.array(pred) == np.array(truth)).sum() / len(truth)

    return {'loss': total_loss / (i + 1), 'F1-macro': np.mean(acc_c), 'F1-micro': acc}
